"""COD-based stress-intensity factor (SIF) utilities.

This module contains COD/CSD-based near-tip estimators:
- Williams near-tip fit on a straight edge (sif_from_cod_fit)
- Jump-based estimators
- Robust Euclidean (distance-to-tip) fits for network/branched edges

Notes
-----
- All mappings use the same prefactor convention:
    pref = (kappa+1)/(2*mu) * sqrt(2/pi)
  so that KI = A_I / pref, KII = A_II / pref, where COD ~ A_I * sqrt(r/(2pi)).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import math


def rotate_sifs(K_I, K_II, theta):
    """
    Rotate/kink SIFs for a crack extension angle ``theta`` (radians).

    Uses the standard 2D LEFM kink-angle transformation:
        K_I'  = C11*K_I + C12*K_II
        K_II' = C21*K_I + C22*K_II
    with
        C11 = cos(theta/2)^3
        C12 = -(3/2) * sin(theta) * cos(theta/2)
        C21 = (1/2) * sin(theta) * cos(theta/2)
        C22 = cos(theta/2) * (1 - 3*sin(theta/2)^2)
    """
    th = float(theta)
    c2 = np.cos(th / 2.0)
    s2 = np.sin(th / 2.0)
    s = np.sin(th)

    C11 = c2 ** 3
    C12 = -1.5 * s * c2
    C21 = 0.5 * s * c2
    C22 = c2 * (1.0 - 3.0 * s2 * s2)

    K_I_rot = C11 * K_I + C12 * K_II
    K_II_rot = C21 * K_I + C22 * K_II
    return float(K_I_rot), float(K_II_rot)

# -------------------------
# POLYLINE tip-fit helpers  
# -------------------------

def solve_K_polyline(
    net,
    V,
    ne_half,
    edge_index_use,
    a_fit_global,
    *,
    E,
    nu,
    plane,
    sig_xx,
    sig_yy,
    sig_xy,
    base_knobs,
    fit_frac: float = 0.20,
    min_segs: int = 2,
    rmax_frac: float = 0.30,
    rmin_frac: float = 1e-6,
    min_pts: int = 8,
    two_term: bool = True,
    theta_rot_override=None,
):
    """
    Polyline SIF extraction using a multi-segment Euclidean (distance-to-tip) COD/CSD fit.

    This function was historically provided by Uprocessor/SIF.py and is retained
    here as part of the SIF.py -> {SIF_cod,SIF_pk} split.
    """

    # ---- Local imports (avoid top-level circularity) ----
    from fracture_utils.Usolver.material import Material, AppliedStress
    from fracture_utils.Usolver.parametrization import DCENetworkStaticV4
    from fracture_utils.Uprocessor.results import DCEResultsNetworkV4

    plane_l = str(plane).lower().strip()
    plane_stress = plane_l in ("stress", "plane_stress", "planestress")

    mat = Material(E=float(E), nu=float(nu), plane_stress=bool(plane_stress))
    mu = float(getattr(mat, "mu", float(E) / (2.0 * (1.0 + float(nu)))))
    kappa = float(getattr(mat, "kappa", np.nan))
    if not np.isfinite(kappa):
        nu_ = float(nu)
        kappa = (3.0 - nu_) / (1.0 + nu_) if plane_stress else (3.0 - 4.0 * nu_)

    # applied stress (support multiple constructor signatures)
    sx = float(sig_xx)
    sy = float(sig_yy)
    txy = float(sig_xy)
    applied = None
    for kw in (
        dict(sigma_xx=sx, sigma_yy=sy, sigma_xy=txy),
        dict(sig_xx=sx, sig_yy=sy, sig_xy=txy),
        dict(sxx=sx, syy=sy, sxy=txy),
        dict(sx=sx, sy=sy, txy=txy),
    ):
        try:
            applied = AppliedStress(**kw)
            break
        except TypeError:
            continue
    if applied is None:
        applied = AppliedStress(sx, sy, txy)

    calc = DCENetworkStaticV4(mat, net, applied)
    sol = calc.solve(ne_half=int(ne_half), **dict(base_knobs))
    res = DCEResultsNetworkV4(calc, sol)

    edge_index_use = int(edge_index_use)

    # Tip frame: use reconstructor's ex if available; else last geometry segment
    _, _, _, extra_tip = res.reconstruct_cod_csd_panel_midpoints(
        edge_index=edge_index_use,
        enforce_global_tip_zero=True,
    )
    extra_tip = dict(extra_tip) if isinstance(extra_tip, dict) else {}

    V_arr = np.asarray(V, float)
    if V_arr.ndim != 2 or V_arr.shape[1] < 3:
        raise ValueError("V must be array (n,>=3) with columns [id,x,y,...]")
    P = np.asarray(V_arr[:, 1:3], float)
    if P.shape[0] < 2:
        raise ValueError("V must contain at least 2 points to define a polyline.")

    p_tip = np.asarray(V_arr[-1, 1:3], float)

    ex_tip = extra_tip.get("ex", None)
    if ex_tip is None:
        ex_tip = P[-1] - P[-2]
    ex_tip = np.asarray(ex_tip, float).reshape(2,)
    exn = float(np.linalg.norm(ex_tip))
    if exn <= 0:
        ex_tip = np.array([1.0, 0.0], float)
        exn = 1.0
    ex_tip /= exn
    ey_tip = np.array([-ex_tip[1], ex_tip[0]], float)
    theta_tip = float(np.arctan2(ex_tip[1], ex_tip[0]))

    # ---- Multi-segment window selection by arclength ----
    seglen = np.linalg.norm(P[1:] - P[:-1], axis=1)
    L = float(np.sum(seglen))
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Invalid polyline total length")

    if edge_index_use < 0 or edge_index_use >= len(seglen):
        raise IndexError(
            f"edge_index_use={edge_index_use} out of range for {len(seglen)} polyline segments"
        )

    target = float(fit_frac) * L
    acc = 0.0
    j0 = edge_index_use
    # walk backwards from tip segment
    while j0 > 0 and (acc < target or (edge_index_use - j0 + 1) < int(min_segs)):
        acc += float(seglen[j0])
        j0 -= 1

    xy_all = []
    COD_all = []
    CSD_all = []

    for j in range(j0, edge_index_use + 1):
        _, CODj, CSDj, extraj = res.reconstruct_cod_csd_panel_midpoints(
            edge_index=int(j),
            enforce_global_tip_zero=(int(j) == edge_index_use),
        )
        extraj = dict(extraj) if isinstance(extraj, dict) else {}

        CODj = np.asarray(CODj, float).reshape(-1)
        CSDj = np.asarray(CSDj, float).reshape(-1)

        # Segment basis
        ex_j = extraj.get("ex", None)
        if ex_j is None:
            ex_j = P[j + 1] - P[j]
        ex_j = np.asarray(ex_j, float).reshape(2,)
        exjn = float(np.linalg.norm(ex_j))
        if exjn <= 0:
            ex_j = np.array([1.0, 0.0], float)
            exjn = 1.0
        ex_j /= exjn
        ey_j = np.array([-ex_j[1], ex_j[0]], float)

        # Midpoint coordinates
        xy_mid = extraj.get("xy_mid", None)
        if xy_mid is None:
            pm = 0.5 * (P[j] + P[j + 1])
            xy_mid = np.repeat(pm[None, :], CODj.size, axis=0)
        else:
            xy_mid = np.asarray(xy_mid, float)

        # segment-local jumps -> global vector
        jump_global = CODj[:, None] * ey_j[None, :] + CSDj[:, None] * ex_j[None, :]

        # global -> tip frame scalars
        COD_tipframe = jump_global @ ey_tip
        CSD_tipframe = jump_global @ ex_tip

        xy_all.append(xy_mid)
        COD_all.append(COD_tipframe)
        CSD_all.append(CSD_tipframe)

    xy_mid_all = np.vstack(xy_all)
    COD_fit = np.concatenate(COD_all)
    CSD_fit = np.concatenate(CSD_all)

    KI, KII, meta_fit = sif_from_cod_fit_euclid_arrays(
        xy_mid=xy_mid_all,
        COD=COD_fit,
        CSD=CSD_fit,
        p_tip=p_tip,
        a_fit=float(a_fit_global),
        mu=mu,
        kappa=kappa,
        rmin_frac=float(rmin_frac),
        rmax_frac=float(rmax_frac),
        min_pts=int(min_pts),
        two_term=bool(two_term),
    )

    theta_rot = float(theta_rot_override) if (theta_rot_override is not None) else (-2.0 * theta_tip)
    KI_CR, KII_CR = rotate_sifs(KI, KII, theta_rot)

    meta = dict(
        theta_tip=theta_tip,
        theta_rot=theta_rot,
        p_tip=np.asarray(p_tip, float),
        fit_frac=float(fit_frac),
        seg_window=(int(j0), int(edge_index_use)),
        meta_fit=dict(meta_fit),
    )
    return float(KI), float(KII), float(KI_CR), float(KII_CR), meta

# -------------------------
# Euclidean tip-fit helper
# -------------------------


def cotterell_rice_K(alpha_total, a_arc, sxx, syy, sxy):
    A0 = 0.5 * (syy + sxx)
    D0 = 0.5 * (syy - sxx)

    s = np.sin(alpha_total/2)
    c = np.cos(alpha_total/2)

    bracket = A0 - D0 * (s**2) * (c**2)

    KI  = np.sqrt(np.pi * a_arc) * (
        bracket * (c / (1.0 + s**2))
        + D0 * np.cos(3.0 * alpha_total/2)
        - sxy * (np.sin(3.0 * alpha_total/2) + s**3)
    )

    KII = np.sqrt(np.pi * a_arc) * (
        bracket * (s / (1.0 + s**2))
        + D0 * np.sin(3.0 * alpha_total/2)
        + sxy * (np.cos(3.0 * alpha_total/2) + c * s**2)
    )
    return float(KI), float(KII)

# -------------------------
# SIF rotation formulas
# -------------------------


def sif_from_cod_fit(
    x: np.ndarray,
    COD: np.ndarray,
    CSD: np.ndarray,
    *,
    a: float,
    mu: float,
    kappa: float,
    tip: str = "right",
    window: str = "auto",
    rho_min: float = 5e-4,
    rho_max: float = 5e-2,
    c1: float = 10.0,
    c2: float = 0.20,
    n_fit: int = 120,
    two_term: bool = True,
    dx_tip: float | None = None,
    min_pts: int = 16,
):
    x = np.asarray(x, float)
    COD = np.asarray(COD, float)
    CSD = np.asarray(CSD, float)
    a = float(a)

    tip_l = str(tip).lower()
    if tip_l == "right":
        r = a - x
    elif tip_l == "left":
        r = a + x
    else:
        raise ValueError("tip must be 'left' or 'right'")

    if dx_tip is None:
        xx = np.sort(x[np.isfinite(x)])
        if xx.size < 3:
            dx_tip = 0.05 * a
        else:
            slab = xx[-min(25, xx.size):] if tip_l == "right" else xx[:min(25, xx.size)]
            dxx = np.diff(slab)
            dxx = dxx[np.isfinite(dxx) & (dxx > 0)]
            dx_tip = float(np.min(dxx)) if dxx.size else 0.05 * a
    dx_tip = float(dx_tip)

    wmode = str(window).lower()
    if wmode == "auto":
        r_min = max(float(rho_min) * a, float(c1) * dx_tip)
        r_max = min(float(rho_max) * a, float(c2) * math.sqrt(max(a * dx_tip, 1e-30)))
        if r_max <= 1.05 * r_min:
            r_max = 2.5 * r_min
    elif wmode == "fixed":
        r_min = float(rho_min) * a
        r_max = float(rho_max) * a
    else:
        raise ValueError("window must be 'auto' or 'fixed'")

    max_expand_iters = 12
    expand = 1.6
    relax = 0.8
    r_max_cap = float(rho_max) * a

    for _ in range(max_expand_iters):
        m = np.isfinite(r) & np.isfinite(COD) & np.isfinite(CSD) & (r > r_min) & (r < r_max)
        if np.count_nonzero(m) >= int(min_pts):
            break
        r_max = min(r_max * expand, r_max_cap)
        if r_max >= 0.999 * r_max_cap:
            r_min = max(r_min * relax, 1e-10 * a)

    m = np.isfinite(r) & np.isfinite(COD) & np.isfinite(CSD) & (r > r_min) & (r < r_max)
    rr = r[m]
    CODw = COD[m]
    CSDw = CSD[m]

    if rr.size < 4:
        raise ValueError(
            f"Fit window too small after expansion: rr.size={rr.size}. "
            f"Try increasing rho_max (currently {rho_max}) or increasing n_pts."
        )

    order = np.argsort(rr)
    rr = rr[order][: int(n_fit)]
    CODw = CODw[order][: int(n_fit)]
    CSDw = CSDw[order][: int(n_fit)]

    phi = np.sqrt(rr / (2.0 * np.pi))

    if two_term:
        rbar = rr / (2.0 * np.pi)
        X = np.column_stack([phi, phi * rbar])
        AI, BI = np.linalg.lstsq(X, CODw, rcond=None)[0]
        AII, BII = np.linalg.lstsq(X, CSDw, rcond=None)[0]
    else:
        AI = float(np.dot(phi, CODw) / np.dot(phi, phi))
        AII = float(np.dot(phi, CSDw) / np.dot(phi, phi))
        BI = 0.0
        BII = 0.0
    mu = float(mu)
    kappa = float(kappa)    
    KI = (mu/(kappa + 1.0)) * AI
    KII = (mu/(kappa + 1.0)) * AII
    
    # KI = (float(mu) / (float(kappa) + 1.0)) * float(AI)
    # KII = (float(mu) / (float(kappa) + 1.0)) * float(AII)

    meta = dict(
        window=wmode,
        r_min=float(r_min),
        r_max=float(r_max),
        dx_tip=float(dx_tip),
        n_fit=int(len(rr)),
        two_term=bool(two_term),
        A_I=float(AI),
        B_I=float(BI),
        A_II=float(AII),
        B_II=float(BII),
    )
    return KI, KII, meta



def estimate_K_from_jump_near_tip(x, jump, a, Eprime, side="right", frac_window=0.08):
    x = np.asarray(x, float)
    jump = np.asarray(jump, float)
    a = float(a)

    if str(side).lower().startswith("r"):
        r = a - x
    else:
        r = a + x

    rmax = float(frac_window) * a
    m = np.isfinite(r) & np.isfinite(jump) & (r > 0) & (r < rmax)
    if np.count_nonzero(m) < 20:
        return np.nan

    X = np.sqrt(r[m])
    Y = jump[m]
    A = float(np.dot(X, Y) / np.dot(X, X))
    K = A * float(Eprime) * np.sqrt(2.0 * np.pi) / 8.0
    return K



def estimate_KI_KII_from_jumps_near_tip(
    x,
    COD,
    CSD,
    a,
    Eprime,
    side="right",
    frac_window=0.08,
):
    """
    Near-tip jump estimator for both Mode I and Mode II using:
        KI  from COD (opening jump)
        KII from CSD (sliding jump)

    Uses the same scalar estimator internally to maintain consistency.

    Returns
    -------
    KI_jump, KII_jump : floats
    """
    KI_jump = estimate_K_from_jump_near_tip(
        x=x, jump=COD, a=a, Eprime=Eprime, side=side, frac_window=frac_window
    )
    KII_jump = estimate_K_from_jump_near_tip(
        x=x, jump=CSD, a=a, Eprime=Eprime, side=side, frac_window=frac_window
    )
    return KI_jump, KII_jump



def sif_from_cod_fit_euclid_arrays(
    *,
    xy_mid: np.ndarray,
    COD: np.ndarray,
    CSD: np.ndarray,
    p_tip: np.ndarray,
    a_fit: float,
    mu: float,
    kappa: float,
    rmin_frac: float = 1e-6,
    rmax_frac: float = 0.08,
    min_pts: int = 10,
    two_term: bool = False,
    expand: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
) -> Tuple[float, float, Dict[str, Any]]:
    """Euclidean near-tip fit using r = ||x - x_tip||.

    Fits COD and CSD against Williams leading form in terms of
        phi = sqrt(r/(2*pi))
    optionally with a second term phi*rbar where rbar = r/(2*pi).

    Returns KI, KII using the same prefactor convention as sif_from_cod_fit:
        pref = (kappa+1)/(2*mu) * sqrt(2/pi)
        KI = A_I / pref, KII = A_II / pref
    """
    xy_mid = np.asarray(xy_mid, float)
    COD = np.asarray(COD, float).reshape(-1)
    CSD = np.asarray(CSD, float).reshape(-1)
    p_tip = np.asarray(p_tip, float).reshape(2,)

    r = np.linalg.norm(xy_mid - p_tip[None, :], axis=1)
    a_fit = float(a_fit)

    rmin = max(float(rmin_frac) * a_fit, 0.0)

    rr = None
    mask = None
    rmax_used = None
    for fac in expand:
        rmax_used = float(rmax_frac) * float(fac) * a_fit
        mask = (r >= rmin) & (r <= rmax_used) & np.isfinite(r) & np.isfinite(COD) & np.isfinite(CSD)
        rr = r[mask]
        if rr.size >= int(min_pts):
            break

    if rr is None or rr.size < int(min_pts):
        raise ValueError(
            f"Euclid tip-fit window too small: rr.size={0 if rr is None else rr.size}. "
            f"Try increasing rmax_frac (currently {rmax_frac}) or lowering min_pts."
        )

    order = np.argsort(rr)
    rr = rr[order]
    CODw = COD[mask][order]
    CSDw = CSD[mask][order]

    phi = np.sqrt(rr / (2.0 * np.pi))

    if two_term:
        rbar = rr / (2.0 * np.pi)
        X = np.column_stack([phi, phi * rbar])
        AI, BI = np.linalg.lstsq(X, CODw, rcond=None)[0]
        AII, BII = np.linalg.lstsq(X, CSDw, rcond=None)[0]
    else:
        AI = float(np.dot(phi, CODw) / np.dot(phi, phi))
        AII = float(np.dot(phi, CSDw) / np.dot(phi, phi))
        BI = 0.0
        BII = 0.0

    mu = float(mu)
    kappa = float(kappa)    
    KI = (mu/(kappa + 1.0)) * AI
    KII = (mu/(kappa + 1.0)) * AII

    meta = dict(
        rr_size=int(rr.size),
        rmin=float(rmin),
        rmax=float(rmax_used),
        two_term=bool(two_term),
        A_I=float(AI),
        B_I=float(BI),
        A_II=float(AII),
        B_II=float(BII),
    )
    return KI, KII, meta



# ---------------------------------------------------------------------
# Near-tip oversampling helper (panel-midpoint refinement without re-solve)
# ---------------------------------------------------------------------
def _oversample_midpoint_arrays(
    xy: np.ndarray,
    COD: np.ndarray,
    CSD: np.ndarray,
    npts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Increase sampling density along an ordered set of panel-midpoint samples.

    This is a lightweight, solver-agnostic refinement: it linearly interpolates
    between consecutive midpoint samples to create additional points per interval.
    It does NOT change the solve (ne_half), only the regression sample set used
    in Euclid COD->SIF fits.

    Parameters
    ----------
    xy : (N,2) array
        Ordered midpoint coordinates along the edge.
    COD, CSD : (N,) arrays
        Ordered midpoint COD and CSD samples.
    npts : int
        Points per interval between consecutive midpoints (>=1).
        npts=1 returns inputs unchanged.

    Returns
    -------
    xy_r, COD_r, CSD_r
        Refined arrays with approximately (N-1)*npts + 1 points.
    """
    xy = np.asarray(xy, float)
    COD = np.asarray(COD, float).reshape(-1)
    CSD = np.asarray(CSD, float).reshape(-1)

    if npts <= 1 or xy.shape[0] <= 1:
        return xy, COD, CSD

    N = xy.shape[0]
    xy_list = []
    COD_list = []
    CSD_list = []

    for i in range(N - 1):
        x0 = xy[i]
        x1 = xy[i + 1]
        c0 = float(COD[i])
        c1 = float(COD[i + 1])
        s0 = float(CSD[i])
        s1 = float(CSD[i + 1])

        for k in range(npts):
            t = k / float(npts)
            xy_list.append((1.0 - t) * x0 + t * x1)
            COD_list.append((1.0 - t) * c0 + t * c1)
            CSD_list.append((1.0 - t) * s0 + t * s1)

    xy_list.append(xy[-1])
    COD_list.append(float(COD[-1]))
    CSD_list.append(float(CSD[-1]))

    return np.asarray(xy_list, float), np.asarray(COD_list, float), np.asarray(CSD_list, float)

def euclid_tip_fit_from_edge(
    res,
    *,
    edge_index: int,
    tip_xy,
    a_fit: float,
    material=None,
    theta: float = 0.0,
    rotate: bool = False,
    rmin_frac: float = 1e-6,
    rmax_frac: float = 0.08,
    min_pts: int = 10,
    two_term: bool = False,
    expand: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    expand_pts: Tuple[int, ...] = (1, 3, 5, 9),
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """High-level Euclidean tip-fit on an edge of a network result.

    Parameters
    ----------
    res : DCEResultsNetworkV4
    edge_index : int
        Edge to reconstruct.
    tip_xy : array-like (2,)
        Global coordinate of the tip to fit around.
    a_fit : float
        Length scale for window fractions (typically branch length l or main half-length c).
    material : Material or None
        If None, uses res.calc.material.
    theta : float
        Rotation angle for rotate_sifs (only used if rotate=True).
    rotate : bool
        If True, also return rotated (KI_rot,KII_rot) else they equal (KI,KII).

    Returns
    -------
    KI, KII, KI_out, KII_out, meta
        Where (KI_out,KII_out) are rotated if rotate=True.
    """
    x, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(
        edge_index=int(edge_index),
        enforce_global_tip_zero=True,
    )
    extra = dict(extra) if isinstance(extra, dict) else {}

    xy_mid = extra.get("xy_mid", None)
    if xy_mid is None:
        raise KeyError("extra['xy_mid'] missing from reconstruction; needed for Euclidean tip-fit.")

    mat = material if material is not None else getattr(res.calc, "material", None)
    if mat is None:
        raise ValueError("euclid_tip_fit_from_edge: material not found. Pass material=... or ensure res.calc.material exists.")

    mu = float(getattr(mat, "mu", mat.E / (2.0 * (1.0 + mat.nu))))
    kappa = float(getattr(mat, "kappa", np.nan))
    if not np.isfinite(kappa):
        nu = float(mat.nu)
        plane_stress = bool(getattr(mat, "plane_stress", False))
        kappa = (3.0 - nu) / (1.0 + nu) if plane_stress else (3.0 - 4.0 * nu)


    # ------------------------------------------------------------
    # Adaptive oversampling (without re-solve):
    # If the Euclid window is underpopulated, refine the sample set by
    # linearly interpolating between ordered panel-midpoint samples.
    # ------------------------------------------------------------
    last_err = None
    xy_use = np.asarray(xy_mid, float)
    COD_use = np.asarray(COD, float)
    CSD_use = np.asarray(CSD, float)

    for npts in tuple(int(n) for n in (expand_pts or (1,))):
        if npts < 1:
            continue
        if npts > 1:
            xy_try, COD_try, CSD_try = _oversample_midpoint_arrays(xy_use, COD_use, CSD_use, npts=npts)
        else:
            xy_try, COD_try, CSD_try = xy_use, COD_use, CSD_use

        try:
            KI, KII, meta_fit = sif_from_cod_fit_euclid_arrays(
                xy_mid=np.asarray(xy_try, float),
                COD=np.asarray(COD_try, float),
                CSD=np.asarray(CSD_try, float),
                p_tip=np.asarray(tip_xy, float),
                a_fit=float(a_fit),
                mu=mu,
                kappa=kappa,
                rmin_frac=rmin_frac,
                rmax_frac=rmax_frac,
                min_pts=min_pts,
                two_term=two_term,
                expand=expand,
            )
            meta_fit = dict(meta_fit) if isinstance(meta_fit, dict) else {"meta_fit": meta_fit}
            meta_fit["expand_pts_used"] = int(npts)
            last_err = None
            break
        except ValueError as e:
            last_err = e

    if last_err is not None:
        raise last_err

    if rotate:
        KI_out, KII_out = rotate_sifs(KI, KII, float(theta))
    else:
        KI_out, KII_out = KI, KII

    meta = dict(meta_fit=meta_fit, mu=mu, kappa=kappa, theta=float(theta), rotated=bool(rotate))
    return float(KI), float(KII), float(KI_out), float(KII_out), meta


# =====================================================================
# OO Facade (v2.1): organize SIF utilities into three semantic classes
#   1) DisplacementSIF            : COD/CSD-based estimators
#   2) DisplacementGradientSIF    : jump / gradient-based estimators
#   3) AnalysisSIF                : analytical formulas and transformations
#
# NOTE:
# - Implementations below reuse the existing free functions in this module.
# - Backward-compatible free-function API is preserved (aliases are provided).
# =====================================================================

class DisplacementSIF:
    """
    Displacement-based SIF estimators (COD/CSD fits).

    This is a thin namespace facade over the module-level implementations:
      - sif_from_cod_fit
      - sif_from_cod_fit_euclid_arrays
      - euclid_tip_fit_from_edge
      - solve_K_polyline
    """

    @staticmethod
    def from_cod_fit(*args, **kwargs):
        return sif_from_cod_fit(*args, **kwargs)

    @staticmethod
    def euclid_arrays(*args, **kwargs):
        return sif_from_cod_fit_euclid_arrays(*args, **kwargs)

    @staticmethod
    def euclid_from_edge(*args, **kwargs):
        return euclid_tip_fit_from_edge(*args, **kwargs)

    @staticmethod
    def polyline(*args, **kwargs):
        return solve_K_polyline(*args, **kwargs)


class DisplacementGradientSIF:
    """
    Displacement-gradient / jump-based SIF estimators.

    Wraps:
      - estimate_K_from_jump_near_tip
      - estimate_KI_KII_from_jumps_near_tip
    """

    @staticmethod
    def from_jump(*args, **kwargs):
        return estimate_K_from_jump_near_tip(*args, **kwargs)

    @staticmethod
    def from_cod_csd(*args, **kwargs):
        return estimate_KI_KII_from_jumps_near_tip(*args, **kwargs)


class AnalysisSIF:
    """
    Analytical SIF utilities and transformations.

    Wraps:
      - rotate_sifs
      - cotterell_rice_K
    """

    @staticmethod
    def rotate(*args, **kwargs):
        return rotate_sifs(*args, **kwargs)

    @staticmethod
    def cotterell_rice(*args, **kwargs):
        return cotterell_rice_K(*args, **kwargs)

__all__ = [
    "DisplacementSIF",
    "DisplacementGradientSIF",
    "AnalysisSIF",
    "sif_from_cod_fit",
    "sif_from_cod_fit_euclid_arrays",
    "euclid_tip_fit_from_edge",
    "solve_K_polyline",
    "estimate_K_from_jump_near_tip",
    "estimate_KI_KII_from_jumps_near_tip",
    "rotate_sifs",
    "cotterell_rice_K",
]
