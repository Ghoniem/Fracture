"""Stress-intensity extraction utilities (near-tip fits)."""
from __future__ import annotations
import numpy as np
import math
from fracture_utils.Usolver import material
from fracture_utils.Usolver.network import CrackNetworkV4 as CrackNetworkV4
from fracture_utils.Usolver.material import Material, AppliedStress
from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4


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
    material = Material(E=200e9, nu=0.3, plane_stress=False)
    kappa = material.kappa
    mu = material.mu
    pref = (kappa + 1.0) / (2.0 * mu) * np.sqrt(2.0 / np.pi)
    KI = AI / pref
    KII = AII / pref
    
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

def pk_sif_from_window(
    pk,
    *,
    edge_index: int = 0,
    at: str = "end",
    window_panels: int = 12,
    window_frac: float = 0.5,
    exclude_self: bool = True,
    plane_strain: bool = True,
    theta_rot: float = 0.0,
):
    """
    Compute (KI, KII) from the Peach–Koehler (PK) window force.

    IMPORTANT (frame consistency)
    -----------------------------
    The PK output is a *vector* force F in the global frame. To compare against a
    target convention/frame (e.g. Cotterell–Rice "global" frame), you must rotate
    the *force* (and the local basis vectors) BEFORE converting (J1,J2)->(KI,KII).
    Rotating (KI,KII) afterwards is not equivalent in mixed mode.

    Parameters
    ----------
    pk : PKProcessor
        Instance constructed from a DCEResultsNetworkV4 object.
    edge_index : int
        Crack edge index (as used by results reconstructor).
    at : {'end','start'}
        Which tip to use.
    window_panels : int
        Number of panels to include in the PK force window (preferred).
    window_frac : float
        Fractional window length fallback (used if window_panels not supported).
    exclude_self : bool
        Whether to exclude self-stress in the stress evaluation.
    plane_strain : bool
        If True, use E' = E/(1-nu^2). If False, use E' = E (plane stress).
    theta_rot : float
        Optional *global* rotation angle (radians) applied to BOTH the PK force
        vector and the local basis (ex,ey) prior to projection. Use this to map
        your computational/global frame into the comparison frame.

    Returns
    -------
    KI, KII : floats
    meta : dict
        Diagnostics: raw/rotated forces, bases, J1,J2, Eprime.
    """
    # -------------------------------
    # 1) PK force in global frame
    # -------------------------------
    try:
        F = pk.F_PK_window(
            edge_index=int(edge_index),
            at=at,
            exclude_self=exclude_self,
            window_panels=window_panels,
            window_frac=window_frac,
        )
    except TypeError:
        # backward compatibility
        F = pk.F_PK_window(
            edge_index=int(edge_index),
            at=at,
            exclude_self=exclude_self,
            window_panels=window_panels,
        )

    if F is None:
        return float("nan"), float("nan"), {"reason": "F_PK_window returned None"}

    F = np.asarray(F, float).reshape(2,)

    # -------------------------------
    # 2) Local basis from reconstructor
    # -------------------------------
    _, _, _, extra = pk.res.reconstruct_cod_csd_panel_midpoints(
        edge_index=int(edge_index),
        enforce_global_tip_zero=False,
    )
    extra = dict(extra) if isinstance(extra, dict) else {}

    ex = np.asarray(extra.get("ex", [1.0, 0.0]), float).reshape(2,)
    ey = np.asarray(extra.get("ey", [0.0, 1.0]), float).reshape(2,)

    # -------------------------------
    # 3) Optional rotation (vector rotation) BEFORE J->K
    # -------------------------------
    def _rot(v, th):
        c = float(np.cos(th)); s = float(np.sin(th))
        return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]], float)

    theta_rot = float(theta_rot)
    F_use  = _rot(F,  theta_rot) if abs(theta_rot) > 0 else F.copy()
    ex_use = _rot(ex, theta_rot) if abs(theta_rot) > 0 else ex.copy()
    ey_use = _rot(ey, theta_rot) if abs(theta_rot) > 0 else ey.copy()

    # normalize basis defensively
    nex = float(np.linalg.norm(ex_use))
    ney = float(np.linalg.norm(ey_use))
    if nex > 0:
        ex_use /= nex
    if ney > 0:
        ey_use /= ney

    # -------------------------------
    # 4) Project PK force into local crack frame: J1 (tangent), J2 (normal)
    # -------------------------------
    J1 = float(np.dot(F_use, ex_use))
    J2 = float(np.dot(F_use, ey_use))

    # -------------------------------
    # 5) Convert (J1,J2)->(KI,KII)
    # -------------------------------
    E = float(pk.res.calc.material.E)
    nu = float(pk.res.calc.material.nu)
    Eprime = E / (1.0 - nu**2) if bool(plane_strain) else E

    # Relations (plane strain/stress via Eprime):
    #   J1 = (KI^2 + KII^2)/Eprime
    #   J2 = -(2*KI*KII)/Eprime
    A = Eprime * J1
    C = -0.5 * Eprime * J2

    disc = float(max(A*A - 4.0*C*C, 0.0))
    D = float(np.sqrt(disc))

    KI2  = float(max(0.5 * (A + D), 0.0))
    KII2 = float(max(0.5 * (A - D), 0.0))

    KI  = float(np.sqrt(KI2))
    KII = float(np.sign(C) * np.sqrt(KII2))  # enforce KI*KII sign via C

    meta = dict(
        F_raw=np.asarray(F, float),
        F=np.asarray(F_use, float),
        ex_raw=np.asarray(ex, float),
        ey_raw=np.asarray(ey, float),
        ex=np.asarray(ex_use, float),
        ey=np.asarray(ey_use, float),
        J1=J1,
        J2=J2,
        Eprime=Eprime,
        plane_strain=bool(plane_strain),
        theta_rot=theta_rot,
        window_panels=int(window_panels),
        window_frac=float(window_frac),
        exclude_self=bool(exclude_self),
        edge_index=int(edge_index),
        at=str(at),
    )
    return KI, KII, meta

# -------------------------
# Cotterell–Rice analytical SIFs (paper form with alpha/2)
# IMPORTANT: for arc crack use a_arc = R*alpha (half arc length)
# ------------------------

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
def rotate_sifs(K_I, K_II, theta):
    K_I_rot  = K_I * np.cos(theta/2)**2 - K_II * np.sin(theta)
    K_II_rot = K_II * np.cos(theta) + 0.5 * K_I * np.sin(theta)
    return float(K_I_rot), float(K_II_rot)
# -------------------------
# Euclidean tip-fit helper
# -------------------------
def sif_from_cod_fit_euclid_tip(xy_mid, COD, CSD, p_tip, *, a_fit, mu,
                                rmin_frac=1e-6, rmax_frac=0.06, min_pts=10, two_term=True):
    xy_mid = np.asarray(xy_mid, float)
    COD = np.asarray(COD, float).reshape(-1)
    CSD = np.asarray(CSD, float).reshape(-1)
    p_tip = np.asarray(p_tip, float).reshape(2,)

    r = np.linalg.norm(xy_mid - p_tip[None, :], axis=1)
    rmin = max(float(rmin_frac) * float(a_fit), 0.0)
    rmax = float(rmax_frac) * float(a_fit)

    m = (r >= rmin) & (r <= rmax) & np.isfinite(r) & np.isfinite(COD) & np.isfinite(CSD)
    rr = r[m]
    if rr.size < min_pts:
        raise ValueError(
            f"Euclid tip-fit window too small: rr.size={rr.size}. "
            f"Try increasing rmax_frac (currently {rmax_frac}) or lowering min_pts."
        )

    order = np.argsort(rr)
    rr = rr[order]
    CODw = COD[m][order]
    CSDw = CSD[m][order]

    sr = np.sqrt(rr)
    if two_term:
        X = np.vstack([sr, rr*sr]).T
    else:
        X = sr[:, None]

    Ac, *_ = np.linalg.lstsq(X, CODw, rcond=None)
    As, *_ = np.linalg.lstsq(X, CSDw, rcond=None)

    A_COD = float(Ac[0])
    A_CSD = float(As[0])

    # CORRECT Leading-term mapping (includes kappa)
    # Near-tip asymptotic: COD ~ (kappa+1)/(4*mu) * sqrt(2*r/pi) * K
    kappa = material.kappa
    mu = material.mu
    pref = (kappa + 1.0) / (2.0 * mu) * np.sqrt(2.0 / np.pi)
    KI = A_COD / pref
    KII = A_CSD / pref

    meta = dict(rr=rr, rmin=rmin, rmax=rmax, A_COD=A_COD, A_CSD=A_CSD)
    return float(KI), float(KII), meta

# -------------------------
# Solver wrapper (TIP segment!)
# -------------------------
def solve_K_polyline(
    net, V, ne_half, edge_index_use, a_fit_global, *,
    E, nu, plane, sig_xx, sig_yy, sig_xy, base_knobs,
    fit_frac=0.20,
    min_segs=2,
    rmax_frac=0.30,
    rmin_frac=1e-6,
    min_pts=8,
    two_term=True,
    theta_rot_override=None,
):
    """
    Polyline arc (or general polyline) SIF extraction using existing fit utilities.
    Implements a multi-segment near-tip window for the Euclidean tip-fit, without
    introducing new fit functions.

    Parameters match the notebook validation cell call.
    """

    # ---- Material ----
    plane_l = str(plane).lower().strip()
    plane_stress = (plane_l in ("stress", "plane_stress", "planestress"))
    material = Material(E=float(E), nu=float(nu), plane_stress=bool(plane_stress))

    # ---- Applied stress ----
    def _make_applied(sx, sy, txy):
        try:
            # Prefer keyword construction if supported
            for kwset in (
                dict(sxx=sx, syy=sy, sxy=txy),
                dict(sig_xx=sx, sig_yy=sy, sig_xy=txy),
                dict(sx=sx, sy=sy, txy=txy),
                dict(sigma_xx=sx, sigma_yy=sy, sigma_xy=txy),
            ):
                try:
                    return AppliedStress(**kwset)
                except TypeError:
                    pass
            return AppliedStress(sx, sy, txy)
        except Exception:
            return AppliedStress(sx, sy, txy)

    applied = _make_applied(float(sig_xx), float(sig_yy), float(sig_xy))

    # ---- Solver ----
    # Your DCENetworkStaticV4 constructor: (material, network, applied)
    calc = DCENetworkStaticV4(material, net, applied)

    # base_knobs belong to solve(), not __init__()
    sol = calc.solve(ne_half=int(ne_half), **dict(base_knobs))
    res = DCEResultsNetworkV4(calc, sol)

    edge_index_use = int(edge_index_use)

    # Tip reconstruction for tip frame and p_tip
    x_tip, COD_tip, CSD_tip, extra_tip = res.reconstruct_cod_csd_panel_midpoints(
        edge_index=edge_index_use,
        enforce_global_tip_zero=True,
    )
    extra_tip = dict(extra_tip) if isinstance(extra_tip, dict) else {}

    V_arr = np.asarray(V, float)
    if V_arr.ndim != 2 or V_arr.shape[1] < 3:
        raise ValueError("V must be array (n,>=3) with columns [id,x,y,...]")
    p_tip = np.asarray(V_arr[-1, 1:3], float)

    # Tip tangent ex_tip: from extra if present, else from last segment geometry
    P = np.asarray(V_arr[:, 1:3], float)
    ex_tip = extra_tip.get("ex", None)
    if ex_tip is None:
        ex_tip = P[-1] - P[-2]
    ex_tip = np.asarray(ex_tip, float).reshape(2,)
    ex_tip /= max(np.linalg.norm(ex_tip), 1e-30)
    ey_tip = np.array([-ex_tip[1], ex_tip[0]], float)
    theta_tip = float(np.arctan2(ex_tip[1], ex_tip[0]))

    mu = float(E) / (2.0 * (1.0 + float(nu)))

    # ---- Multi-segment window selection by arclength ----
    seglen = np.linalg.norm(P[1:] - P[:-1], axis=1)
    L = float(np.sum(seglen))
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Invalid polyline total length")

    if edge_index_use < 0 or edge_index_use >= len(seglen):
        raise IndexError(f"edge_index_use={edge_index_use} out of range for {len(seglen)} polyline segments")

    target = float(fit_frac) * L
    acc = 0.0
    j0 = edge_index_use
    while j0 > 0 and (acc < target or (edge_index_use - j0 + 1) < int(min_segs)):
        acc += float(seglen[j0])
        j0 -= 1

    xs_all, COD_all, CSD_all = [], [], []

    for j in range(j0, edge_index_use + 1):
        xj, CODj, CSDj, extraj = res.reconstruct_cod_csd_panel_midpoints(
            edge_index=int(j),
            enforce_global_tip_zero=(int(j) == edge_index_use),
        )
        extraj = dict(extraj) if isinstance(extraj, dict) else {}

        CODj = np.asarray(CODj, float).reshape(-1)
        CSDj = np.asarray(CSDj, float).reshape(-1)

        # Segment tangent ex_j: from extra or geometry
        ex_j = extraj.get("ex", None)
        if ex_j is None:
            ex_j = P[j+1] - P[j]
        ex_j = np.asarray(ex_j, float).reshape(2,)
        ex_j /= max(np.linalg.norm(ex_j), 1e-30)
        ey_j = np.array([-ex_j[1], ex_j[0]], float)

        # Midpoint coordinates: xy_mid if provided, else geometric midpoint repeated
        xy_mid = extraj.get("xy_mid", None)
        if xy_mid is None:
            pm = 0.5 * (P[j] + P[j+1])
            xy_mid = np.repeat(pm[None, :], CODj.size, axis=0)
        else:
            xy_mid = np.asarray(xy_mid, float)

        # edge-local -> global jump vector
        jump_global = CODj[:, None] * ey_j[None, :] + CSDj[:, None] * ex_j[None, :]

        # global -> tip frame
        COD_tipframe = jump_global @ ey_tip
        CSD_tipframe = jump_global @ ex_tip

        xs_all.append(xy_mid)
        COD_all.append(COD_tipframe)
        CSD_all.append(CSD_tipframe)

    xy_mid_all = np.vstack(xs_all)
    COD_fit = np.concatenate(COD_all)
    CSD_fit = np.concatenate(CSD_all)

    # ---- Existing Euclidean tip fit ----
    KI, KII, meta_fit = sif_from_cod_fit_euclid_tip(
        xy_mid=xy_mid_all,
        COD=COD_fit,
        CSD=CSD_fit,
        p_tip=p_tip,
        a_fit=float(a_fit_global),
        mu=float(mu),
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
        p_tip=p_tip,
        fit_frac=float(fit_frac),
        seg_window=(int(j0), int(edge_index_use)),
        meta_fit=meta_fit,
    )
    return KI, KII, KI_CR, KII_CR, meta


# ----------------------------
# Robust Euclidean tip-fit
# ----------------------------
def euclid_tip_fit_from_edge(
    res,
    *,
    edge_index: int,
    tip_xy,
    a_fit: float,
    rmin_frac: float = 1e-6,
    rmax_frac: float = 0.08,
    min_pts: int = 10,
    two_term: bool = False,
    expand=(1.0, 1.5, 2.0, 3.0),
    theta: float = 0.0,
    material=None,
    prefactor: str = "code",  # "code" uses (kappa+1)/(2mu)*sqrt(2/pi); "canonical" uses 4/mu*sqrt(1/(2pi))
):
    """
    Robust near-tip SIF extraction for *any* edge in a network using Euclidean distance r=||x-x_tip||.

    Parameters
    ----------
    res : DCEResultsNetworkV4
    edge_index : int
    tip_xy : array-like (2,)
        Tip point in global coordinates.
    a_fit : float
        Characteristic length for scaling rmin/rmax (e.g. branch length l).
    theta : float
        Optional rotation angle applied to (KI,KII) using rotate_sifs(). Use 0.0 unless you are sure.
    material : Material or None
        If None, uses res.calc.material. If provided, uses that.
    prefactor : {"code","canonical"}
        - "code": pref = (kappa+1)/(2mu)*sqrt(2/pi)
        - "canonical": pref = 4/mu*sqrt(1/(2pi))
    """
    # ----- material constants -----
    mat = material if material is not None else getattr(res.calc, "material", None)
    if mat is None:
        raise ValueError("euclid_tip_fit_from_edge: could not obtain material. Pass material=... or ensure res.calc.material exists.")

    mu = float(getattr(mat, "mu", None) or (mat.E / (2.0 * (1.0 + mat.nu))))
    # kappa depends on plane stress/strain convention of your Material class
    kappa = float(getattr(mat, "kappa", None))
    if not np.isfinite(kappa):
        # fallback
        nu = float(mat.nu)
        plane_stress = bool(getattr(mat, "plane_stress", False))
        kappa = (3.0 - nu) / (1.0 + nu) if plane_stress else (3.0 - 4.0 * nu)

    # ----- reconstruct -----
    x, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(
        edge_index=int(edge_index),
        enforce_global_tip_zero=True,
    )
    extra = dict(extra) if isinstance(extra, dict) else {}

    xy_mid = extra.get("xy_mid", None)
    if xy_mid is None:
        raise KeyError("extra['xy_mid'] missing from reconstruct_cod_csd_panel_midpoints; needed for Euclidean windowing.")

    xy_mid = np.asarray(xy_mid, float)
    COD = np.asarray(COD, float).reshape(-1)
    CSD = np.asarray(CSD, float).reshape(-1)
    tip_xy = np.asarray(tip_xy, float).reshape(2,)

    r = np.linalg.norm(xy_mid - tip_xy[None, :], axis=1)

    a_fit = float(a_fit)
    rmin = max(float(rmin_frac) * a_fit, 0.0)

    mask = None
    rr = None
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

    sr = np.sqrt(rr)
    if two_term:
        X = np.vstack([sr, rr * sr]).T
    else:
        X = sr[:, None]

    Ac, *_ = np.linalg.lstsq(X, CODw, rcond=None)
    As, *_ = np.linalg.lstsq(X, CSDw, rcond=None)

    A_COD = float(Ac[0])
    A_CSD = float(As[0])

    prefactor = str(prefactor).lower().strip()
    if prefactor == "code":
        pref = (kappa + 1.0) / (2.0 * mu) * np.sqrt(2.0 / np.pi)
    elif prefactor == "canonical":
        pref = 4.0 / mu * np.sqrt(1.0 / (2.0 * np.pi))
    else:
        raise ValueError("prefactor must be 'code' or 'canonical'")

    KI = A_COD / pref
    KII = A_CSD / pref

    # optional rotation to comparison frame (usually leave theta=0 for branched crack tips)
    KI_rot, KII_rot = rotate_sifs(KI, KII, float(theta))

    meta = dict(
        rr_size=int(rr.size),
        rmin=rmin,
        rmax=rmax_used,
        A_COD=A_COD,
        A_CSD=A_CSD,
        mu=mu,
        kappa=kappa,
        prefactor=prefactor,
    )
    return float(KI), float(KII), float(KI_rot), float(KII_rot), meta
