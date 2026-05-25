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


def rotate_sifs(K_I, K_II, theta):
    K_I_rot  = K_I * np.cos(theta/2)**2 - K_II * np.sin(theta)
    K_II_rot = K_II * np.cos(theta) + 0.5 * K_I * np.sin(theta)
    return float(K_I_rot), float(K_II_rot)

# -------------------------
# POLYLINE tip-fit helpers  
# -------------------------

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

    KI, KII, meta_fit = sif_from_cod_fit_euclid_arrays(
        xy_mid=np.asarray(xy_mid, float),
        COD=np.asarray(COD, float),
        CSD=np.asarray(CSD, float),
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

    if rotate:
        KI_out, KII_out = rotate_sifs(KI, KII, float(theta))
    else:
        KI_out, KII_out = KI, KII

    meta = dict(meta_fit=meta_fit, mu=mu, kappa=kappa, theta=float(theta), rotated=bool(rotate))
    return float(KI), float(KII), float(KI_out), float(KII_out), meta
