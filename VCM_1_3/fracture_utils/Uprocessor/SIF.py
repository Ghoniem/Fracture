"""Stress-intensity extraction utilities (near-tip fits)."""
from __future__ import annotations
import numpy as np
import math

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

    KI = (float(mu) / (float(kappa) + 1.0)) * float(AI)
    KII = (float(mu) / (float(kappa) + 1.0)) * float(AII)

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
    K_I_rot  = K_I * np.cos(theta/2)**2 + K_II * np.sin(theta)
    K_II_rot = K_II * np.cos(theta) - 0.5 * K_I * np.sin(theta)
    return float(K_I_rot), float(K_II_rot)
