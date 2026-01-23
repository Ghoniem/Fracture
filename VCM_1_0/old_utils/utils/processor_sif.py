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

    if rr.size < 8:
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


def sih_circular_sector_sif_biaxial(alpha, sigma, rho=1.0, tip=+1):
    """
    Sih–Paris (1962): Curved crack (circular-sector crack) in an infinite sheet
    under uniform *equal biaxial tension* (sigma in x and y).

    Geometry:
      - crack is a circular arc of radius rho about the origin
      - the arc subtends total angle 2*alpha (alpha in radians), so endpoints at ±alpha
      - chord length: c = 2*rho*sin(alpha)

    Returns:
      KI, KII, c

    Notes:
      - This is Eq. (13) from Sih et al., "Crack-Tip, Stress-Intensity Factors..."
        in the section "Curved Crack in Sheet Under Biaxial Tension".
      - The published derivation is for unit radius; scaling with sqrt(rho) is applied.
      - tip sets the sign convention for KII:
          tip = +1  -> one end (e.g., +alpha)
          tip = -1  -> the opposite end (e.g., -alpha)
        KI is the same magnitude at both ends for this symmetric biaxial case.
    """
    alpha = float(alpha)
    if not (0.0 < alpha < np.pi):
        raise ValueError("alpha must be in (0, pi) radians.")

    # Eq (13) denominator
    denom = 1.0 + (np.sin(alpha/2.0))**2

    # Eq (13) square-root factors
    s1 = np.sin(alpha) * (1.0 + np.cos(alpha)) / 2.0
    s2 = np.sin(alpha) * (1.0 - np.cos(alpha)) / 2.0

    # Numerical safety (should be >=0 for alpha in (0, pi), but guard rounding)
    s1 = max(s1, 0.0)
    s2 = max(s2, 0.0)

    # Unit-radius SIFs from Eq (13)
    k1_unit = (sigma / denom) * np.sqrt(s1)
    k2_unit = (sigma / denom) * np.sqrt(s2)

    # Scale from unit radius to radius rho: K ~ sqrt(length) => multiply by sqrt(rho)
    KI  = k1_unit
    KII = k2_unit

    # chord length (useful for normalization used in later papers/figures)
    c = rho * np.sin(2*alpha)

    return KI, KII, c


def sih_normalized_F(alpha, sigma, rho=1.0, tip=+1):
    """
    Convenience: returns normalized factors
      FI  = KI  / [sigma * sqrt(pi * (c/2))]
      FII = KII / [sigma * sqrt(pi * (c/2))]
    using chord length c = rho*sin(2*alpha).
    """
    KI, KII, c = sih_circular_sector_sif_biaxial(alpha, sigma, rho=rho, tip=tip)
    Fden = sigma * np.sqrt(np.pi * (c/2.0))
    return (KI / Fden), (KII / Fden), c


def cotterell_rice_K_circular_arc(alpha, a, sigma_xx, sigma_yy, sigma_xy):
    """
    Cotterell & Rice (1980), Section 3: exact KI, KII for a circular arc crack
    under a uniform far-field stress state (sigma_xx, sigma_yy, sigma_xy).

    Implements Eqs. (20) and (21) in:
      B. Cotterell and J.R. Rice, "Slightly curved or kinked cracks",
      Int. J. Fracture 16 (1980) 155–169.  (See Section 3, Eqs. 20–21.)

    Parameters
    ----------
    alpha : float
        Total included angle of the circular arc (radians).
        (The equations use sin(alpha/2), cos(alpha/2).)
    a : float
        Radius parameter appearing in (pi*a)^(1/2) prefactor in Eqs. (20)-(21).
        Units: length.
    sigma_xx, sigma_yy, sigma_xy : float
        Remote uniform stress components (same units as desired for K / sqrt(length)).

    Returns
    -------
    KI, KII : floats
        Mode I and Mode II stress intensity factors.

    Notes
    -----
    - Uses the Cotterell–Rice corrected exact solution for the circular arc crack.
    - Angle convention is that used in their Fig. 2 / Eqs. (20)-(21).
    """
    alpha = float(alpha)
    a = float(a)
    if a <= 0.0:
        raise ValueError("a must be > 0.")
    if not (0.0 < alpha < 2.0*np.pi):
        raise ValueError("alpha should be in (0, 2*pi) radians for a proper arc.")

    sh = np.sin(alpha/2.0)
    ch = np.cos(alpha/2.0)

    denom = 1.0 + sh**2

    # Common bracket term in Eqs. (20)-(21):
    # [(σyy+σxx)/2 - ((σyy-σxx)/2) sin^2(alpha/2) cos^2(alpha/2)]
    common = 0.5*(sigma_yy + sigma_xx) - 0.5*(sigma_yy - sigma_xx)*(sh**2)*(ch**2)

    pref = np.sqrt(np.pi * a)

    # Eq. (20)
    KI = pref * (
        common * (ch/denom)
        + 0.5*(sigma_yy - sigma_xx)*np.cos(3.0*alpha/2.0)
        - sigma_xy*(np.sin(3.0*alpha/2.0) + sh**3)
    )

    # Eq. (21)
    KII = pref * (
        common * (sh/denom)
        + 0.5*(sigma_yy - sigma_xx)*np.sin(3.0*alpha/2.0)
        + sigma_xy*(np.cos(3.0*alpha/2.0) + ch*(sh**2))
    )

    return KI, KII

def cotterell_rice_F_circular_arc(alpha, a, sigma_xx, sigma_yy, sigma_xy):
    KI, KII = cotterell_rice_K_circular_arc(alpha, a, sigma_xx, sigma_yy, sigma_xy)
    c = 2.0*a*np.sin(alpha/2.0)
    Fden = 1.0 * np.sqrt(np.pi*(c/2.0))  # = sqrt(pi*c/2)
    # If you want FI = KI / (sigma*sqrt(pi*c/2)), you must choose what "sigma" is
    # for mixed loading; here we return the raw scaling with sqrt(pi*c/2).
    return KI/Fden, KII/Fden, c




