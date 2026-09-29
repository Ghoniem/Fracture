"""
bem_brazilian_disk_analytic.py
==============================

Analytical stress field for the Brazilian disk (diametral compression) with
two *concentrated* balancing forces (Flamant/Hertz/Hondros-type solution).

The formulas implemented here match the equations shown in the user's reference
image (Eq. (2)), written in terms of the non-dimensional coordinates:

    ξ = 2x/d = x/R
    ζ = 2y/d = y/R

with d = 2R the disk diameter, and h the specimen thickness (out-of-plane).

Sign convention
---------------
- P is the magnitude of each concentrated compressive force (top and bottom).
- Stresses follow the convention of the reference equation:
    σ_x, σ_y, τ_xy in Cartesian coordinates.

Implementation note on the prefactor
------------------------------------
The reference shows a prefactor -2P/(π d h) multiplying the bracket with "-1/2".
For the classic Brazilian disk center stresses to match:
    σ_x(0,0) = +2P/(π d h)   (tension across the loaded diameter)
    σ_y(0,0) = -6P/(π d h)   (compression along the load axis)
the effective prefactor must be:
    pref = -4P/(π d h)

This is a common source of confusion because some texts define P as the *total*
applied load rather than the force at each loading point. In this module:
    P = force at each loading point (top and bottom).

If your P definition differs, scale P accordingly (e.g., if your P_total is the
sum of magnitudes over top+bottom, then P = P_total/2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np


@dataclass(frozen=True)
class Stress:
    sxx: float
    syy: float
    sxy: float


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise num/den with NaN where den==0."""
    out = np.full_like(num, np.nan, dtype=float)
    mask = den != 0.0
    out[mask] = num[mask] / den[mask]
    return out


def stress_brazilian_disk_point(
    x: float,
    y: float,
    *,
    R: float,
    P: float,
    h: float,
) -> Stress:
    """Analytical (σ_xx, σ_yy, σ_xy) at a single point (x,y) inside the disk."""
    x = float(x); y = float(y)
    R = float(R); P = float(P); h = float(h)
    d = 2.0 * R

    xi = x / R     # 2x/d
    zeta = y / R   # 2y/d

    # denominators
    Dm = ((1.0 - zeta)**2 + xi**2)
    Dp = ((1.0 + zeta)**2 + xi**2)

    # avoid singular points exactly at load points: (xi=0, zeta=±1) -> D=0
    if Dm == 0.0 or Dp == 0.0:
        return Stress(np.nan, np.nan, np.nan)

    # ASTM/textbook convention: P is the total compressive load at each platen,
    # giving sigma_yy(0,0) = -6P/(pi*D*t) and sigma_xx(0,0) = +2P/(pi*D*t).
    # Must match the vectorized stress_brazilian_disk below to machine precision.
    pref = -4.0 * P / (np.pi * d * h)

    sxx = pref * (
        (1.0 - zeta) * xi**2 / (Dm**2) +
        (1.0 + zeta) * xi**2 / (Dp**2) -
        0.5
    )

    syy = pref * (
        (1.0 - zeta)**3 / (Dm**2) +
        (1.0 + zeta)**3 / (Dp**2) -
        0.5
    )

    sxy = pref * (
        (1.0 - zeta)**2 * xi / (Dm**2) +
        (1.0 + zeta)**2 * xi / (Dp**2)
    )

    return Stress(float(sxx), float(syy), float(sxy))


def stress_brazilian_disk(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    *,
    R: float,
    P: float,
    h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized analytical stress evaluation.

    x, y can be scalars or arrays broadcastable to the same shape.
    Returns (Sxx, Syy, Sxy) arrays of that shape.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    R = float(R); P = float(P); h = float(h)
    d = 2.0 * R

    xi = x / R
    zeta = y / R

    Dm = ((1.0 - zeta)**2 + xi**2)
    Dp = ((1.0 + zeta)**2 + xi**2)

    pref = -4.0 * P / (np.pi * d * h)

    sxx = pref * (
        _safe_div((1.0 - zeta) * xi**2, Dm**2) +
        _safe_div((1.0 + zeta) * xi**2, Dp**2) -
        0.5
    )

    syy = pref * (
        _safe_div((1.0 - zeta)**3, Dm**2) +
        _safe_div((1.0 + zeta)**3, Dp**2) -
        0.5
    )

    sxy = pref * (
        _safe_div((1.0 - zeta)**2 * xi, Dm**2) +
        _safe_div((1.0 + zeta)**2 * xi, Dp**2)
    )

    return sxx, syy, sxy


def stress_center(*, R: float, P: float, h: float) -> Stress:
    """
    Simplified closed-form stresses at the disk center (0,0):

        σ_xx(0,0) = +2P/(π d h)
        σ_yy(0,0) = -6P/(π d h)
        σ_xy(0,0) = 0

    where d = 2R and P is the force at each loading point (top/bottom).
    """
    R = float(R); P = float(P); h = float(h)
    d = 2.0 * R
    sxx = +2.0 * P / (np.pi * d * h)
    syy = -6.0 * P / (np.pi * d * h)
    sxy = 0.0
    return Stress(sxx, syy, sxy)


def l2_error_on_line(
    x: np.ndarray,
    num: np.ndarray,
    ana: np.ndarray,
    *,
    relative: bool = True,
) -> float:
    """
    Discrete L2 error on a line using trapezoidal integration.

    If relative=True:
        ||num-ana||_2 / ||ana||_2
    else:
        ||num-ana||_2
    """
    x = np.asarray(x, dtype=float)
    num = np.asarray(num, dtype=float)
    ana = np.asarray(ana, dtype=float)

    mask = np.isfinite(x) & np.isfinite(num) & np.isfinite(ana)
    if mask.sum() < 2:
        return float("nan")

    x = x[mask]
    e = (num - ana)[mask]
    a = ana[mask]

    # L2 norms via integral
    n_e = np.sqrt(np.trapz(e*e, x))
    if not relative:
        return float(n_e)

    n_a = np.sqrt(np.trapz(a*a, x))
    return float(n_e / n_a) if n_a > 0 else float("nan")
