"""Configuration objects for crack propagation (Upropagation).

This module is intentionally solver-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PropagationConfig:
    """Global propagation controls.

    Parameters
    ----------
    f0
        Initial non-dimensional extension fraction, Δa = f * L_total.
    f_min, f_max
        Bounds on f during adaptive step-size selection.
    tol_theta_rel, tol_theta_abs
        Relative/absolute tolerances for propagation angle convergence.
    tol_keff_rel
        Relative tolerance for driving SIF convergence (recommended).
    prefer_larger_step
        If True, accept 2f when stable vs f (aggressive stepping).
    max_trials
        Maximum number of trial step evaluations per tip per increment.
    simultaneous_tip_growth
        If True, apply all accepted tip updates to the same base geometry.
        If False, apply updates sequentially (ordering-dependent).
    """
    f0: float = 0.1
    f_min: float = 0.005
    f_max: float = 0.25

    tol_theta_rel: float = 0.10
    tol_theta_abs: float = 1.0 * 3.141592653589793 / 180.0  # 1 degree in radians
    tol_keff_rel: float = 0.10

    prefer_larger_step: bool = True
    max_trials: int = 6
    simultaneous_tip_growth: bool = True
