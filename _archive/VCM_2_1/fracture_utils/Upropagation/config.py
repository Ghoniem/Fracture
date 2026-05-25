"""Configuration objects for crack propagation (Upropagation).

This module is intentionally solver-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PropagationConfig:
    """Global propagation controls.

    Step modes
    ----------
    step_mode
        - "fixed_step": always use a fixed fraction f_fixed (Δa = f_fixed * L_total).
        - "adaptive_step": use the f / 2f / (f/2) stability search (legacy logic).
    f_fixed
        Fixed non-dimensional extension fraction used when step_mode="fixed_step".

    Adaptive parameters
    -------------------
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

    Global options
    --------------
    simultaneous_tip_growth
        If True, apply all accepted tip updates to the same base geometry.
        If False, apply updates sequentially (ordering-dependent).
    delta_a_min
        Minimum physical increment (meters). If > 0, enforce Δa >= delta_a_min.
        Useful to prevent near-zero "wiggle" steps when running fixed-step demos.
    """

    # --- Step strategy
    step_mode: Literal["fixed_step", "adaptive_step"] = "fixed_step"
    f_fixed: float = 0.10

    # --- Adaptive (legacy) controls
    f0: float = 0.10
    f_min: float = 0.005
    f_max: float = 0.25

    tol_theta_rel: float = 0.10
    tol_theta_abs: float = 1.0 * 3.141592653589793 / 180.0  # 1 degree in radians
    tol_keff_rel: float = 0.10

    prefer_larger_step: bool = True
    max_trials: int = 6

    # --- Global
    simultaneous_tip_growth: bool = True
    delta_a_min: float = 0.0
