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
        Minimum physical increment (meters). If > 0, tips whose scaled Δa
        falls below this tolerance are *eliminated* (no kink added that
        step), rather than bumped up. Useful to suppress near-zero
        "wiggle" kinks on tips whose K_eff is just barely above Kc when
        the dominant tip drives the global ds_ref.

    Disk-radius step (preferred for the Brazilian-disk pipeline)
    -----------------------------------------------------------
    disk_radius_m
        If > 0, every growth increment uses Δa_ref = f_disk_radius * disk_radius_m
        instead of f_fixed * L_total. This makes the absolute step size
        invariant under network length (no meander from a growing L_ref).
        When set, step_mode / f_fixed / f0 are ignored by NetworkGrowthRunner.
    f_disk_radius
        Step size as a fraction of the disk radius (default 0.05).
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

    # --- Disk-radius step (overrides L_total scaling when disk_radius_m > 0)
    f_disk_radius: float = 0.05
    disk_radius_m: float = 0.0
