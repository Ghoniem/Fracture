# -*- coding: utf-8 -*-
"""
fracture_utils/Ubem/bem_benchmark_tests.py

Patch tests for bem_solver_clean.py

A) Rigid translation (Dirichlet everywhere) on circle:
   u_x=c, u_y=0 -> tractions ~0, stresses ~0.

B) Uniform pressure (Neumann everywhere) on circle:
   t=-p n -> interior stress ≈ -p I.

Run these before attempting Brazilian disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .bem_solver import BEMSolver2D


@dataclass
class BenchResult:
    passed: bool
    metrics: Dict[str, float]


def _circle_pts(R: float, n_elem: int) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n_elem + 1)
    return np.c_[R * np.cos(th), R * np.sin(th)]


def run_benchmark_rigid_translation(*, R: float = 1.0, n_elem: int = 120,
                                    E: float = 200e9, nu: float = 0.3,
                                    ux: float = 1e-4, uy: float = 0.0,
                                    plane_strain: bool = True) -> BenchResult:
    pts = _circle_pts(R, n_elem)
    solver = BEMSolver2D(E=E, nu=nu, plane_strain=plane_strain)
    for i in range(n_elem):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        solver.add_element(x1, y1, x2, y2, is_traction=False, bc_x=ux, bc_y=uy)

    solver.solve(gauss_n=8)

    tmag_rms = float(np.sqrt(np.mean(solver.t_x ** 2 + solver.t_y ** 2)))
    sxx0, syy0, sxy0 = solver.compute_stress_at_point(0.0, 0.0)
    snorm0 = float(np.sqrt(sxx0 ** 2 + syy0 ** 2 + 2.0 * sxy0 ** 2))

    scale = E * abs(ux) / max(R, 1e-12)
    passed = (tmag_rms / scale < 1e-2) and (snorm0 / scale < 1e-2)

    return BenchResult(
        passed=passed,
        metrics={
            "t_rms": tmag_rms,
            "sigma_norm_center": snorm0,
            "scale_Eux_over_R": scale,
            "t_rms_over_scale": tmag_rms / scale,
            "sigma_over_scale": snorm0 / scale,
        }
    )


def run_benchmark_uniform_pressure(*, R: float = 1.0, n_elem: int = 160,
                                   E: float = 200e9, nu: float = 0.3,
                                   p: float = 1e6,
                                   plane_strain: bool = True) -> BenchResult:
    pts = _circle_pts(R, n_elem)
    solver = BEMSolver2D(E=E, nu=nu, plane_strain=plane_strain)
    for i in range(n_elem):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))
        tx = dx / L
        ty = dy / L
        nx = ty
        ny = -tx  # outward for CCW
        solver.add_element(x1, y1, x2, y2, is_traction=True, bc_x=-p * nx, bc_y=-p * ny)

    solver.solve(gauss_n=8)

    sxx0, syy0, sxy0 = solver.compute_stress_at_point(0.0, 0.0)
    err_sxx = abs((sxx0 + p) / p)
    err_syy = abs((syy0 + p) / p)
    err_sxy = abs(sxy0 / p)

    passed = (err_sxx < 5e-2) and (err_syy < 5e-2) and (err_sxy < 5e-2)

    return BenchResult(
        passed=passed,
        metrics={
            "sxx_center": float(sxx0),
            "syy_center": float(syy0),
            "sxy_center": float(sxy0),
            "err_sxx": float(err_sxx),
            "err_syy": float(err_syy),
            "err_sxy": float(err_sxy),
        }
    )