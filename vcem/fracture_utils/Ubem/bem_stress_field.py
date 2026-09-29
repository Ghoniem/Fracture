"""
bem_stress_field.py
===================

Convenience functions for evaluating BEM stress components inside the domain.

Wraps:
    solver.compute_stress_at_point(x,y) -> (sxx, syy, sxy)

and provides vectorized/grid helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StressAtPoint:
    sxx: float
    syy: float
    sxy: float


def stress_at_point(solver, x: float, y: float) -> StressAtPoint:
    sxx, syy, sxy = solver.compute_stress_at_point(float(x), float(y))
    return StressAtPoint(float(sxx), float(syy), float(sxy))


def stress_on_grid(
    solver,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    inside: Optional[Callable[[float, float], bool]] = None,
    fill: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Nx = xs.size
    Ny = ys.size

    Sxx = np.full((Ny, Nx), fill, dtype=float)
    Syy = np.full((Ny, Nx), fill, dtype=float)
    Sxy = np.full((Ny, Nx), fill, dtype=float)

    for j in range(Ny):
        y = float(ys[j])
        for i in range(Nx):
            x = float(xs[i])
            if inside is not None and (not bool(inside(x, y))):
                continue
            try:
                a, b, c = solver.compute_stress_at_point(x, y)
                Sxx[j, i] = a
                Syy[j, i] = b
                Sxy[j, i] = c
            except Exception:
                pass

    return Sxx, Syy, Sxy


def circle_inside(R: float, center=(0.0, 0.0), pad: float = 0.0):
    cx, cy = float(center[0]), float(center[1])
    R_eff = float(R) - float(pad)
    R2 = R_eff * R_eff

    def _inside(x: float, y: float) -> bool:
        dx = float(x) - cx
        dy = float(y) - cy
        return (dx*dx + dy*dy) <= R2

    return _inside
