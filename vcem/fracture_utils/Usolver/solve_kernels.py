"""Elastic kernels used by the solver (stress from differential edge dislocation).

Plane-strain forms; set ``plane_stress=True`` to apply ``nu -> nu / (1 + nu)``
and obtain the plane-stress equivalents.
"""
from __future__ import annotations
import numpy as np
import math


def _nu_eff(nu: float, plane_stress: bool) -> float:
    return float(nu) / (1.0 + float(nu)) if plane_stress else float(nu)


def edge_dislocation_u(
    dx: np.ndarray, dy: np.ndarray, dBx: float, dBy: float, nu: float,
    *, plane_stress: bool = False,
):
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    nu = _nu_eff(nu, plane_stress)

    eps = 1e-30
    r2 = dx * dx + dy * dy + eps
    inv = 1.0 / r2

    atan = np.arctan2(dy, dx)
    ln = np.log(r2)

    c1 = 1.0 / (2.0 * np.pi)
    c2 = 1.0 / (4.0 * np.pi * (1.0 - nu))

    ux_bx = c1 * atan + c2 * (dx * dy) * inv
    uy_bx = -c2 * ((1.0 - 2.0 * nu) * 0.5 * ln + 0.5 * (dx * dx - dy * dy) * inv)

    ux_by = -c2 * ((1.0 - 2.0 * nu) * 0.5 * ln - 0.5 * (dx * dx - dy * dy) * inv)
    uy_by = c1 * atan - c2 * (dx * dy) * inv

    ux = dBx * ux_bx + dBy * ux_by
    uy = dBx * uy_bx + dBy * uy_by
    return ux, uy


def stress_edge_dislocation(
    dx: np.ndarray, dy: np.ndarray, dBx: float, dBy: float, mu: float, nu: float,
    *, plane_stress: bool = False,
):
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    mu = float(mu)
    nu = _nu_eff(nu, plane_stress)

    eps = 1e-30
    coef = mu / (2.0*math.pi*(1.0 - nu))

    def bx_stress(x, y, b):
        r2 = x*x + y*y + eps
        r4 = r2*r2
        sxx = -coef*b * (y*(3.0*x*x + y*y)) / r4
        syy =  coef*b * (y*(x*x - y*y))     / r4
        sxy =  coef*b * (x*(x*x - y*y))     / r4
        return sxx, syy, sxy

    sxx = np.zeros_like(dx)
    syy = np.zeros_like(dx)
    sxy = np.zeros_like(dx)

    if abs(dBx) > 0:
        a,b,c = bx_stress(dx, dy, dBx)
        sxx += a; syy += b; sxy += c

    if abs(dBy) > 0:
        x1 = dy
        y1 = -dx
        sxx1, syy1, sxy1 = bx_stress(x1, y1, dBy)
        sxx += syy1
        syy += sxx1
        sxy += -sxy1

    return sxx, syy, sxy
