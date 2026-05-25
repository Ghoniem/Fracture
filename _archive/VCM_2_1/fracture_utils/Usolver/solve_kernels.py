"""Elastic kernels used by the solver (stress from differential edge dislocation)."""
from __future__ import annotations
import numpy as np
import math

def stress_edge_dislocation(dx: np.ndarray, dy: np.ndarray, dBx: float, dBy: float, mu: float, nu: float):
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    mu = float(mu); nu = float(nu)

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
