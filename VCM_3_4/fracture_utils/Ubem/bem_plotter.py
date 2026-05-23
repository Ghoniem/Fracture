"""
bem_plotter.py
==============

Plotting utilities for the pure-python BEMSolver2D.

Primary goal: reproduce the "line stress" plots used in the Brazilian disk
cell (σ_xx and σ_yy along horizontal and vertical centerlines).

Notes
-----
- The solver interface used:
    solver.compute_stress_at_point(x,y) -> (sxx, syy, sxy)
- No seaborn; pure matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class LineStressPlotOpts:
    n_points: int = 100
    pad_frac: float = 0.95   # plot from -pad_frac*R to +pad_frac*R for circular domains
    normalize_by: Optional[float] = None  # e.g. P_total to plot stresses/P_total
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    add_zero_line: bool = True

def plot_centerline_stresses_circle(
    solver,
    R: float,
    out_dir: Path,
    basename: str = "line_stresses",
    normalize_by: Optional[float] = None,
    n_points: int = 100,
    pad_frac: float = 0.95,
    analytical_sigma_yy_over_P: Optional[float] = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Replicates the two-panel plot:
      - stresses along horizontal centerline y=0 (x/R vs stress/P)
      - stresses along vertical centerline x=0 (y/R vs stress/P)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_line = np.linspace(-pad_frac*R, pad_frac*R, n_points)
    y_line = np.linspace(-pad_frac*R, pad_frac*R, n_points)

    sxx_h, syy_h = _eval_line(solver, x_line, np.zeros_like(x_line))
    sxx_v, syy_v = _eval_line(solver, np.zeros_like(y_line), y_line)

    if normalize_by is not None:
        sxx_h = sxx_h / normalize_by
        syy_h = syy_h / normalize_by
        sxx_v = sxx_v / normalize_by
        syy_v = syy_v / normalize_by

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x_line/R, sxx_h, linewidth=2, label="σ_xx (BEM)")
    ax.plot(x_line/R, syy_h, linewidth=2, label="σ_yy (BEM)")
    if analytical_sigma_yy_over_P is not None:
        ax.axhline(analytical_sigma_yy_over_P, linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"σ_yy Analytical = {analytical_sigma_yy_over_P:.3f}P")
    ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("x/R")
    ax.set_ylabel("Stress" if normalize_by is None else "Stress / P")
    ax.set_title("Stress Along Horizontal Centerline (y=0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim([-1, 1])

    ax = axes[1]
    ax.plot(y_line/R, sxx_v, linewidth=2, label="σ_xx (BEM)")
    ax.plot(y_line/R, syy_v, linewidth=2, label="σ_yy (BEM)")
    if analytical_sigma_yy_over_P is not None:
        ax.axhline(analytical_sigma_yy_over_P, linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"σ_yy Analytical = {analytical_sigma_yy_over_P:.3f}P")
    ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("y/R")
    ax.set_ylabel("Stress" if normalize_by is None else "Stress / P")
    ax.set_title("Stress Along Vertical Centerline (x=0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim([-1, 1])

    plt.tight_layout()
    out_path = out_dir / f"{basename}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path

def plot_stress_map(
    solver,
    bbox: Tuple[float, float, float, float],
    out_dir: Path,
    basename: str = "stress_map",
    component: str = "syy",
    n: int = 200,
    normalize_by: Optional[float] = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Optional: 2D heatmap for a stress component over a rectangular bbox.

    bbox = (xmin, xmax, ymin, ymax)
    component in {"sxx","syy","sxy","vonmises"}
    """
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)

    for j in range(n):
        for i in range(n):
            sxx, syy, sxy = solver.compute_stress_at_point(float(X[j, i]), float(Y[j, i]))
            if component == "sxx":
                val = sxx
            elif component == "syy":
                val = syy
            elif component == "sxy":
                val = sxy
            elif component == "vonmises":
                val = np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy)
            else:
                raise ValueError("component must be one of sxx, syy, sxy, vonmises")
            Z[j, i] = val

    if normalize_by is not None:
        Z = Z / normalize_by

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    im = ax.imshow(Z, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.9, label=(component if normalize_by is None else f"{component} / P"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Stress map: {component}")
    out_path = out_dir / f"{basename}_{component}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path

def _eval_line(solver, xs: np.ndarray, ys: np.ndarray):
    sxx = np.empty_like(xs, dtype=float)
    syy = np.empty_like(xs, dtype=float)
    for k in range(xs.size):
        try:
            a, b, _ = solver.compute_stress_at_point(float(xs[k]), float(ys[k]))
            sxx[k] = a
            syy[k] = b
        except Exception:
            sxx[k] = np.nan
            syy[k] = np.nan
    return sxx, syy
