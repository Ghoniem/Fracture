"""
bem_stress_plotter.py
=====================

Plotting utilities for interior stress fields for BEMSolver2D.

This module supports two visualization styles:

A) Heatmaps (imshow)
   - fast, good for quick inspection

B) Contours (contourf + contour)
   - "crack-like" plots with contour lines and strong contrast
   - recommended for publications / reports

Key clarity controls
--------------------
BEM stress evaluation can show very large magnitudes near:
- boundary elements (kernel singularity sensitivity)
- short loaded arcs (Brazilian disk pads)

To get readable, high-contrast fields:
- evaluate only well inside the domain (use an inside predicate with pad)
- use robust color limits (percentile clipping)
- optionally enforce symmetric +/- limits about 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .bem_stress_field import stress_on_grid


# -----------------------------
# helpers
# -----------------------------

def _finite_vals(Z: np.ndarray) -> np.ndarray:
    return Z[np.isfinite(Z)]


def _robust_limits(Z: np.ndarray, pct: float, symmetric: bool) -> Tuple[float, float]:
    """
    Compute vmin/vmax from finite values.
    pct=99 means clip to [1st, 99th] percentiles (asymmetric unless symmetric=True).
    """
    vals = _finite_vals(Z)
    if vals.size == 0:
        return -1.0, 1.0
    lo = np.percentile(vals, 100.0 - pct)
    hi = np.percentile(vals, pct)
    if symmetric:
        m = float(max(abs(lo), abs(hi)))
        return -m, m
    return float(lo), float(hi)


def _make_levels(vmin: float, vmax: float, n_levels: int) -> np.ndarray:
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    if vmin == vmax:
        dv = 1.0 if vmin == 0 else 0.05 * abs(vmin)
        vmin, vmax = vmin - dv, vmax + dv
    return np.linspace(vmin, vmax, n_levels)


# -----------------------------
# options
# -----------------------------

@dataclass(frozen=True)
class HeatmapOpts:
    dpi: int = 200
    show: bool = False
    interpolation: Literal["nearest", "bilinear"] = "bilinear"
    robust: bool = True
    robust_pct: float = 99.0
    symmetric: bool = False
    cmap: Optional[str] = None  # None -> matplotlib default


@dataclass(frozen=True)
class ContourOpts:
    dpi: int = 200
    show: bool = False
    robust: bool = True
    robust_pct: float = 99.0
    symmetric: bool = False
    n_levels: int = 35            # contourf levels
    n_line_levels: int = 12       # contour line levels (subset)
    line_color: str = "k"
    line_width: float = 0.6
    line_alpha: float = 0.55
    cmap: Optional[str] = None    # None -> matplotlib default
    extend: Literal["neither","both","min","max"] = "both"
    # Unit formatting
    x_scale: float = 1.0          # e.g. 1e3 for mm if x is in meters
    y_scale: float = 1.0
    x_label: str = "x"
    y_label: str = "y"
    cbar_label: str = "Stress"
    # Convert values for display (e.g., Pa -> MPa: scale=1e-6)
    value_scale: float = 1.0
    title: str = ""


# -----------------------------
# contour plots (recommended)
# -----------------------------

def plot_stress_component_contour(
    xs: np.ndarray,
    ys: np.ndarray,
    Z: np.ndarray,
    *,
    out_path: Path,
    opts: ContourOpts,
    normalize_by: Optional[float] = None,
):
    """
    Single-component contour plot.

    If normalize_by is provided, Z_display = (Z/normalize_by) * opts.value_scale.
    Otherwise, Z_display = Z * opts.value_scale.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.shape != (ys.size, xs.size):
        raise ValueError("Z must have shape (Ny, Nx) matching ys and xs.")

    # scale x/y for plotting (e.g. m->mm)
    X = xs * float(opts.x_scale)
    Y = ys * float(opts.y_scale)
    XX, YY = np.meshgrid(X, Y)

    Zp = (Z / float(normalize_by)) if normalize_by is not None else Z
    Zp = Zp * float(opts.value_scale)

    # limits + levels
    if opts.robust:
        vmin, vmax = _robust_limits(Zp, pct=opts.robust_pct, symmetric=opts.symmetric)
    else:
        vals = _finite_vals(Zp)
        vmin = float(np.min(vals)) if vals.size else -1.0
        vmax = float(np.max(vals)) if vals.size else 1.0
        if opts.symmetric:
            m = float(max(abs(vmin), abs(vmax)))
            vmin, vmax = -m, m

    levels = _make_levels(vmin, vmax, opts.n_levels)
    line_levels = _make_levels(vmin, vmax, max(2, opts.n_line_levels))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))

    cf = ax.contourf(
        XX, YY, Zp,
        levels=levels,
        cmap=opts.cmap,
        extend=opts.extend,
    )
    ax.contour(
        XX, YY, Zp,
        levels=line_levels,
        colors=opts.line_color,
        linewidths=opts.line_width,
        alpha=opts.line_alpha,
    )

    cblab = opts.cbar_label
    if normalize_by is not None:
        cblab = f"{cblab} / P"
    cb = fig.colorbar(cf, ax=ax, shrink=0.92, label=cblab)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(opts.x_label)
    ax.set_ylabel(opts.y_label)
    if opts.title:
        ax.set_title(opts.title)

    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=opts.dpi, bbox_inches="tight")
    if opts.show:
        plt.show()
    return fig, ax, out_path


def plot_stress_components_contours_separate(
    xs: np.ndarray,
    ys: np.ndarray,
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Sxy: np.ndarray,
    *,
    out_dir: Path,
    basename: str = "stress",
    normalize_by: Optional[float] = None,
    # per-component options (lets you tune symmetric limits for Sxy, etc.)
    opts_xx: Optional[ContourOpts] = None,
    opts_yy: Optional[ContourOpts] = None,
    opts_xy: Optional[ContourOpts] = None,
):
    """
    Save three separate contour plots (one per component) with contour lines.

    Returns list of output paths:
      {basename}_sxx_contour.png
      {basename}_syy_contour.png
      {basename}_sxy_contour.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # defaults
    if opts_xx is None:
        opts_xx = ContourOpts(title="σ_xx")
    if opts_yy is None:
        opts_yy = ContourOpts(title="σ_yy")
    if opts_xy is None:
        opts_xy = ContourOpts(title="σ_xy", symmetric=True)  # shear often best symmetric

    paths = []

    _, _, p = plot_stress_component_contour(
        xs, ys, Sxx,
        out_path=out_dir / f"{basename}_sxx_contour.png",
        opts=opts_xx,
        normalize_by=normalize_by,
    ); paths.append(p)

    _, _, p = plot_stress_component_contour(
        xs, ys, Syy,
        out_path=out_dir / f"{basename}_syy_contour.png",
        opts=opts_yy,
        normalize_by=normalize_by,
    ); paths.append(p)

    _, _, p = plot_stress_component_contour(
        xs, ys, Sxy,
        out_path=out_dir / f"{basename}_sxy_contour.png",
        opts=opts_xy,
        normalize_by=normalize_by,
    ); paths.append(p)

    return paths


def eval_and_plot_stress_components_contours_separate(
    solver,
    bbox: Tuple[float, float, float, float],
    *,
    out_dir: Path,
    basename: str = "stress",
    n: int = 300,
    inside=None,
    normalize_by: Optional[float] = None,
    opts_xx: Optional[ContourOpts] = None,
    opts_yy: Optional[ContourOpts] = None,
    opts_xy: Optional[ContourOpts] = None,
):
    """
    Convenience: evaluate stresses on a grid in bbox and save three contour plots.

    Returns:
      xs, ys, Sxx, Syy, Sxy, paths
    """
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    Sxx, Syy, Sxy = stress_on_grid(solver, xs, ys, inside=inside)
    paths = plot_stress_components_contours_separate(
        xs, ys, Sxx, Syy, Sxy,
        out_dir=out_dir,
        basename=basename,
        normalize_by=normalize_by,
        opts_xx=opts_xx,
        opts_yy=opts_yy,
        opts_xy=opts_xy,
    )
    return xs, ys, Sxx, Syy, Sxy, paths


# -----------------------------
# legacy heatmap API (kept)
# -----------------------------

def plot_stress_component_map(
    xs: np.ndarray,
    ys: np.ndarray,
    Z: np.ndarray,
    *,
    out_path: Path,
    title: str,
    label: str,
    normalize_by: Optional[float] = None,
    opts: HeatmapOpts = HeatmapOpts(),
):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.shape != (ys.size, xs.size):
        raise ValueError("Z must have shape (Ny, Nx) matching ys and xs.")

    Zp = Z / float(normalize_by) if normalize_by is not None else Z

    if opts.robust:
        vmin, vmax = _robust_limits(Zp, pct=opts.robust_pct, symmetric=opts.symmetric)
    else:
        vals = _finite_vals(Zp)
        vmin = float(np.min(vals)) if vals.size else -1.0
        vmax = float(np.max(vals)) if vals.size else 1.0
        if opts.symmetric:
            m = float(max(abs(vmin), abs(vmax)))
            vmin, vmax = -m, m

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6.7, 5.6))
    im = ax.imshow(
        Zp,
        origin="lower",
        extent=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation=opts.interpolation,
        cmap=opts.cmap,
    )
    cb_label = label if normalize_by is None else f"{label} / P"
    fig.colorbar(im, ax=ax, shrink=0.9, label=cb_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=opts.dpi, bbox_inches="tight")
    if opts.show:
        plt.show()
    return fig, ax, out_path
