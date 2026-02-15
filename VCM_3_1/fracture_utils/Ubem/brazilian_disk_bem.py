"""
BEM utilities for Brazilian disk compression field.

Intended location in repo:
  fracture_utils/Ubem/brazilian_disk_bem.py

This module:
- builds/solves the Brazilian disk BEM problem (same logic as your Cell 1)
- produces the initial BEM contour plots (σ_xx, σ_yy, σ_xy) and saves arrays.
  (Avoid re-plotting BEM arrays elsewhere unless you explicitly want a second scaling.)
- evaluates/saves the stress field on a regular grid (xs, ys, Sxx, Syy, Sxy)
- provides a helper to load that saved field

Notes
-----
- The saved stress arrays are in Pa.
- xs, ys are 1D arrays in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from fracture_utils.Ubem.bem_solver import BEMSolver2D
from fracture_utils.Ubem.boundary_conditions import (
    build_boundary,
    BCSpec,
    assemble_segment_bcs,
    add_boundary_to_solver,
)
from fracture_utils.Ubem.bem_plotter import plot_centerline_stresses_circle
from fracture_utils.Ubem.bem_stress_field import circle_inside
from fracture_utils.Ubem.bem_stress_plotter import (
    ContourOpts,
    eval_and_plot_stress_components_contours_separate,
)


@dataclass
class BrazilianDiskParams:
    R: float
    P_total: float
    E: float
    nu: float
    n_elem: int = 120
    arc_half_angle_deg: float = 15.0
    h: float = 6.35e-3
    plane_strain: bool = True
    gauss_n: int = 4
    # stress grid
    n_grid: int = 120
    pad_frac: float = 0.03  # inside-disk mask pad fraction


def compute_bem_brazilian_disk_field(
    params: BrazilianDiskParams,
    out_dir: Path,
    *,
    show: bool = True,
    save_arrays: bool = True,
    save_contours: bool = True,
    normalize_by: Optional[float] = None,
) -> Tuple[BEMSolver2D, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Solve the Brazilian disk BEM problem and evaluate stresses on a grid.

    Returns
    -------
    solver : BEMSolver2D
    (xs, ys, Sxx, Syy, Sxy) : arrays in meters (xs,ys) and Pa (stresses)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build boundary
    boundary_spec = {"type": "circle", "R": params.R, "n_boundary": params.n_elem, "center": (0.0, 0.0)}
    mesh = build_boundary(boundary_spec)

    # 2) BCs (distributed pressure on top/bottom arcs)
    theta_deg = np.asarray(mesh.theta_deg, dtype=float)
    L = np.asarray(mesh.length, dtype=float)

    top = (theta_deg >= 90 - params.arc_half_angle_deg) & (theta_deg <= 90 + params.arc_half_angle_deg)
    bot = (theta_deg >= -90 - params.arc_half_angle_deg) & (theta_deg <= -90 + params.arc_half_angle_deg)

    L_top = float(np.sum(L[top]))
    L_bot = float(np.sum(L[bot]))
    if L_top <= 0 or L_bot <= 0:
        raise RuntimeError("Top/bottom loaded arc has zero length — check selector ranges or n_elem.")

    pressure_top = params.P_total / L_top
    pressure_bot = params.P_total / L_bot

    bc_specs = [
        BCSpec(
            load_type="pressure_normal",
            bc_value=pressure_top,
            selector_kind="theta_deg_range",
            selector_data=(90 - params.arc_half_angle_deg, 90 + params.arc_half_angle_deg),
        ),
        BCSpec(
            load_type="pressure_normal",
            bc_value=pressure_bot,
            selector_kind="theta_deg_range",
            selector_data=(-90 - params.arc_half_angle_deg, -90 + params.arc_half_angle_deg),
        ),
    ]
    is_traction, bc_x, bc_y = assemble_segment_bcs(mesh, bc_specs=bc_specs, default=("traction", 0.0, 0.0))

    # 3) Solve
    solver = BEMSolver2D(E=params.E, nu=params.nu, h=params.h, plane_strain=bool(params.plane_strain))
    add_boundary_to_solver(solver, mesh, is_traction, bc_x, bc_y)
    solver.solve(gauss_n=int(params.gauss_n))

    # 4) Centerline plot (optional)
    plot_centerline_stresses_circle(
        solver,
        R=params.R,
        out_dir=out_dir,
        basename="brazilian_disk_line_stresses",
        normalize_by=params.P_total if normalize_by is None else normalize_by,
        n_points=100,
    )

    # 5) Contours + arrays
    bbox = (-params.R, params.R, -params.R, params.R)
    inside = circle_inside(params.R, center=(0.0, 0.0), pad=float(params.pad_frac) * params.R)

    if save_contours:
        opts_common = dict(
            dpi=300,
            show=bool(show),
            robust=True,
            robust_pct=97.0,
            n_levels=30,
            n_line_levels=20,
            x_scale=1e3,
            y_scale=1e3,
            x_label="x [mm]",
            y_label="y [mm]",
            value_scale=1e-6,
            cbar_label="Stress [MPa]",
            cmap="jet",
        )
        opts_xx = ContourOpts(**opts_common, title="σ_xx", symmetric=True)
        opts_yy = ContourOpts(**opts_common, title="σ_yy", symmetric=True)
        opts_xy = ContourOpts(**opts_common, title="σ_xy", symmetric=True)
    else:
        opts_xx = opts_yy = opts_xy = None

    xs, ys, Sxx, Syy, Sxy, _paths = eval_and_plot_stress_components_contours_separate(
        solver,
        bbox=bbox,
        out_dir=out_dir,
        basename="brazilian_disk",
        n=int(params.n_grid),
        inside=inside,
        normalize_by=normalize_by,
        opts_xx=opts_xx,
        opts_yy=opts_yy,
        opts_xy=opts_xy,
    )

    if save_arrays:
        np.save(out_dir / "Sxx.npy", Sxx)
        np.save(out_dir / "Syy.npy", Syy)
        np.save(out_dir / "Sxy.npy", Sxy)
        np.save(out_dir / "xs.npy", xs)
        np.save(out_dir / "ys.npy", ys)

    return solver, (xs, ys, Sxx, Syy, Sxy)


def load_saved_bem_field(bem_dir: Path):
    """Load xs, ys, Sxx, Syy, Sxy saved by compute_bem_brazilian_disk_field()."""
    bem_dir = Path(bem_dir)
    xs = np.load(bem_dir / "xs.npy")
    ys = np.load(bem_dir / "ys.npy")
    Sxx = np.load(bem_dir / "Sxx.npy")
    Syy = np.load(bem_dir / "Syy.npy")
    Sxy = np.load(bem_dir / "Sxy.npy")
    return xs, ys, Sxx, Syy, Sxy


def ensure_bem_field(
    params: BrazilianDiskParams,
    out_dir: Path,
    *,
    recompute: bool = False,
    show: bool = True,
    save_contours: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: either (a) load an existing saved BEM field from out_dir,
    or (b) compute it if missing or recompute=True.

    This supports fast iteration when BEM solves are expensive.

    Returns
    -------
    xs, ys, Sxx, Syy, Sxy : arrays in meters (xs,ys) and Pa (stresses)
    """
    out_dir = Path(out_dir)
    need = ["xs.npy", "ys.npy", "Sxx.npy", "Syy.npy", "Sxy.npy"]
    have_all = all((out_dir / n).exists() for n in need)
    if (not recompute) and have_all:
        return load_saved_bem_field(out_dir)

    _solver, (xs, ys, Sxx, Syy, Sxy) = compute_bem_brazilian_disk_field(
        params,
        out_dir,
        show=show,
        save_arrays=True,
        save_contours=save_contours,
        normalize_by=None,
    )
    return xs, ys, Sxx, Syy, Sxy
