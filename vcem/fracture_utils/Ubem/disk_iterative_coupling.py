"""
disk_iterative_coupling.py

Mode_2 = "iterative" coupling scaffold for Brazilian disk + VCM crack network.

Goal
----
At selected outer-cycle checkpoints (typically each post_simplify):
  1) Compute crack-induced stress field (in infinite medium) on/near the disk boundary.
  2) Convert that to boundary tractions t = σ_crack · n.
  3) Apply REVERSED tractions (-t) to the BEM boundary solver in addition to the external loads.
  4) Re-solve BEM and overwrite the saved BEM stress grid (xs/ys/Sxx/Syy/Sxy).
  5) Next crack solve uses the updated BEM field.

This file intentionally uses a grid+interpolate approach for step (1) so it works with your current API.
Later, we can replace it with a direct point evaluator (faster + more accurate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class IterativeCouplingParams:
    # crack-stress grid used to compute boundary tractions
    extent_factor: float = 1.0        # bbox scale relative to disk bbox
    n_grid: int = 401                 # higher -> better traction estimate
    vmax_factor: float = 2.0          # only for optional debug plots
    save_debug_crack_grid: bool = True
    arrays_prefix: str = "crack_for_bem"

    # under-relaxation on BEM stress update (optional)
    relax: float = 1.0                # 1.0 full update, 0.5 under-relax


def _disk_boundary_midpoints_normals(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary element midpoints and outward normals from the BEM mesh.

    Tries common attribute names:
      - (x_mid, y_mid) or (xm, ym) or computed from (x0,y0,x1,y1)
      - (nx, ny) or (normal_x, normal_y)
    """
    if hasattr(mesh, "x_mid") and hasattr(mesh, "y_mid"):
        Xm = np.column_stack([np.asarray(mesh.x_mid, float), np.asarray(mesh.y_mid, float)])
    elif hasattr(mesh, "xm") and hasattr(mesh, "ym"):
        Xm = np.column_stack([np.asarray(mesh.xm, float), np.asarray(mesh.ym, float)])
    else:
        x0 = np.asarray(getattr(mesh, "x0"), float)
        y0 = np.asarray(getattr(mesh, "y0"), float)
        x1 = np.asarray(getattr(mesh, "x1"), float)
        y1 = np.asarray(getattr(mesh, "y1"), float)
        Xm = np.column_stack([0.5 * (x0 + x1), 0.5 * (y0 + y1)])

    if hasattr(mesh, "nx") and hasattr(mesh, "ny"):
        N = np.column_stack([np.asarray(mesh.nx, float), np.asarray(mesh.ny, float)])
    elif hasattr(mesh, "normal_x") and hasattr(mesh, "normal_y"):
        N = np.column_stack([np.asarray(mesh.normal_x, float), np.asarray(mesh.normal_y, float)])
    else:
        raise AttributeError("BEM mesh does not expose normals (nx,ny) or (normal_x,normal_y).")

    return Xm, N


def _interp_from_grid(xs, ys, Z, Xq):
    """Bilinear interpolation of Z(ys,xs) at query points Xq[:,0]=x, Xq[:,1]=y."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    Z = np.asarray(Z, float)

    xq = np.asarray(Xq[:, 0], float)
    yq = np.asarray(Xq[:, 1], float)

    ix = np.searchsorted(xs, xq) - 1
    iy = np.searchsorted(ys, yq) - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)

    x0 = xs[ix]; x1 = xs[ix + 1]
    y0 = ys[iy]; y1 = ys[iy + 1]

    tx = (xq - x0) / np.maximum(x1 - x0, 1e-300)
    ty = (yq - y0) / np.maximum(y1 - y0, 1e-300)

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11


def compute_crack_boundary_tractions_via_grid(
    *,
    res,
    boundary_mesh,
    bem_dir: Path,
    out_dir: Path,
    coupling: IterativeCouplingParams,
):
    """
    Estimate *CRACK-INDUCED* boundary tractions:

        t_cr = (σ_total - σ_applied) · n

    Why subtract σ_applied?
    ----------------------
    In your DCE solve, the applied field is spatial (from the BEM grid) and is
    included in the stress evaluation even when add_remote=False. For iterative
    coupling we need ONLY the perturbation caused by the crack; otherwise we would
    (incorrectly) reverse the entire applied traction and blow up the BEM solution.

    Inputs
    ------
    bem_dir must contain (Pa):
      xs.npy, ys.npy, Sxx.npy, Syy.npy, Sxy.npy

    Returns
    -------
    tx_cr, ty_cr : arrays of length n_boundary_elements, in Pa
    """
    from fracture_utils.Uplotter.core import DCEPlotterV4, StressPlotOptsV4

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # boundary sample points + normals
    Xb, Nb = _disk_boundary_midpoints_normals(boundary_mesh)

    # bbox from boundary points
    x_min, y_min = np.min(Xb[:, 0]), np.min(Xb[:, 1])
    x_max, y_max = np.max(Xb[:, 0]), np.max(Xb[:, 1])
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    w = (x_max - x_min)
    h = (y_max - y_min)
    half = 0.5 * max(w, h) * float(coupling.extent_factor)

    xs = np.linspace(cx - half, cx + half, int(coupling.n_grid))
    ys = np.linspace(cy - half, cy + half, int(coupling.n_grid))

    # --- σ_total from DCE on grid (includes σ_applied + σ_crack)
    plotter = DCEPlotterV4(res, out_dir=out_dir)
    opts = StressPlotOptsV4(
        extent_factor=1.0,
        n_grid=int(coupling.n_grid),
        add_remote=False,  # does NOT remove spatial applied; we subtract it explicitly below
        cmap="jet",
        n_bands=40,
        label_contours=False,
        vmax_factor=float(coupling.vmax_factor),
        dpi=300,
    )
    plotter.plot_stress_components_global(
        opts=opts,
        components=("sxx", "syy", "sxy"),
        grid=(xs, ys),
        save_arrays=True,
        arrays_prefix=coupling.arrays_prefix,
        show=False,
        save=bool(coupling.save_debug_crack_grid),
    )

    Sxx_tot = np.load(out_dir / f"{coupling.arrays_prefix}_sxx.npy")
    Syy_tot = np.load(out_dir / f"{coupling.arrays_prefix}_syy.npy")
    Sxy_tot = np.load(out_dir / f"{coupling.arrays_prefix}_sxy.npy")

    # --- σ_applied from BEM grid -> interpolate onto (xs,ys)
    xs_b = np.load(Path(bem_dir) / "xs.npy")
    ys_b = np.load(Path(bem_dir) / "ys.npy")
    Sxx_b = np.nan_to_num(np.load(Path(bem_dir) / "Sxx.npy"), nan=0.0)
    Syy_b = np.nan_to_num(np.load(Path(bem_dir) / "Syy.npy"), nan=0.0)
    Sxy_b = np.nan_to_num(np.load(Path(bem_dir) / "Sxy.npy"), nan=0.0)

    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    pts_grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    Sxx_app = _interp_from_grid(xs_b, ys_b, Sxx_b, pts_grid).reshape(len(ys), len(xs))
    Syy_app = _interp_from_grid(xs_b, ys_b, Syy_b, pts_grid).reshape(len(ys), len(xs))
    Sxy_app = _interp_from_grid(xs_b, ys_b, Sxy_b, pts_grid).reshape(len(ys), len(xs))

    # --- σ_crack = σ_total - σ_applied
    Sxx_cr = Sxx_tot - Sxx_app
    Syy_cr = Syy_tot - Syy_app
    Sxy_cr = Sxy_tot - Sxy_app

    # interpolate σ_crack at boundary points
    sxx = _interp_from_grid(xs, ys, Sxx_cr, Xb)
    syy = _interp_from_grid(xs, ys, Syy_cr, Xb)
    sxy = _interp_from_grid(xs, ys, Sxy_cr, Xb)

    nx = Nb[:, 0]
    ny = Nb[:, 1]
    tx_cr = sxx * nx + sxy * ny
    ty_cr = sxy * nx + syy * ny

    return tx_cr, ty_cr


def compute_crack_boundary_tractions_direct(
    *,
    res,
    boundary_mesh,
):
    """
    Fast crack-only boundary traction extraction without building any grid.

    This evaluates crack-induced stresses directly at boundary element midpoints
    via DCEResultsNetworkV4.stress_field_global(..., add_remote=False), then maps:

        t_cr = sigma_cr * n

    Returns
    -------
    tx_cr, ty_cr : arrays of length n_boundary_elements, in Pa
    """
    # boundary sample points + normals
    Xb, Nb = _disk_boundary_midpoints_normals(boundary_mesh)
    nx = Nb[:, 0]
    ny = Nb[:, 1]

    # Crack-only stress (exclude applied/background field)
    sxx, syy, sxy = res.stress_field_global(
        Xb[:, 0],
        Xb[:, 1],
        add_remote=False,
    )
    sxx = np.asarray(sxx, float).reshape(-1)
    syy = np.asarray(syy, float).reshape(-1)
    sxy = np.asarray(sxy, float).reshape(-1)

    tx_cr = sxx * nx + sxy * ny
    ty_cr = sxy * nx + syy * ny
    return tx_cr, ty_cr


def solve_bem_with_extra_boundary_tractions(
    *,
    disk_params,
    bem_dir: Path,
    tx_extra: np.ndarray,
    ty_extra: np.ndarray,
    gauss_n: Optional[int] = None,
    show: bool = False,
    engine: str = "python",
):
    """
    Solve BEM disk with original external pressure arcs + added elementwise tractions.

    IMPORTANT:
      To "reverse" crack tractions, pass tx_extra = -tx_crack, ty_extra = -ty_crack.

    Parameters
    ----------
    gauss_n : int | None
        Quadrature order for BEM solve. If None, uses disk_params.gauss_n (fallback 4).

    Overwrites the standard arrays in bem_dir:
      xs.npy, ys.npy, Sxx.npy, Syy.npy, Sxy.npy
    """
    from fracture_utils.Ubem.bem_solver import BEMSolver2D
    from fracture_utils.Ubem.boundary_conditions import (
        build_boundary, BCSpec, assemble_segment_bcs, add_boundary_to_solver
    )
    from fracture_utils.Ubem.bem_stress_field import circle_inside, stress_on_grid

    bem_dir = Path(bem_dir)
    bem_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild boundary mesh (uniform or clustered per disk_params.boundary_mesh)
    from fracture_utils.Ubem.brazilian_disk_bem import build_disk_boundary
    mesh = build_disk_boundary(disk_params)

    theta_deg = np.asarray(mesh.theta_deg, dtype=float)
    L = np.asarray(mesh.length, dtype=float)

    arc_half = float(getattr(disk_params, "arc_half_angle_deg", 15.0))
    top = (theta_deg >= 90 - arc_half) & (theta_deg <= 90 + arc_half)
    bot = (theta_deg >= -90 - arc_half) & (theta_deg <= -90 + arc_half)

    L_top = float(np.sum(L[top])); L_bot = float(np.sum(L[bot]))
    if L_top <= 0 or L_bot <= 0:
        raise RuntimeError("Top/bottom loaded arc has zero length.")

    pressure_top = float(disk_params.P_total) / (L_top * float(getattr(disk_params, 'h', 1.0)))
    pressure_bot = float(disk_params.P_total) / (L_bot * float(getattr(disk_params, 'h', 1.0)))

    bc_specs = [
        BCSpec("pressure_normal", pressure_top, "theta_deg_range", (90 - arc_half, 90 + arc_half)),
        BCSpec("pressure_normal", pressure_bot, "theta_deg_range", (-90 - arc_half, -90 + arc_half)),
    ]
    is_traction, bc_x, bc_y = assemble_segment_bcs(mesh, bc_specs=bc_specs, default=("traction", 0.0, 0.0))

    tx_extra = np.asarray(tx_extra, float).reshape(-1)
    ty_extra = np.asarray(ty_extra, float).reshape(-1)
    if tx_extra.size != len(bc_x) or ty_extra.size != len(bc_y):
        raise ValueError("tx_extra/ty_extra must match number of boundary elements.")

    bc_x = np.asarray(bc_x, float) + tx_extra
    bc_y = np.asarray(bc_y, float) + ty_extra

    # engine='cpp' uses bem_cpp.BEMSolver2D (same API; OpenMP-parallel)
    if str(engine).lower().strip() == "cpp":
        try:
            import bem_cpp
            _SolverClass = bem_cpp.BEMSolver2D
        except ImportError as _e:
            import warnings
            warnings.warn(f"engine='cpp' requested but bem_cpp not importable "
                          f"({_e!r}); using Python BEMSolver2D.",
                          RuntimeWarning, stacklevel=2)
            _SolverClass = BEMSolver2D
    else:
        _SolverClass = BEMSolver2D
    solver = _SolverClass(E=disk_params.E, nu=disk_params.nu,
                          h=getattr(disk_params, "h", 1.0), plane_strain=True)
    add_boundary_to_solver(solver, mesh, is_traction, bc_x, bc_y)
    gauss_n_eff = int(getattr(disk_params, "gauss_n", 4)) if gauss_n is None else int(gauss_n)
    solver.solve(gauss_n=gauss_n_eff)

    # Save updated stress grid (overwrite standard arrays). BEM-only
    # contour plots are intentionally suppressed -- the only stress
    # figure rendered for this run is the BEM+Crack total field in
    # disk_crack_plotting.plot_total_field.
    R = float(disk_params.R)
    inside = circle_inside(R, center=(0.0, 0.0), pad=0.03 * R)
    n = int(getattr(disk_params, "n_grid", 120))
    xs = np.linspace(-R, R, n)
    ys = np.linspace(-R, R, n)
    Sxx, Syy, Sxy = stress_on_grid(solver, xs, ys, inside=inside)

    np.save(bem_dir / "xs.npy", xs)
    np.save(bem_dir / "ys.npy", ys)
    np.save(bem_dir / "Sxx.npy", Sxx)
    np.save(bem_dir / "Syy.npy", Syy)
    np.save(bem_dir / "Sxy.npy", Sxy)

    return solver, mesh, (xs, ys, Sxx, Syy, Sxy)
