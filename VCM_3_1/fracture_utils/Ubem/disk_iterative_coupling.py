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

Boundary tractions from the crack field are evaluated directly at boundary points
using the network result object (no grid interpolation of crack stresses).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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


def compute_crack_boundary_tractions_at_points(
    *,
    res,
    boundary_mesh,
    bem_dir: Path,
    out_dir: Path,
    coupling: IterativeCouplingParams,
):
    """
    Estimate *CRACK-INDUCED* boundary tractions directly at boundary points:

        t_cr = σ_crack · n

    This uses:
      res.stress_field_global(X, Y, add_remote=False)
    which returns crack-induced stresses from the solved crack network only.
    No crack-stress grid interpolation is used for boundary traction extraction.

    Returns
    -------
    tx_cr, ty_cr : arrays of length n_boundary_elements, in Pa
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # boundary points + outward normals
    Xb, Nb = _disk_boundary_midpoints_normals(boundary_mesh)
    xb = np.asarray(Xb[:, 0], float)
    yb = np.asarray(Xb[:, 1], float)

    sxx, syy, sxy = res.stress_field_global(xb, yb, add_remote=False)
    sxx = np.nan_to_num(np.asarray(sxx, float), nan=0.0, posinf=0.0, neginf=0.0)
    syy = np.nan_to_num(np.asarray(syy, float), nan=0.0, posinf=0.0, neginf=0.0)
    sxy = np.nan_to_num(np.asarray(sxy, float), nan=0.0, posinf=0.0, neginf=0.0)

    nx = Nb[:, 0]
    ny = Nb[:, 1]
    tx_cr = sxx * nx + sxy * ny
    ty_cr = sxy * nx + syy * ny

    if bool(getattr(coupling, "save_debug_crack_grid", True)):
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_x.npy", xb)
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_y.npy", yb)
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_sxx.npy", sxx)
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_syy.npy", syy)
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_sxy.npy", sxy)
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_tx.npy", tx_cr)
        np.save(out_dir / f"{coupling.arrays_prefix}_boundary_ty.npy", ty_cr)

    return tx_cr, ty_cr


def compute_crack_boundary_tractions_via_grid(
    *,
    res,
    boundary_mesh,
    bem_dir: Path,
    out_dir: Path,
    coupling: IterativeCouplingParams,
):
    """Backward-compatible alias.

    The implementation is now direct point evaluation at boundary points.
    """
    return compute_crack_boundary_tractions_at_points(
        res=res,
        boundary_mesh=boundary_mesh,
        bem_dir=bem_dir,
        out_dir=out_dir,
        coupling=coupling,
    )


def solve_bem_with_extra_boundary_tractions(
    *,
    disk_params,
    bem_dir: Path,
    tx_extra: np.ndarray,
    ty_extra: np.ndarray,
    show: bool = False,
):
    """
    Solve BEM disk with original external pressure arcs + added elementwise tractions.

    IMPORTANT:
      To "reverse" crack tractions, pass tx_extra = -tx_crack, ty_extra = -ty_crack.

    Overwrites the standard arrays in bem_dir:
      xs.npy, ys.npy, Sxx.npy, Syy.npy, Sxy.npy
    """
    from fracture_utils.Ubem.bem_solver import BEMSolver2D
    from fracture_utils.Ubem.boundary_conditions import (
        build_boundary, BCSpec, assemble_segment_bcs, add_boundary_to_solver
    )
    from fracture_utils.Ubem.bem_stress_field import circle_inside
    from fracture_utils.Ubem.bem_stress_plotter import (
        ContourOpts, eval_and_plot_stress_components_contours_separate
    )

    bem_dir = Path(bem_dir)
    bem_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild boundary mesh
    mesh = build_boundary({"type": "circle", "R": disk_params.R, "n_boundary": disk_params.n_elem, "center": (0.0, 0.0)})

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

    solver = BEMSolver2D(E=disk_params.E, nu=disk_params.nu, h=getattr(disk_params, "h", 1.0), plane_strain=True)
    add_boundary_to_solver(solver, mesh, is_traction, bc_x, bc_y)
    solver.solve(gauss_n=4)

    # Save updated stress grid (overwrite standard arrays)
    R = float(disk_params.R)
    bbox = (-R, R, -R, R)
    inside = circle_inside(R, center=(0.0, 0.0), pad=0.03 * R)

    opts_common = dict(
        dpi=300,
        show=bool(show),
        robust=True,
        robust_pct=97.0,
        n_levels=30,
        n_line_levels=20,
        x_scale=1e3, y_scale=1e3,
        x_label="x [mm]", y_label="y [mm]",
        value_scale=1e-6,
        cbar_label="Stress [MPa]",
        cmap="jet",
    )
    opts_xx = ContourOpts(**opts_common, title="σ_xx", symmetric=True)
    opts_yy = ContourOpts(**opts_common, title="σ_yy", symmetric=True)
    opts_xy = ContourOpts(**opts_common, title="σ_xy", symmetric=True)

    xs, ys, Sxx, Syy, Sxy, _paths = eval_and_plot_stress_components_contours_separate(
        solver,
        bbox=bbox,
        out_dir=bem_dir,
        basename="brazilian_disk_iter",
        n=int(getattr(disk_params, "n_grid", 120)),
        inside=inside,
        normalize_by=None,
        opts_xx=opts_xx,
        opts_yy=opts_yy,
        opts_xy=opts_xy,
    )

    np.save(bem_dir / "xs.npy", xs)
    np.save(bem_dir / "ys.npy", ys)
    np.save(bem_dir / "Sxx.npy", Sxx)
    np.save(bem_dir / "Syy.npy", Syy)
    np.save(bem_dir / "Sxy.npy", Sxy)

    return solver, mesh, (xs, ys, Sxx, Syy, Sxy)
