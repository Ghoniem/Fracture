"""
brazilian_disk_bem.py

Brazilian disk compression BEM utilities.

Key points
----------
- Saved stress arrays (Sxx,Syy,Sxy) are ALWAYS in Pa.
- xs, ys are ALWAYS in meters.
- Disk thickness `h` is part of the model and MUST be used consistently.
  If `h` is accidentally omitted (or defaults to 1.0), stresses can jump by ~1/h.

Public API
----------
- BrazilianDiskParams
- compute_bem_brazilian_disk_field(params, out_dir, ...)
- ensure_bem_field(params, out_dir, recompute=False, ...)
- load_bem_field(out_dir)
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
from fracture_utils.Ubem.bem_stress_field import circle_inside, stress_on_grid
from fracture_utils.Ubem.bem_stress_plotter import (
    ContourOpts,
    eval_and_plot_stress_components_contours_separate,
)


@dataclass
class BrazilianDiskParams:
    R: float
    P_total: float           # total compressive force per platen (N)
    E: float
    nu: float

    # discretization / numerics
    n_boundary_elements: int = 120
    gauss_n: int = 4
    plane_strain: bool = True

    # loading arc extent (deg)
    arc_half_angle_deg: float = 15.0

    # IMPORTANT: disk thickness (m)
    h: float = 6.35e-3

    # stress grid
    n_grid: int = 120
    pad_frac: float = 0.03

    # boundary node distribution on the outer circle:
    #   'uniform'  -- equal arc-length spacing (the historical default)
    #   'clustered'-- node density peaks at the loading platens (+/-90 deg)
    #                 and tapers toward the equator. Concentration controls
    #                 peak-to-baseline density ratio; taper_exponent
    #                 controls how sharply the density returns to baseline.
    boundary_mesh: str = "uniform"
    boundary_concentration: float = 8.0
    boundary_taper_exponent: float = 4.0


def build_disk_boundary(params: "BrazilianDiskParams"):
    """Build the outer-circle boundary mesh for a Brazilian disk.

    Switches between uniform and clustered (graded) node distributions
    based on ``params.boundary_mesh``. Clustered mode concentrates nodes
    at +/-90 deg (the loading platens) with peak density set by
    ``params.boundary_concentration`` and taper rate set by
    ``params.boundary_taper_exponent``.
    """
    kind = str(getattr(params, "boundary_mesh", "uniform")).lower().strip()
    if kind in ("clustered", "graded", "circle_graded"):
        spec = {
            "type": "circle_graded",
            "R": float(params.R),
            "n_boundary": int(params.n_boundary_elements),
            "center": (0.0, 0.0),
            "focal_angles_deg": [90.0, -90.0],
            "concentration": float(getattr(params, "boundary_concentration", 8.0)),
            "taper_exponent": float(getattr(params, "boundary_taper_exponent", 4.0)),
        }
    else:
        spec = {
            "type": "circle",
            "R": float(params.R),
            "n_boundary": int(params.n_boundary_elements),
            "center": (0.0, 0.0),
        }
    return build_boundary(spec)


def load_bem_field(bem_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bem_dir = Path(bem_dir)
    xs = np.load(bem_dir / "xs.npy")
    ys = np.load(bem_dir / "ys.npy")
    Sxx = np.load(bem_dir / "Sxx.npy")
    Syy = np.load(bem_dir / "Syy.npy")
    Sxy = np.load(bem_dir / "Sxy.npy")
    return xs, ys, Sxx, Syy, Sxy


def compute_bem_brazilian_disk_field(
    params: BrazilianDiskParams,
    out_dir: Path,
    *,
    show: bool = True,
    save_arrays: bool = True,
    save_contours: bool = True,
    normalize_by: Optional[float] = None,
    engine: str = "python",
) -> Tuple[BEMSolver2D, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Solve the Brazilian disk BEM problem and evaluate stresses on a grid.

    engine: 'python' (default) -- pure-Python BEMSolver2D.
            'cpp'              -- bem_cpp.BEMSolver2D (OpenMP); identical
                                   API + math, dramatically faster (~1000x
                                   on the per-grid-point stress evaluation
                                   that dominates wall-clock for n_grid=120).
    Falls back to Python with a warning if bem_cpp is unavailable.

    Returns
    -------
    solver : BEMSolver2D-like (Python or C++ class depending on engine)
    (xs, ys, Sxx, Syy, Sxy) : xs/ys in meters; stresses in Pa
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build boundary (uniform or clustered per params.boundary_mesh)
    mesh = build_disk_boundary(params)

    # 2) BCs: distributed normal pressure on top/bottom arcs
    theta_deg = np.asarray(mesh.theta_deg, dtype=float)
    L = np.asarray(mesh.length, dtype=float)

    arc = float(params.arc_half_angle_deg)
    top = (theta_deg >= 90 - arc) & (theta_deg <= 90 + arc)
    bot = (theta_deg >= -90 - arc) & (theta_deg <= -90 + arc)

    L_top = float(np.sum(L[top]))
    L_bot = float(np.sum(L[bot]))
    if L_top <= 0 or L_bot <= 0:
        raise RuntimeError("Top/bottom loaded arc has zero length — check arc selector or n_boundary_elements.")

    # NOTE: params.P_total is force (N) per platen.
    # Convert to physical pressure traction (Pa) on each loaded arc:
    #     p = P_total / (L_arc * h)
    # so stresses recovered from BEM remain in Pa without extra thickness scaling.
    h = float(params.h)
    pressure_top = float(params.P_total) / (L_top * h)
    pressure_bot = float(params.P_total) / (L_bot * h)

    bc_specs = [
        BCSpec("pressure_normal", pressure_top, "theta_deg_range", (90 - arc, 90 + arc)),
        BCSpec("pressure_normal", pressure_bot, "theta_deg_range", (-90 - arc, -90 + arc)),
    ]
    is_traction, bc_x, bc_y = assemble_segment_bcs(mesh, bc_specs=bc_specs, default=("traction", 0.0, 0.0))

    # 3) Solve  (engine='cpp' uses bem_cpp.BEMSolver2D, same API)
    use_cpp = (str(engine).lower().strip() == "cpp")
    if use_cpp:
        try:
            import bem_cpp
            _SolverClass = bem_cpp.BEMSolver2D
        except ImportError as _e:
            import warnings
            warnings.warn(f"engine='cpp' requested but bem_cpp not importable "
                          f"({_e!r}); falling back to Python BEMSolver2D.",
                          RuntimeWarning, stacklevel=2)
            _SolverClass = BEMSolver2D
            use_cpp = False
    else:
        _SolverClass = BEMSolver2D
    solver = _SolverClass(E=params.E, nu=params.nu, h=float(params.h),
                          plane_strain=bool(params.plane_strain))
    add_boundary_to_solver(solver, mesh, is_traction, bc_x, bc_y)
    solver.solve(gauss_n=int(params.gauss_n))

    # 4) Centerline plot (optional)
    try:
        plot_centerline_stresses_circle(
            solver,
            R=float(params.R),
            out_dir=out_dir,
            basename="brazilian_disk_line_stresses",
            normalize_by=normalize_by,
            n_points=100,
        )
    except Exception:
        # plotting helper should never break the solve pipeline
        pass

    # 5) Contour stress maps + arrays
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
            out_dir=out_dir,
            basename="brazilian_disk",
            n=int(params.n_grid),
            inside=inside,
            normalize_by=None,     # keep Pa in arrays
            opts_xx=opts_xx,
            opts_yy=opts_yy,
            opts_xy=opts_xy,
        )
    else:
        # Eval-only: compute arrays without producing BEM-only contour PNGs.
        xmin, xmax, ymin, ymax = bbox
        xs = np.linspace(xmin, xmax, int(params.n_grid))
        ys = np.linspace(ymin, ymax, int(params.n_grid))
        Sxx, Syy, Sxy = stress_on_grid(solver, xs, ys, inside=inside)

    if save_arrays:
        np.save(out_dir / "xs.npy", xs)
        np.save(out_dir / "ys.npy", ys)
        np.save(out_dir / "Sxx.npy", Sxx)
        np.save(out_dir / "Syy.npy", Syy)
        np.save(out_dir / "Sxy.npy", Sxy)

    return solver, (xs, ys, Sxx, Syy, Sxy)


def ensure_bem_field(
    params: BrazilianDiskParams,
    out_dir: Path,
    *,
    recompute: bool = False,
    show: bool = True,
    save_contours: bool = True,
    engine: str = "python",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    If out_dir already contains saved arrays and recompute=False, just load them.
    Otherwise compute a fresh BEM solution and overwrite arrays.

    engine: 'python' (default) | 'cpp' -- forwarded to
        compute_bem_brazilian_disk_field. The C++ backend uses
        bem_cpp.BEMSolver2D, dramatically reducing the per-grid-point
        stress-evaluation cost that dominates BEM wall-clock.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    req = ["xs.npy", "ys.npy", "Sxx.npy", "Syy.npy", "Sxy.npy"]
    have = all((out_dir / f).exists() for f in req)

    if have and not recompute:
        return load_bem_field(out_dir)

    _solver, field = compute_bem_brazilian_disk_field(
        params,
        out_dir,
        show=show,
        save_arrays=True,
        save_contours=save_contours,
        normalize_by=None,
        engine=engine,
    )
    return field
