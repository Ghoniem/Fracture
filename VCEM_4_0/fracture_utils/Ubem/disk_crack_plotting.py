
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fracture_utils.Uplotter.core import DCEPlotterV4, StressPlotOptsV4
from fracture_utils.Ubem.bem_solver import BEMSolver2D
from fracture_utils.Ubem.bem_stress_field import stress_on_grid
from fracture_utils.Ubem.disk_iterative_coupling import _interp_from_grid as _bilinear_on_grid


# ============================================================
# Plot parameter container
# ============================================================
class PlotParams:
    def __init__(
        self,
        cmap="jet",
        n_bands=40,
        vmax_factor=2.0,
        dpi=300,
        match_limits_to_crack=True,
        robust=True,
        robust_pct=97.0,
        symmetric=True,
        n_levels=30,
        n_line_levels=20,
        fixed_vlim_mpa=None,
        augmented_correction_stride=4,
        lock_vlim_to_first_step=True,
    ):
        self.cmap = cmap
        self.n_bands = n_bands
        self.vmax_factor = vmax_factor
        self.dpi = dpi
        self.match_limits_to_crack = match_limits_to_crack
        self.robust = robust
        self.robust_pct = robust_pct
        self.symmetric = symmetric
        self.n_levels = n_levels
        self.n_line_levels = n_line_levels
        # If set, enforce [-fixed_vlim_mpa, +fixed_vlim_mpa] for contour scaling.
        self.fixed_vlim_mpa = fixed_vlim_mpa
        # Evaluate augmented correction field on coarse grid and interpolate to speed up.
        # 1 => full grid (slow), 2/4 => much faster.
        self.augmented_correction_stride = int(max(1, augmented_correction_stride))
        # When True, the first per-STEP contour call computes the robust
        # vlim from its data and caches it to run_dir/stress_vlim_mpa.json;
        # subsequent STEPs read that cached value so the colorbar stays
        # constant across all frames (defects remain visible at the same
        # color throughout the run). Overridden by fixed_vlim_mpa when set.
        self.lock_vlim_to_first_step = bool(lock_vlim_to_first_step)


# ============================================================
# Shared vlim cache (locked across STEPs)
# ============================================================
_VLIM_CACHE_NAME = "stress_vlim_mpa.json"


def _vlim_cache_path(out_dir: Path) -> Path:
    """Resolve cache path: out_dir is the STEP dir, cache lives in run dir."""
    return Path(out_dir).parent / _VLIM_CACHE_NAME


def _resolve_vlim_mpa(name, Z_ref_pa, Z_tot_pa, params, out_dir,
                      robust_vlim_fn):
    """Decide the colorbar half-range in MPa for one component.

    Resolution order:
      1) ``params.fixed_vlim_mpa`` -- explicit absolute cap, always wins.
      2) ``params.lock_vlim_to_first_step`` + cached value for this
         component name in ``out_dir.parent/stress_vlim_mpa.json``.
         Subsequent STEPs read from the cache so colorbar stays constant.
      3) Compute the robust vlim from this STEP's data and (if locking is
         on) write it into the cache for later STEPs to pick up.

    Returns the half-range (positive MPa) or None when no finite vlim can
    be determined; callers fall back to letting matplotlib auto-pick.
    """
    fixed = None
    if params.fixed_vlim_mpa is not None:
        try:
            fixed = float(params.fixed_vlim_mpa)
        except Exception:
            fixed = None
        if fixed is not None and fixed <= 0.0:
            fixed = None
    if fixed is not None:
        return fixed

    lock = bool(getattr(params, "lock_vlim_to_first_step", True))
    cache_path = _vlim_cache_path(out_dir)
    cached_data = {}
    if lock and cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text())
        except Exception:
            cached_data = {}
    cached_val = cached_data.get(name) if isinstance(cached_data, dict) else None
    if cached_val is not None:
        try:
            cv = float(cached_val)
            if np.isfinite(cv) and cv > 0.0:
                return cv
        except Exception:
            pass

    vlim_pa = robust_vlim_fn(Z_ref_pa if params.match_limits_to_crack else Z_tot_pa)
    if vlim_pa is None or not np.isfinite(vlim_pa) or vlim_pa <= 0.0:
        return None
    vlim_mpa = float(vlim_pa) * 1e-6

    if lock:
        try:
            cached_data[name] = vlim_mpa
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached_data, indent=2))
        except Exception as e:
            print(f"[plot_total_field] vlim cache write failed: {e}")
    return vlim_mpa


# ============================================================
# TOTAL FIELD PLOTTING ONLY  (BEM + Crack)
# ============================================================
def _build_augmented_solver(res):
    """Reconstruct the boundary BEM solver from res.sol['coupling'] if the
    run was augmented; return (solver, ...) or None."""
    sol = getattr(res, "sol", {})
    if not isinstance(sol, dict):
        return None
    c = sol.get("coupling", {})
    if not isinstance(c, dict) or (not bool(c.get("augmented", False))):
        return None

    y = np.asarray(c.get("y_bc", []), float).reshape(-1)
    nb = int(c.get("n_boundary", 0))
    x1 = np.asarray(c.get("boundary_x1", []), float).reshape(-1)
    y1 = np.asarray(c.get("boundary_y1", []), float).reshape(-1)
    x2 = np.asarray(c.get("boundary_x2", []), float).reshape(-1)
    y2 = np.asarray(c.get("boundary_y2", []), float).reshape(-1)
    if nb <= 0 or y.size != 4 * nb:
        return None
    if x1.size != nb or y1.size != nb or x2.size != nb or y2.size != nb:
        return None

    u = y[: 2 * nb]
    t = y[2 * nb :]
    ux = u[0::2]; uy = u[1::2]
    tx = t[0::2]; ty = t[1::2]

    mat = getattr(getattr(res, "calc", None), "material", None)
    if mat is None:
        return None
    E = float(getattr(mat, "E"))
    nu = float(getattr(mat, "nu"))
    plane_stress = bool(getattr(mat, "plane_stress", False))

    solver = BEMSolver2D(E=E, nu=nu, h=1.0, plane_strain=(not plane_stress))
    for i in range(nb):
        solver.add_element(
            float(x1[i]), float(y1[i]),
            float(x2[i]), float(y2[i]),
            True, 0.0, 0.0,
        )
    solver.u_x = np.asarray(ux, float).copy()
    solver.u_y = np.asarray(uy, float).copy()
    solver.t_x = np.asarray(tx, float).copy()
    solver.t_y = np.asarray(ty, float).copy()
    return solver


def _plot_total_field_polar(bem_dir, res, out_dir, tag, params: PlotParams, show=True):
    """Polar/curvilinear variant of plot_total_field.

    Loads xs_polar/ys_polar (Nt, Nr) and the matching polar BEM stress
    arrays from bem_dir, evaluates the crack stress and augmented-BC
    correction at the same polar Cartesian points, and renders with
    contourf on the curvilinear (Xp, Yp) mesh so the disk boundary at
    r = R is exact.
    """
    bem_dir = Path(bem_dir)
    Xp = np.load(bem_dir / "xs_polar.npy")
    Yp = np.load(bem_dir / "ys_polar.npy")
    Sxx_bem = np.load(bem_dir / "Sxx_polar.npy")
    Syy_bem = np.load(bem_dir / "Syy_polar.npy")
    Sxy_bem = np.load(bem_dir / "Sxy_polar.npy")

    # Close the angular seam at theta=0/2pi so contourf doesn't leave a
    # missing wedge. The polar grid stores theta in [0, 2 pi), so the
    # first and last rows are NOT coincident — wrap by appending the
    # first row at the end.
    if not (np.allclose(Xp[0, :], Xp[-1, :]) and np.allclose(Yp[0, :], Yp[-1, :])):
        Xp = np.vstack([Xp, Xp[:1, :]])
        Yp = np.vstack([Yp, Yp[:1, :]])
        Sxx_bem = np.vstack([Sxx_bem, Sxx_bem[:1, :]])
        Syy_bem = np.vstack([Syy_bem, Syy_bem[:1, :]])
        Sxy_bem = np.vstack([Sxy_bem, Sxy_bem[:1, :]])

    # --- Crack stress at the polar Cartesian points (2D arrays OK).
    Sxx_cr, Syy_cr, Sxy_cr = res.stress_field_global(Xp, Yp, add_remote=False)
    Sxx_cr = np.asarray(Sxx_cr, float)
    Syy_cr = np.asarray(Syy_cr, float)
    Sxy_cr = np.asarray(Sxy_cr, float)

    # --- Augmented-BC correction: evaluate on a coarse rect bbox of the
    # polar Cartesian extent, then bilinear-interpolate to the polar
    # points. Mirrors the stride trick used in the rectangular branch.
    solver = _build_augmented_solver(res)
    if solver is None:
        Sxx_bc = np.zeros_like(Sxx_bem)
        Syy_bc = np.zeros_like(Syy_bem)
        Sxy_bc = np.zeros_like(Sxy_bem)
    else:
        x_lo, x_hi = float(np.nanmin(Xp)), float(np.nanmax(Xp))
        y_lo, y_hi = float(np.nanmin(Yp)), float(np.nanmax(Yp))
        # Coarse rect grid sized roughly like the polar resolution so the
        # interpolation error stays sub-percent.
        n_coarse_x = max(32, int(Xp.shape[1]) + 1)
        n_coarse_y = max(32, int(Xp.shape[1]) + 1)
        stride = int(max(1, getattr(params, "augmented_correction_stride", 1)))
        n_coarse_x = max(8, n_coarse_x // stride)
        n_coarse_y = max(8, n_coarse_y // stride)
        xs_c = np.linspace(x_lo, x_hi, n_coarse_x)
        ys_c = np.linspace(y_lo, y_hi, n_coarse_y)
        Sxx_c, Syy_c, Sxy_c = stress_on_grid(solver, xs_c, ys_c, inside=None, fill=np.nan)

        q = np.column_stack([Xp.ravel(), Yp.ravel()])
        Sxx_bc = _bilinear_on_grid(xs_c, ys_c, Sxx_c, q).reshape(Xp.shape)
        Syy_bc = _bilinear_on_grid(xs_c, ys_c, Syy_c, q).reshape(Xp.shape)
        Sxy_bc = _bilinear_on_grid(xs_c, ys_c, Sxy_c, q).reshape(Xp.shape)

    # --- Combine. Same d_mode logic as the rect branch.
    coupling = getattr(res, "sol", {}).get("coupling", {}) if isinstance(getattr(res, "sol", {}), dict) else {}
    d_mode = str(coupling.get("d_mode", "")).lower().strip() if isinstance(coupling, dict) else ""
    use_baseline_bem = not (bool(coupling.get("augmented", False)) and d_mode == "external_total")

    if use_baseline_bem:
        Sxx_tot = Sxx_bem + Sxx_cr + Sxx_bc
        Syy_tot = Syy_bem + Syy_cr + Syy_bc
        Sxy_tot = Sxy_bem + Sxy_cr + Sxy_bc
    else:
        Sxx_tot = Sxx_cr + Sxx_bc
        Syy_tot = Syy_cr + Syy_bc
        Sxy_tot = Sxy_cr + Sxy_bc

    # NaN-mask outside-disk consistent with the stored polar BEM arrays.
    bem_mask = np.isfinite(Sxx_bem) & np.isfinite(Syy_bem) & np.isfinite(Sxy_bem)
    Sxx_tot = np.where(bem_mask, Sxx_tot, np.nan)
    Syy_tot = np.where(bem_mask, Syy_tot, np.nan)
    Sxy_tot = np.where(bem_mask, Sxy_tot, np.nan)

    def _robust_vlim(Z_ref_pa):
        Z = np.asarray(Z_ref_pa, float)
        Z = Z[np.isfinite(Z)]
        if Z.size == 0:
            return None
        if params.robust:
            q = np.nanpercentile(np.abs(Z), float(params.robust_pct))
        else:
            q = np.nanmax(np.abs(Z))
        if not np.isfinite(q) or q <= 0:
            return None
        return float(q) * float(params.vmax_factor)

    def _plot(Z_tot_pa, Z_ref_pa, name):
        Zm = Z_tot_pa * 1e-6
        vlim_mpa = _resolve_vlim_mpa(
            name, Z_ref_pa, Z_tot_pa, params, out_dir, _robust_vlim,
        )
        if vlim_mpa is not None:
            if params.symmetric:
                vmin, vmax = -vlim_mpa, vlim_mpa
            else:
                vmin, vmax = np.nanmin(Zm), np.nanmax(Zm)
            levels = np.linspace(vmin, vmax, int(params.n_levels))
            line_levels = np.linspace(vmin, vmax, int(params.n_line_levels))
        else:
            levels = int(params.n_levels)
            line_levels = int(params.n_line_levels)

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        cf = ax.contourf(
            Xp * 1e3, Yp * 1e3, Zm,
            levels=levels, cmap=params.cmap, extend="both",
        )
        ax.contour(
            Xp * 1e3, Yp * 1e3, Zm,
            levels=line_levels, colors="k", linewidths=0.5, alpha=0.6,
        )
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"TOTAL {name} = BEM + Crack ({tag})")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Stress [MPa]")
        fig.savefig(out_dir / f"total_{name}_{tag}.png", dpi=params.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    _plot(Sxx_tot, Sxx_cr, "sxx")
    _plot(Syy_tot, Syy_cr, "syy")
    _plot(Sxy_tot, Sxy_cr, "sxy")

    return Sxx_tot, Syy_tot, Sxy_tot


def plot_total_field(bem_dir, res, out_dir, tag, params: PlotParams, show=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bem_dir = Path(bem_dir)

    # Curvilinear polar branch: when the BEM pipeline saved a polar grid
    # (xs_polar / ys_polar of shape (Nt, Nr)), use it so the disk boundary
    # is exact instead of stair-stepping along a masked rectangular grid.
    polar_files = ["xs_polar.npy", "ys_polar.npy",
                   "Sxx_polar.npy", "Syy_polar.npy", "Sxy_polar.npy"]
    if all((bem_dir / f).exists() for f in polar_files):
        _xs_p = np.load(bem_dir / "xs_polar.npy")
        _ys_p = np.load(bem_dir / "ys_polar.npy")
        if _xs_p.ndim == 2 and _ys_p.ndim == 2 and _xs_p.shape == _ys_p.shape:
            return _plot_total_field_polar(bem_dir, res, out_dir, tag, params, show)

    # --- Load BEM arrays (Pa)
    xs = np.load(bem_dir / "xs.npy")
    ys = np.load(bem_dir / "ys.npy")
    Sxx_bem = np.load(bem_dir / "Sxx.npy")
    Syy_bem = np.load(bem_dir / "Syy.npy")
    Sxy_bem = np.load(bem_dir / "Sxy.npy")

    # --- Crack stress on SAME grid (Pa)
    plotter = DCEPlotterV4(res, out_dir=out_dir)

    opts = StressPlotOptsV4(
        extent_factor=3,
        n_grid=len(xs),
        add_remote=False,
        cmap=params.cmap,
        n_bands=params.n_bands,
        label_contours=False,
        vmax_factor=params.vmax_factor,
        dpi=params.dpi,
    )

    figs = plotter.plot_stress_components_global(
        opts=opts,
        components=("sxx", "syy", "sxy"),
        grid=(xs, ys),
        save_arrays=True,
        arrays_prefix="crack_tmp",
        show=False,
        save=False,
    )
    # plot_stress_components_global builds 3 matplotlib figures even when
    # save=False/show=False (we only want the saved npy arrays here).
    # Closing them is mandatory: with enable_per_cycle_total_contour_save
    # this runs on every inner cycle, and the orphaned figures pile up
    # in pyplot's global manager until the kernel is OOM-killed.
    if isinstance(figs, dict):
        for _f in figs.values():
            plt.close(_f)

    Sxx_cr = np.load(out_dir / "crack_tmp_sxx.npy")
    Syy_cr = np.load(out_dir / "crack_tmp_syy.npy")
    Sxy_cr = np.load(out_dir / "crack_tmp_sxy.npy")

    # _interp_from_grid was a local copy of the bilinear-interpolation routine in
    # disk_iterative_coupling; we import it as _bilinear_on_grid above.
    _interp_from_grid = _bilinear_on_grid

    def _augmented_correction_field():
        sol = getattr(res, "sol", {})
        if not isinstance(sol, dict):
            return None
        c = sol.get("coupling", {})
        if not isinstance(c, dict) or (not bool(c.get("augmented", False))):
            return None

        y = np.asarray(c.get("y_bc", []), float).reshape(-1)
        nb = int(c.get("n_boundary", 0))
        x1 = np.asarray(c.get("boundary_x1", []), float).reshape(-1)
        y1 = np.asarray(c.get("boundary_y1", []), float).reshape(-1)
        x2 = np.asarray(c.get("boundary_x2", []), float).reshape(-1)
        y2 = np.asarray(c.get("boundary_y2", []), float).reshape(-1)
        if nb <= 0 or y.size != 4 * nb:
            return None
        if x1.size != nb or y1.size != nb or x2.size != nb or y2.size != nb:
            return None

        u = y[: 2 * nb]
        t = y[2 * nb :]
        ux = u[0::2]
        uy = u[1::2]
        tx = t[0::2]
        ty = t[1::2]

        mat = getattr(getattr(res, "calc", None), "material", None)
        if mat is None:
            return None
        E = float(getattr(mat, "E"))
        nu = float(getattr(mat, "nu"))
        plane_stress = bool(getattr(mat, "plane_stress", False))

        solver = BEMSolver2D(E=E, nu=nu, h=1.0, plane_strain=(not plane_stress))
        for i in range(nb):
            solver.add_element(
                float(x1[i]), float(y1[i]),
                float(x2[i]), float(y2[i]),
                True, 0.0, 0.0,
            )
        solver.u_x = np.asarray(ux, float).copy()
        solver.u_y = np.asarray(uy, float).copy()
        solver.t_x = np.asarray(tx, float).copy()
        solver.t_y = np.asarray(ty, float).copy()

        stride = int(max(1, getattr(params, "augmented_correction_stride", 1)))
        if stride <= 1:
            Sxx_bc, Syy_bc, Sxy_bc = stress_on_grid(solver, xs, ys, inside=None, fill=np.nan)
            return Sxx_bc, Syy_bc, Sxy_bc

        # Coarse evaluation + bilinear interpolation to full grid
        xs_c = np.asarray(xs[::stride], float)
        ys_c = np.asarray(ys[::stride], float)
        if xs_c[-1] != xs[-1]:
            xs_c = np.r_[xs_c, xs[-1]]
        if ys_c[-1] != ys[-1]:
            ys_c = np.r_[ys_c, ys[-1]]
        Sxx_c, Syy_c, Sxy_c = stress_on_grid(solver, xs_c, ys_c, inside=None, fill=np.nan)

        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
        q = np.column_stack([Xg.ravel(), Yg.ravel()])
        Sxx_bc = _interp_from_grid(xs_c, ys_c, Sxx_c, q).reshape(len(ys), len(xs))
        Syy_bc = _interp_from_grid(xs_c, ys_c, Syy_c, q).reshape(len(ys), len(xs))
        Sxy_bc = _interp_from_grid(xs_c, ys_c, Sxy_c, q).reshape(len(ys), len(xs))
        return Sxx_bc, Syy_bc, Sxy_bc

    corr = _augmented_correction_field()
    if corr is None:
        Sxx_bc = np.zeros_like(Sxx_bem)
        Syy_bc = np.zeros_like(Syy_bem)
        Sxy_bc = np.zeros_like(Sxy_bem)
    else:
        Sxx_bc, Syy_bc, Sxy_bc = corr

    # In augmented direct mode with d_mode='external_total', the solved boundary
    # unknowns y already carry the externally loaded boundary response. Adding the
    # baseline BEM field again would double-count the external load.
    coupling = getattr(res, "sol", {}).get("coupling", {}) if isinstance(getattr(res, "sol", {}), dict) else {}
    d_mode = str(coupling.get("d_mode", "")).lower().strip() if isinstance(coupling, dict) else ""
    use_baseline_bem = not (bool(coupling.get("augmented", False)) and d_mode == "external_total")

    if use_baseline_bem:
        Sxx_tot = Sxx_bem + Sxx_cr + Sxx_bc
        Syy_tot = Syy_bem + Syy_cr + Syy_bc
        Sxy_tot = Sxy_bem + Sxy_cr + Sxy_bc
    else:
        Sxx_tot = Sxx_cr + Sxx_bc
        Syy_tot = Syy_cr + Syy_bc
        Sxy_tot = Sxy_cr + Sxy_bc

    # Keep outside-domain mask consistent with stored BEM arrays.
    bem_mask = np.isfinite(Sxx_bem) & np.isfinite(Syy_bem) & np.isfinite(Sxy_bem)
    Sxx_tot = np.where(bem_mask, Sxx_tot, np.nan)
    Syy_tot = np.where(bem_mask, Syy_tot, np.nan)
    Sxy_tot = np.where(bem_mask, Sxy_tot, np.nan)

    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

    def _robust_vlim(Z_ref_pa):
        Z = np.asarray(Z_ref_pa, float)
        Z = Z[np.isfinite(Z)]
        if Z.size == 0:
            return None
        if params.robust:
            q = np.nanpercentile(np.abs(Z), float(params.robust_pct))
        else:
            q = np.nanmax(np.abs(Z))
        if not np.isfinite(q) or q <= 0:
            return None
        return float(q) * float(params.vmax_factor)

    def _plot(Z_tot_pa, Z_ref_pa, name):
        Zm = Z_tot_pa * 1e-6  # MPa
        vlim_mpa = _resolve_vlim_mpa(
            name, Z_ref_pa, Z_tot_pa, params, out_dir, _robust_vlim,
        )
        if vlim_mpa is not None:
            if params.symmetric:
                vmin, vmax = -vlim_mpa, vlim_mpa
            else:
                vmin, vmax = np.nanmin(Zm), np.nanmax(Zm)
            levels = np.linspace(vmin, vmax, int(params.n_levels))
            line_levels = np.linspace(vmin, vmax, int(params.n_line_levels))
        else:
            levels = int(params.n_levels)
            line_levels = int(params.n_line_levels)

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        cf = ax.contourf(
            Xg * 1e3,
            Yg * 1e3,
            Zm,
            levels=levels,
            cmap=params.cmap,
            extend="both",
        )
        ax.contour(
            Xg * 1e3,
            Yg * 1e3,
            Zm,
            levels=line_levels,
            colors="k",
            linewidths=0.5,
            alpha=0.6,
        )

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"TOTAL {name} = BEM + Crack ({tag})")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Stress [MPa]")
        fig.savefig(out_dir / f"total_{name}_{tag}.png", dpi=params.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    _plot(Sxx_tot, Sxx_cr, "sxx")
    _plot(Syy_tot, Syy_cr, "syy")
    _plot(Sxy_tot, Sxy_cr, "sxy")

    return Sxx_tot, Syy_tot, Sxy_tot


# ============================================================
# End-of-run STEP-metrics plots
# ============================================================
def plot_step_metrics(
    step_history,
    out_dir,
    *,
    dpi: int = 200,
) -> dict:
    """Render the three end-of-run STEP plots and save the underlying arrays.

    ``step_history`` is a list of dicts produced by ``run_growth_steps``,
    with one row per STEP_NN containing ``alpha``, ``Q`` (topology
    metrics), and a per-tip ``tips`` list of {tip_vid, KI, KII}.

    Writes three PNGs into ``out_dir``:
      * ``metrics_alpha_vs_step.png``  -- damage parameter alpha vs STEP.
      * ``metrics_Q_vs_step.png``      -- network alignment factor Q vs STEP.
      * ``metrics_SIF_per_tip_vs_step.png`` -- one colored curve per tip vid.

    Also saves ``step_metrics.npz`` (steps, alpha, Q, P_infty, L_total_m,
    tip_vids, K_eff[step,tip] in MPa*sqrt(m)).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not step_history:
        print("[step-metrics] empty history; nothing to plot")
        return {}

    steps = np.array([int(r["step"]) for r in step_history], int)
    alpha = np.array([float(r.get("alpha", np.nan)) for r in step_history], float)
    Q     = np.array([float(r.get("Q", np.nan)) for r in step_history], float)
    Pinf  = np.array([float(r.get("P_infty", np.nan)) for r in step_history], float)
    Ltot  = np.array([float(r.get("L_total_m", np.nan)) for r in step_history], float)

    # Collect union of tip_vids and assemble a (n_steps, n_tips) K_eff array
    # in MPa*sqrt(m). Missing values are NaN (tip absent at that step).
    all_vids = []
    for row in step_history:
        for tip in row.get("tips", []) or []:
            all_vids.append(int(tip["tip_vid"]))
    tip_vids = sorted(set(all_vids))
    vid_to_col = {v: i for i, v in enumerate(tip_vids)}
    keff = np.full((len(steps), len(tip_vids)), np.nan, float)
    for si, row in enumerate(step_history):
        for tip in row.get("tips", []) or []:
            KI = float(tip.get("KI", np.nan))
            KII = float(tip.get("KII", np.nan))
            if np.isfinite(KI) and np.isfinite(KII):
                keff[si, vid_to_col[int(tip["tip_vid"])]] = (
                    float(np.sqrt(KI * KI + KII * KII)) * 1e-6
                )

    # --- alpha vs STEP
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=dpi)
    ax.plot(steps, alpha, marker="o", lw=1.4, color="tab:blue")
    ax.set_xlabel("STEP")
    ax.set_ylabel(r"Damage parameter $\alpha$")
    ax.set_title("Damage parameter vs STEP")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_alpha_vs_step.png", bbox_inches="tight")
    plt.close(fig)

    # --- Q vs STEP
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=dpi)
    ax.plot(steps, Q, marker="s", lw=1.4, color="tab:orange")
    ax.set_xlabel("STEP")
    ax.set_ylabel(r"Alignment factor $Q$")
    ax.set_title("Network alignment vs STEP")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_Q_vs_step.png", bbox_inches="tight")
    plt.close(fig)

    # --- per-tip K_eff vs STEP (one colored curve per tip vid)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=dpi)
    if tip_vids:
        cmap = plt.get_cmap("tab20" if len(tip_vids) > 10 else "tab10")
        for j, vid in enumerate(tip_vids):
            ax.plot(
                steps, keff[:, j],
                marker="o", lw=1.1, ms=3.0,
                color=cmap(j % cmap.N),
                label=f"tip {vid}",
            )
        if len(tip_vids) <= 20:
            ax.legend(loc="best", fontsize=8, ncols=2)
    ax.set_xlabel("STEP")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$ [MPa$\cdot\sqrt{m}$]")
    ax.set_title("SIF per deg-1 tip vs STEP")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_SIF_per_tip_vs_step.png", bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_dir / "step_metrics.npz",
        steps=steps,
        alpha=alpha,
        Q=Q,
        P_infty=Pinf,
        L_total_m=Ltot,
        tip_vids=np.asarray(tip_vids, int),
        K_eff_MPa_sqrt_m=keff,
    )
    print(
        f"[step-metrics] wrote alpha/Q/SIF plots + step_metrics.npz to {out_dir.name} "
        f"({len(steps)} steps, {len(tip_vids)} tip ids)"
    )
    return {
        "steps": steps,
        "alpha": alpha,
        "Q": Q,
        "P_infty": Pinf,
        "L_total_m": Ltot,
        "tip_vids": tip_vids,
        "K_eff_MPa_sqrt_m": keff,
    }


# ============================================================
# Hook factory (TOTAL ONLY)
# ============================================================
def make_standard_plot_hook(
    bem_dir,
    combined_out_dir,
    plot_params: PlotParams,
    show_initial=True,
    show_cycles=False,
    show_final=True,
    only_post_simplify: bool = True,
):
    combined_out_dir = Path(combined_out_dir)

    def _hook(tag, res, step_dir):

        # ---- Initial (STEP_00 in the flat driver, "initial" tag legacy)
        if tag == "initial":
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag="initial",
                params=plot_params,
                show=show_initial,
            )
            return

        # ---- Flat STEP_NN driver tag
        if tag.startswith("step_"):
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag=tag,
                params=plot_params,
                show=show_cycles,
            )
            return

        # ---- Legacy cycle_NN driver tag
        if tag.startswith("cycle_"):
            if only_post_simplify and (not tag.endswith("post_simplify")):
                return
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag=tag,
                params=plot_params,
                show=show_cycles,
            )
            return

        # ---- Final
        if tag == "final":
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag="final",
                params=plot_params,
                show=show_final,
            )
            return

    return _hook
