"""
excel_io.py -- load a VCEM_4_0 case configuration from an Excel workbook.

The workbook layout matches what ``input/_build_template.py`` emits:

    BEM, GEOMETRY                       : key / value / description rows
    CONFIGURATION_LOGIC                 : bool + string-valued run-mode knobs
    CONFIGURATION_PARAM                 : numerical (int/float) run-mode knobs
    CRACK_NETWORK                       : vertices in cols A-C, edges in
                                          cols E-F

Older workbooks with a single ``CONFIGURATION`` sheet are still accepted as
a fallback so legacy files keep loading.

Usage from a notebook:

    from input.excel_io import load_case, build_grid_points
    cfg = load_case("../input/disk_compression_2.xlsx")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from fracture_utils.Ubem.brazilian_disk_bem import BrazilianDiskParams
from fracture_utils.Ubem.disk_network_propagation import CrackGrowthParams
from fracture_utils.Ubem.disk_crack_plotting import PlotParams


# ============================================================
# Public dataclasses
# ============================================================

@dataclass
class GridSpec:
    """Stress-evaluation grid description loaded from the GEOMETRY sheet.

    ``source`` is one of:
      - ``"builtin_rect"``  : rectangular grid; uses n_grid + pad_frac
      - ``"builtin_polar"`` : polar grid; uses n_r + n_theta + pad_frac
      - ``"external_file"`` : (N,2) (x,y) nodes from a .csv or .npy file

    Note: for the current Brazilian-disk pipeline, ensure_bem_field always
    rebuilds its rectangular grid internally from
    BrazilianDiskParams.n_grid + pad_frac. Only ``builtin_rect`` is honored
    by the saved xs.npy/ys.npy arrays; the other modes are intended for
    out-of-band stress sampling via build_grid_points.
    """

    source: str = "builtin_rect"
    n_grid: int = 120
    pad_frac: float = 0.03
    n_r: int = 60
    n_theta: int = 120
    external_file: Optional[str] = None


@dataclass
class CaseConfig:
    # BEM / disk physics
    disk: BrazilianDiskParams
    # Outer boundary (build_boundary spec)
    boundary_spec: Dict[str, Any]
    # Stress-grid mesh
    grid: GridSpec
    # Initial crack network
    vertices: np.ndarray            # (Nv, 3) -- [id, x, y]
    connectivity: np.ndarray        # (Ne, 2) -- [v0, v1]
    # Run mode
    engine: str
    coupling_method: str
    # Total number of grow + solve steps. STEP_00 is the initial pre-grow
    # state; STEP_01 .. STEP_<max_steps> are growth increments. Replaces
    # the legacy (outer_cycles, crack_growth.max_cycles, max_inner_cycles_per_outer)
    # triple with a single flat counter.
    max_steps: int
    # Fire the BEM correction (iterative reverse-traction or direct/full-KKT
    # depending on coupling_method) every K steps. 0 disables correction
    # (equivalent to coupling_method='no_coupling').
    bem_correction_frequency: int
    # Kept for backward compatibility with workbooks built before max_steps;
    # the new STEP-based driver ignores this field.
    outer_cycles: int
    dry_run: bool
    skip_bem_solve: bool
    output_dir_name: str
    # Propagation
    crack_growth: CrackGrowthParams
    n_crack_elements: int
    parametrization: str
    # Crack DOF / junction formulation (controls rank deficiency of K)
    crack_mode: str                 # 'full' | 'half'
    junction_model: str             # 'strict' | 'core' | 'soft' (only used when crack_mode='half')
    soft_eta: float                 # penalty weight for junction_model='soft'
    # Outer-loop early termination when the disk has fractured (a polyline
    # spans from one boundary point to another -- further BEM solves are
    # unphysical).
    stop_on_spanning_cluster: bool
    # Augmented (direct-coupling) knobs
    aug_d_mode: str
    aug_gauss_n: int
    aug_enforce_crack_equilibrium: bool
    aug_equilibrate: bool
    aug_equilibrate_cols: bool
    aug_physical_scaling: bool
    aug_scale_traction_rows: bool
    aug_ridge_q: float
    aug_ridge_y: float
    aug_keep_bem_applied: bool
    # Plotting
    plot: PlotParams
    plot_show_initial: bool
    plot_show_cycles: bool
    plot_show_final: bool


# ============================================================
# Value coercion (Excel cells can come back as int/float/bool/str/NaN)
# ============================================================

def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        if isinstance(x, float) and np.isnan(x):
            return False
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f", "", "nan"):
        return False
    raise ValueError(f"Cannot interpret {x!r} as bool")


def _as_float(x: Any) -> float:
    return float(x)


def _as_int(x: Any) -> int:
    return int(round(float(x)))


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def _kv(sheet: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    if "key" not in sheet.columns or "value" not in sheet.columns:
        raise ValueError(
            f"Sheet {sheet_name!r} must have 'key' and 'value' columns. "
            f"Got: {list(sheet.columns)}"
        )
    keys = sheet["key"].astype(str).str.strip()
    return dict(zip(keys, sheet["value"]))


def _require(d: Dict[str, Any], key: str, sheet: str) -> Any:
    if key not in d:
        raise KeyError(f"Sheet {sheet!r} is missing required key: {key!r}")
    return d[key]


# ============================================================
# Loader
# ============================================================

def load_case(xlsx_path: str | Path) -> CaseConfig:
    """Load a CaseConfig from an .xlsx workbook produced by _build_template.py."""
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")

    # Required everywhere
    base_needed = {"BEM", "GEOMETRY", "CRACK_NETWORK"}
    missing = base_needed - set(sheets)
    if missing:
        raise ValueError(
            f"Workbook {xlsx_path.name} is missing required sheets: {sorted(missing)}"
        )

    # CONFIGURATION is now split across two sheets (LOGIC + PARAM). A single
    # legacy CONFIGURATION sheet is still accepted as a fallback.
    has_split = ("CONFIGURATION_LOGIC" in sheets) and ("CONFIGURATION_PARAM" in sheets)
    has_legacy = "CONFIGURATION" in sheets
    if not (has_split or has_legacy):
        raise ValueError(
            f"Workbook {xlsx_path.name} is missing CONFIGURATION sheets. "
            f"Expected either CONFIGURATION_LOGIC + CONFIGURATION_PARAM, "
            f"or a single legacy CONFIGURATION sheet."
        )

    bem = _kv(sheets["BEM"], "BEM")
    geo = _kv(sheets["GEOMETRY"], "GEOMETRY")
    if has_split:
        cfg_logic = _kv(sheets["CONFIGURATION_LOGIC"], "CONFIGURATION_LOGIC")
        cfg_param = _kv(sheets["CONFIGURATION_PARAM"], "CONFIGURATION_PARAM")
        # Disallow duplicate keys across the two sheets so a stray paste
        # can't silently override the other half.
        dup = set(cfg_logic).intersection(cfg_param)
        if dup:
            raise ValueError(
                f"Keys appear in both CONFIGURATION_LOGIC and "
                f"CONFIGURATION_PARAM: {sorted(dup)}"
            )
        cfg = {**cfg_logic, **cfg_param}
        cfg_sheet = "CONFIGURATION_LOGIC|CONFIGURATION_PARAM"
    else:
        cfg_sheet = "CONFIGURATION"
        cfg = _kv(sheets["CONFIGURATION"], cfg_sheet)

    # ----- BEM / disk params --------------------------------------------
    # boundary_mesh / boundary_concentration / boundary_taper_exponent are
    # optional (older workbooks predate them) -- default to a uniform circle.
    _bmesh = _as_str(bem.get("boundary_mesh", "uniform")).lower().strip() or "uniform"
    if _bmesh not in ("uniform", "clustered"):
        raise ValueError(
            f"BEM.boundary_mesh must be 'uniform' or 'clustered'; got {_bmesh!r}"
        )
    disk = BrazilianDiskParams(
        R=_as_float(_require(bem, "R_m", "BEM")),
        P_total=_as_float(_require(bem, "P_total_N", "BEM")),
        E=_as_float(_require(bem, "E_Pa", "BEM")),
        nu=_as_float(_require(bem, "nu", "BEM")),
        n_boundary_elements=_as_int(_require(bem, "n_boundary_elements", "BEM")),
        gauss_n=_as_int(_require(bem, "gauss_n", "BEM")),
        plane_strain=_as_bool(_require(bem, "plane_strain", "BEM")),
        arc_half_angle_deg=_as_float(_require(bem, "arc_half_angle_deg", "BEM")),
        h=_as_float(_require(bem, "h_thickness_m", "BEM")),
        # n_grid / pad_frac live on BrazilianDiskParams but are user-controlled
        # via the GEOMETRY sheet.
        n_grid=_as_int(_require(geo, "grid_n_grid", "GEOMETRY")),
        pad_frac=_as_float(_require(geo, "grid_pad_frac", "GEOMETRY")),
        boundary_mesh=_bmesh,
        boundary_concentration=_as_float(bem.get("boundary_concentration", 8.0)),
        boundary_taper_exponent=_as_float(bem.get("boundary_taper_exponent", 4.0)),
    )

    # ----- Step-based run-mode knobs (BEM sheet) ------------------------
    # max_steps: total number of growth STEPs (STEP_01..STEP_max_steps).
    # bem_correction_frequency: fire BEM correction every K steps.
    # Both optional for backward compatibility -- defaults from CaseConfig
    # field semantics: max_steps=20, frequency=1 (every step).
    _max_steps = _as_int(bem.get("max_steps", 20))
    _bem_freq = _as_int(bem.get("BEM_correction_frequency", 1))
    if _max_steps < 0:
        raise ValueError(f"BEM.max_steps must be >= 0; got {_max_steps}")
    if _bem_freq < 0:
        raise ValueError(f"BEM.BEM_correction_frequency must be >= 0; got {_bem_freq}")

    # ----- Boundary spec (build_boundary input) -------------------------
    btype = _as_str(_require(geo, "boundary_type", "GEOMETRY")).lower()
    boundary_spec: Dict[str, Any] = {
        "type": btype,
        "n_boundary": _as_int(_require(geo, "boundary_n_boundary", "GEOMETRY")),
        "center": (
            _as_float(_require(geo, "boundary_center_x", "GEOMETRY")),
            _as_float(_require(geo, "boundary_center_y", "GEOMETRY")),
        ),
    }
    if btype in ("circle", "circle_graded"):
        boundary_spec["R"] = _as_float(_require(geo, "boundary_R_m", "GEOMETRY"))
        # Sanity check: BEM and outer boundary share R_m for the disk pipeline.
        if not np.isclose(boundary_spec["R"], disk.R, rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"BEM.R_m ({disk.R}) and GEOMETRY.boundary_R_m "
                f"({boundary_spec['R']}) must match for the disk pipeline."
            )

    # ----- Grid spec ----------------------------------------------------
    ext_file = _as_str(geo.get("grid_external_file", "")) or None
    grid = GridSpec(
        source=_as_str(_require(geo, "grid_source", "GEOMETRY")).lower(),
        n_grid=_as_int(_require(geo, "grid_n_grid", "GEOMETRY")),
        pad_frac=_as_float(_require(geo, "grid_pad_frac", "GEOMETRY")),
        n_r=_as_int(_require(geo, "grid_n_r", "GEOMETRY")),
        n_theta=_as_int(_require(geo, "grid_n_theta", "GEOMETRY")),
        external_file=ext_file,
    )
    if grid.source not in ("builtin_rect", "builtin_polar", "external_file"):
        raise ValueError(
            f"grid_source must be one of "
            f"'builtin_rect' / 'builtin_polar' / 'external_file'; got {grid.source!r}"
        )

    # ----- Crack network ------------------------------------------------
    vertices, connectivity = _load_crack_network(sheets["CRACK_NETWORK"])

    # ----- CrackGrowthParams (subset; rest stays at dataclass defaults) -
    # disk_radius_m is injected from disk.R so the disk-radius step
    # (Δa_ref = f_disk_radius * R) is active in NetworkGrowthRunner.
    # crack_growth.Kc is the fracture toughness [Pa*sqrt(m)]; the legacy
    # workbook key was crack_growth.Kc_demo and is still accepted as a
    # fallback so pre-rename workbooks keep loading.
    #
    # crack_growth.max_cycles was the user-facing "inner cycles" knob for
    # the legacy outer x inner pipeline (maps to max_inner_cycles_per_outer).
    # The new STEP-based driver ignores it; the row was dropped from the
    # workbook template but a value is still consumed when present so old
    # workbooks continue to load.
    #
    # New simplifier knobs are read with `.get(...)` and dataclass defaults
    # as fallback so existing workbooks (built before these rows were added
    # to _build_template.py) continue to load. Once the workbook is
    # regenerated, the Excel-supplied value takes precedence.
    _simp_defaults = CrackGrowthParams()
    _legacy_inner = cfg.get("crack_growth.max_cycles", None)
    _kc = cfg.get("crack_growth.Kc",
                  cfg.get("crack_growth.Kc_demo", _simp_defaults.Kc))
    crack_growth = CrackGrowthParams(
        max_cycles=1,
        max_inner_cycles_per_outer=(_as_int(_legacy_inner)
                                    if _legacy_inner is not None else None),
        L_limit_mm=_as_float(_require(cfg, "crack_growth.L_limit_mm", cfg_sheet)),
        vertex_high=_as_int(_require(cfg, "crack_growth.vertex_high", cfg_sheet)),
        f_disk_radius=_as_float(_require(cfg, "crack_growth.f_disk_radius", cfg_sheet)),
        disk_radius_m=disk.R,
        simultaneous_tip_growth=_as_bool(_require(cfg, "crack_growth.simultaneous_tip_growth", cfg_sheet)),
        Kc=_as_float(_kc),
        rmax_frac=_as_float(_require(cfg, "crack_growth.rmax_frac", cfg_sheet)),
        min_pts=_as_int(_require(cfg, "crack_growth.min_pts", cfg_sheet)),
        simplify_each_step=_as_bool(_require(cfg, "crack_growth.simplify_each_step", cfg_sheet)),
        intersection_detect_mode=_as_str(_require(cfg, "crack_growth.intersection_detect_mode", cfg_sheet)),
        max_kink_deg=_as_float(_require(cfg, "crack_growth.max_kink_deg", cfg_sheet)),
        simp_max_angle_deg=_as_float(cfg.get("crack_growth.simp_max_angle_deg", _simp_defaults.simp_max_angle_deg)),
        simp_min_edge_mm=_as_float(cfg.get("crack_growth.simp_min_edge_mm", _simp_defaults.simp_min_edge_mm)),
        simp_max_merged_edge_mm=_as_float(cfg.get("crack_growth.simp_max_merged_edge_mm", _simp_defaults.simp_max_merged_edge_mm)),
        simp_merge_tolerance_m=_as_float(cfg.get("crack_growth.simp_merge_tolerance_m", _simp_defaults.simp_merge_tolerance_m)),
    )

    # ----- PlotParams ---------------------------------------------------
    plot = PlotParams(
        cmap=_as_str(_require(cfg, "plot.cmap", cfg_sheet)),
        match_limits_to_crack=_as_bool(_require(cfg, "plot.match_limits_to_crack", cfg_sheet)),
        robust=_as_bool(_require(cfg, "plot.robust", cfg_sheet)),
        robust_pct=_as_float(_require(cfg, "plot.robust_pct", cfg_sheet)),
        symmetric=_as_bool(_require(cfg, "plot.symmetric", cfg_sheet)),
        vmax_factor=_as_float(_require(cfg, "plot.vmax_factor", cfg_sheet)),
        n_levels=_as_int(_require(cfg, "plot.n_levels", cfg_sheet)),
        n_line_levels=_as_int(_require(cfg, "plot.n_line_levels", cfg_sheet)),
        dpi=_as_int(_require(cfg, "plot.dpi", cfg_sheet)),
    )

    return CaseConfig(
        disk=disk,
        boundary_spec=boundary_spec,
        grid=grid,
        vertices=vertices,
        connectivity=connectivity,
        engine=_as_str(_require(cfg, "engine", cfg_sheet)).lower(),
        coupling_method=_as_str(_require(cfg, "coupling_method", cfg_sheet)).lower(),
        max_steps=_max_steps,
        bem_correction_frequency=_bem_freq,
        # outer_cycles is dead weight under the STEP-based driver (the loop
        # uses max_steps directly). Default to 1 so post-rebuild workbooks
        # that omit the row still load; legacy workbooks that still carry
        # the row pass their value through.
        outer_cycles=_as_int(cfg.get("outer_cycles", 1)),
        dry_run=_as_bool(_require(cfg, "dry_run", cfg_sheet)),
        skip_bem_solve=_as_bool(_require(cfg, "skip_bem_solve", cfg_sheet)),
        output_dir_name=_as_str(_require(cfg, "output_dir_name", cfg_sheet)),
        crack_growth=crack_growth,
        n_crack_elements=_as_int(_require(cfg, "solver_kwargs.n_crack_elements", cfg_sheet)),
        # Optional in older workbooks; defaults to 'polyline' (piecewise-linear
        # BEM panels with kink refinement at every interior vertex).
        parametrization=_as_str(cfg.get("solver_kwargs.parametrization", "polyline")).lower() or "polyline",
        # 'full' bonds COD across junctions (no jump DOFs -> K is generally
        # full-column-rank). 'half' adds per-side junction DOFs (creates a
        # structural column-rank deficit absorbed by junction continuity
        # equations in C or by soft penalty rows depending on junction_model).
        # 'auto' lets the driver pick per outer cycle based on whether any
        # deg>=2 vertex exists in the current network.
        crack_mode=_as_str(cfg.get("solver_kwargs.crack_mode", "auto")).lower() or "auto",
        # Only used when crack_mode='half'. 'strict' = one shared jump DOF
        # per junction vertex (smallest column count, hardest conditioning).
        # 'core' = per-branch-end DOFs with hard continuity equations in C.
        # 'soft' = per-branch-end DOFs with weighted LS continuity rows
        # appended to K (no constraint deficit, well-conditioned).
        junction_model=_as_str(cfg.get("solver_kwargs.junction_model", "strict")).lower() or "strict",
        # Penalty weight for junction_model='soft'. Larger eta -> tighter
        # continuity at the cost of conditioning. Ignored otherwise.
        soft_eta=_as_float(cfg.get("solver_kwargs.soft_eta", 1.0)),
        # Stop the outer loop as soon as a polyline spans the disk boundary
        # (the disk has fractured; further inner solves are unphysical).
        # Default True even on older workbooks because the bad-K_eff regime
        # past fragmentation is uniformly undesirable.
        stop_on_spanning_cluster=_as_bool(cfg.get(
            "crack_growth.stop_on_spanning_cluster", True)),
        aug_d_mode=_as_str(_require(cfg, "augmented.d_mode", cfg_sheet)),
        aug_gauss_n=_as_int(_require(cfg, "augmented.gauss_n", cfg_sheet)),
        aug_enforce_crack_equilibrium=_as_bool(_require(cfg, "augmented.enforce_crack_equilibrium", cfg_sheet)),
        aug_equilibrate=_as_bool(_require(cfg, "augmented.equilibrate", cfg_sheet)),
        aug_equilibrate_cols=_as_bool(_require(cfg, "augmented.equilibrate_cols", cfg_sheet)),
        aug_physical_scaling=_as_bool(_require(cfg, "augmented.physical_scaling", cfg_sheet)),
        aug_scale_traction_rows=_as_bool(_require(cfg, "augmented.scale_traction_rows", cfg_sheet)),
        aug_ridge_q=_as_float(_require(cfg, "augmented.ridge_q", cfg_sheet)),
        aug_ridge_y=_as_float(_require(cfg, "augmented.ridge_y", cfg_sheet)),
        aug_keep_bem_applied=_as_bool(_require(cfg, "augmented.keep_bem_applied", cfg_sheet)),
        plot=plot,
        plot_show_initial=_as_bool(_require(cfg, "plot.show_initial", cfg_sheet)),
        plot_show_cycles=_as_bool(_require(cfg, "plot.show_cycles", cfg_sheet)),
        plot_show_final=_as_bool(_require(cfg, "plot.show_final", cfg_sheet)),
    )


# ============================================================
# CRACK_NETWORK sheet: vertices in cols A-C, edges in cols E-F
# ============================================================

def _load_crack_network(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Pick out the vertex and edge tables from the CRACK_NETWORK sheet.

    Layout (matches _build_template.py):
      columns A-C : id, x_m, y_m   (vertices)
      columns E-F : v0, v1         (edges)
    Empty rows are tolerated.
    """
    # Locate columns by name -- robust to extra blank-column padding.
    by_name = {str(c).strip(): c for c in df.columns}
    v_cols = ["id", "x_m", "y_m"]
    if not all(c in by_name for c in v_cols):
        raise ValueError(
            f"CRACK_NETWORK sheet must contain vertex columns "
            f"{v_cols}. Got: {list(df.columns)}"
        )
    df_v = df[[by_name[c] for c in v_cols]].dropna(how="any")
    if len(df_v) == 0:
        raise ValueError("CRACK_NETWORK sheet has no vertex rows.")
    vertices = np.column_stack([
        df_v[by_name["id"]].astype(int).values,
        df_v[by_name["x_m"]].astype(float).values,
        df_v[by_name["y_m"]].astype(float).values,
    ])

    e_cols = ["v0", "v1"]
    if all(c in by_name for c in e_cols):
        df_e = df[[by_name[c] for c in e_cols]].dropna(how="any")
        if len(df_e):
            connectivity = np.column_stack([
                df_e[by_name["v0"]].astype(int).values,
                df_e[by_name["v1"]].astype(int).values,
            ])
        else:
            connectivity = np.zeros((0, 2), int)
    else:
        connectivity = np.zeros((0, 2), int)

    return vertices, connectivity


# ============================================================
# Optional: build (x, y) grid points for out-of-band stress sampling
# ============================================================

def save_supplementary_grid(
    cfg: CaseConfig,
    bem_dir: str | Path,
    *,
    solver=None,
    show: bool = True,
    always_polar: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """Evaluate the BEM stress on cfg.grid (polar / external) and save it
    alongside the rect arrays already produced by ensure_bem_field.

    No-op when cfg.grid.source == 'builtin_rect' AND ``always_polar`` is
    False (the rect grid IS the primary grid; nothing supplementary is
    needed).

    When ``always_polar=True``, samples a polar grid regardless of
    ``cfg.grid.source`` using ``cfg.grid.n_r`` / ``cfg.grid.n_theta`` /
    ``cfg.grid.pad_frac``. The original ``cfg.grid`` is left untouched.
    The polar arrays are then picked up by the plotting layer
    (``plot_total_field`` and the baseline polar contour PNGs) so the
    disk boundary is rendered as an exact circle.

    When cfg.grid.source == 'builtin_polar' (or always_polar overrides
    a rect/external source):
      saves rs.npy (n_r,), thetas.npy (n_theta,),
      xs_polar.npy / ys_polar.npy / Sxx_polar.npy / Syy_polar.npy /
      Sxy_polar.npy of shape (n_theta, n_r).

    When cfg.grid.source == 'external_file' (and not always_polar):
      saves xs_polar.npy / ys_polar.npy of shape (N,) and matching
      Sxx_polar / Syy_polar / Sxy_polar of shape (N,).
      (Same '_polar' suffix is used for both non-rect modes for
      consistency; the shape tells you which kind it is.)

    Parameters
    ----------
    solver
        Optional pre-built BEM solver returned by
        compute_bem_brazilian_disk_field. If None, a fresh BEM solve is
        run (with save_arrays=False, save_contours=False) just to get a
        solver. Passing the already-solved one in avoids a duplicate
        BEM solve.
    always_polar
        Force a polar overlay even when ``cfg.grid.source ==
        'builtin_rect'``. Used by the run drivers so the per-cycle TOTAL
        plots can always render on the polar grid (smooth disk boundary)
        regardless of what primary grid the user selected.

    Returns
    -------
    dict | None
        The supplementary arrays (None when source='builtin_rect' and
        always_polar is False).
    """
    polar_mode = (cfg.grid.source == "builtin_polar") or always_polar

    if cfg.grid.source == "builtin_rect" and not always_polar:
        return None

    bem_dir = Path(bem_dir)
    bem_dir.mkdir(parents=True, exist_ok=True)

    if solver is None:
        from fracture_utils.Ubem.brazilian_disk_bem import (
            compute_bem_brazilian_disk_field,
        )
        solver, _field = compute_bem_brazilian_disk_field(
            cfg.disk, bem_dir,
            show=False, save_arrays=False, save_contours=False,
        )

    # Choose evaluation points: forced polar overlay rebuilds them from
    # cfg.grid.n_r/n_theta directly so an underlying source='builtin_rect'
    # still produces a polar mesh. The non-forced branches keep using
    # build_grid_points so the existing 'builtin_polar' / 'external_file'
    # behaviour is unchanged.
    if always_polar and cfg.grid.source != "builtin_polar":
        n_r = int(cfg.grid.n_r)
        n_t = int(cfg.grid.n_theta)
        rs = np.linspace(0.0, cfg.disk.R * (1.0 - float(cfg.grid.pad_frac)), n_r)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
        Rg, Tg = np.meshgrid(rs, thetas, indexing="xy")
        xs_grid = Rg * np.cos(Tg)
        ys_grid = Rg * np.sin(Tg)
        pts = np.column_stack([xs_grid.ravel(), ys_grid.ravel()])
    else:
        pts = build_grid_points(cfg)              # (N, 2)

    Sxx = np.full(len(pts), np.nan, dtype=float)
    Syy = np.full(len(pts), np.nan, dtype=float)
    Sxy = np.full(len(pts), np.nan, dtype=float)
    for i, (x, y) in enumerate(pts):
        try:
            a, b, c = solver.compute_stress_at_point(float(x), float(y))
            Sxx[i] = a
            Syy[i] = b
            Sxy[i] = c
        except Exception:
            pass

    out: Dict[str, np.ndarray] = {}

    if polar_mode:
        n_r = int(cfg.grid.n_r)
        n_t = int(cfg.grid.n_theta)
        rs = np.linspace(0.0, cfg.disk.R * (1.0 - float(cfg.grid.pad_frac)), n_r)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
        Rg, Tg = np.meshgrid(rs, thetas, indexing="xy")    # (n_t, n_r)
        xs_p = Rg * np.cos(Tg)
        ys_p = Rg * np.sin(Tg)
        Sxx_p = Sxx.reshape(n_t, n_r)
        Syy_p = Syy.reshape(n_t, n_r)
        Sxy_p = Sxy.reshape(n_t, n_r)

        np.save(bem_dir / "rs.npy", rs)
        np.save(bem_dir / "thetas.npy", thetas)
        np.save(bem_dir / "xs_polar.npy", xs_p)
        np.save(bem_dir / "ys_polar.npy", ys_p)
        np.save(bem_dir / "Sxx_polar.npy", Sxx_p)
        np.save(bem_dir / "Syy_polar.npy", Syy_p)
        np.save(bem_dir / "Sxy_polar.npy", Sxy_p)

        out.update(rs=rs, thetas=thetas, xs=xs_p, ys=ys_p,
                   Sxx=Sxx_p, Syy=Syy_p, Sxy=Sxy_p)
        n_valid = int(np.isfinite(Sxx_p).sum())
        tag = "polar (forced)" if always_polar and cfg.grid.source != "builtin_polar" else "polar"
        print(f"[{tag}] saved (n_theta={n_t}, n_r={n_r}); "
              f"{n_valid}/{n_t*n_r} points evaluated.")
    else:  # external_file
        np.save(bem_dir / "xs_polar.npy", pts[:, 0])
        np.save(bem_dir / "ys_polar.npy", pts[:, 1])
        np.save(bem_dir / "Sxx_polar.npy", Sxx)
        np.save(bem_dir / "Syy_polar.npy", Syy)
        np.save(bem_dir / "Sxy_polar.npy", Sxy)
        out.update(xs=pts[:, 0], ys=pts[:, 1], Sxx=Sxx, Syy=Syy, Sxy=Sxy)
        n_valid = int(np.isfinite(Sxx).sum())
        print(f"[external] saved {len(pts)} scattered points; "
              f"{n_valid} evaluated.")

    if show:
        plot_supplementary_grid(cfg, bem_dir, arrays=out,
                                force_polar=polar_mode, show=True)

    return out


def plot_supplementary_grid(
    cfg: CaseConfig,
    bem_dir: str | Path,
    *,
    arrays: Optional[Dict[str, np.ndarray]] = None,
    force_polar: bool = False,
    show: bool = False,
) -> None:
    """Save brazilian_disk_Sxx_polar.png / _Syy_polar.png / _Sxy_polar.png
    contour plots of the supplementary grid produced by
    save_supplementary_grid.

    For polar grids, uses contourf on the (xs_p, ys_p) Cartesian
    projection so the plot shape matches the disk, and overlays
    ``cfg.plot.n_line_levels`` contour lines (matching the rect baseline
    ``brazilian_disk_<comp>_contour.png`` style). For external_file
    (scattered) grids, uses tricontourf.

    Set ``force_polar=True`` when the caller emitted polar arrays via
    ``save_supplementary_grid(always_polar=True)`` on top of a
    ``builtin_rect`` source -- otherwise the auto-load branch sees the
    rect source and bails.
    """
    import matplotlib.pyplot as plt
    from matplotlib import tri as _tri

    bem_dir = Path(bem_dir)

    if arrays is None:
        if cfg.grid.source == "builtin_polar" or force_polar:
            arrays = {
                "xs": np.load(bem_dir / "xs_polar.npy"),
                "ys": np.load(bem_dir / "ys_polar.npy"),
                "Sxx": np.load(bem_dir / "Sxx_polar.npy"),
                "Syy": np.load(bem_dir / "Syy_polar.npy"),
                "Sxy": np.load(bem_dir / "Sxy_polar.npy"),
            }
        elif cfg.grid.source == "external_file":
            arrays = {
                "xs": np.load(bem_dir / "xs_polar.npy"),
                "ys": np.load(bem_dir / "ys_polar.npy"),
                "Sxx": np.load(bem_dir / "Sxx_polar.npy"),
                "Syy": np.load(bem_dir / "Syy_polar.npy"),
                "Sxy": np.load(bem_dir / "Sxy_polar.npy"),
            }
        else:
            return

    xs = arrays["xs"]
    ys = arrays["ys"]
    is_polar = (xs.ndim == 2)

    # Close the angular seam at theta=0 / 2pi on the polar grid so contourf
    # doesn't leave a wedge-shaped gap along +x. Polar arrays are stored
    # with theta in [0, 2pi) (endpoint=False), so row 0 and row -1 are not
    # coincident — append row 0 at the end to stitch them.
    if is_polar and not (np.allclose(xs[0, :], xs[-1, :])
                         and np.allclose(ys[0, :], ys[-1, :])):
        xs = np.vstack([xs, xs[:1, :]])
        ys = np.vstack([ys, ys[:1, :]])
        _wrap_polar = True
    else:
        _wrap_polar = False

    # Match the styling of the legacy rect "sigma_xx" baseline plots so the
    # field plots before STEP_00 are visually consistent with the per-step
    # TOTAL polar plots: axes in mm, "Stress [MPa]" colorbar, symmetric
    # robust limits, sigma_xx-style title.
    xs_mm = xs * 1e3
    ys_mm = ys * 1e3
    comp_titles = {"Sxx": "σ_xx", "Syy": "σ_yy", "Sxy": "σ_xy"}
    for name, S in (("Sxx", arrays["Sxx"]),
                    ("Syy", arrays["Syy"]),
                    ("Sxy", arrays["Sxy"])):
        fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=int(cfg.plot.dpi))
        S_mpa = np.asarray(S, dtype=float) * 1e-6
        if _wrap_polar:
            S_mpa = np.vstack([S_mpa, S_mpa[:1, :]])
        if cfg.plot.robust and np.any(np.isfinite(S_mpa)):
            vmax = float(np.nanpercentile(np.abs(S_mpa), float(cfg.plot.robust_pct)))
        else:
            finite = S_mpa[np.isfinite(S_mpa)]
            vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmin = -vmax if cfg.plot.symmetric else float(np.nanmin(S_mpa))
        levels = np.linspace(vmin, vmax, int(cfg.plot.n_levels) + 1)
        line_levels = np.linspace(vmin, vmax, int(cfg.plot.n_line_levels))
        if is_polar:
            cs = ax.contourf(xs_mm, ys_mm, S_mpa, levels=levels,
                             cmap=cfg.plot.cmap, extend="both")
            ax.contour(xs_mm, ys_mm, S_mpa, levels=line_levels,
                       colors="k", linewidths=0.5, alpha=0.6)
        else:
            mask = np.isfinite(S_mpa)
            triang = _tri.Triangulation(xs_mm[mask], ys_mm[mask])
            cs = ax.tricontourf(triang, S_mpa[mask], levels=levels,
                                cmap=cfg.plot.cmap, extend="both")
            ax.tricontour(triang, S_mpa[mask], levels=line_levels,
                          colors="k", linewidths=0.5, alpha=0.6)
        ax.set_aspect("equal", "box")
        ax.set_title(comp_titles[name])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(True, alpha=0.25)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.92)
        cbar.set_label("Stress [MPa]")
        out_path = bem_dir / f"brazilian_disk_{name}_polar.png"
        fig.savefig(out_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def build_grid_points(cfg: CaseConfig) -> np.ndarray:
    """Return an (N, 2) array of (x, y) grid points.

    Used when you want to sample the stress field outside of
    ensure_bem_field (which always builds its own rectangular grid from
    BrazilianDiskParams.n_grid + pad_frac).

    - ``builtin_rect``  -> [-R*(1+pad), R*(1+pad)] x [-R*(1+pad), R*(1+pad)],
                           n_grid points per side. Same nodes
                           ensure_bem_field uses for the disk pipeline.
    - ``builtin_polar`` -> r in [0, R*(1-pad)], theta in [0, 2 pi),
                           n_r x n_theta nodes.
    - ``external_file`` -> loads (N, 2) (x, y) from .csv (skiprows=1)
                           or .npy.
    """
    g = cfg.grid
    R = cfg.disk.R
    if g.source == "builtin_rect":
        n = int(g.n_grid)
        ext = R * (1.0 + float(g.pad_frac))
        xs = np.linspace(-ext, ext, n)
        ys = np.linspace(-ext, ext, n)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([X.ravel(), Y.ravel()])
    if g.source == "builtin_polar":
        n_r = int(g.n_r)
        n_t = int(g.n_theta)
        rs = np.linspace(0.0, R * (1.0 - float(g.pad_frac)), n_r)
        ts = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
        Rg, Tg = np.meshgrid(rs, ts, indexing="xy")
        return np.column_stack([(Rg * np.cos(Tg)).ravel(),
                                (Rg * np.sin(Tg)).ravel()])
    if g.source == "external_file":
        if not g.external_file:
            raise ValueError(
                "grid_source='external_file' requires a non-empty "
                "grid_external_file path in the GEOMETRY sheet."
            )
        p = Path(g.external_file)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"External grid file not found: {p}")
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
        else:
            arr = np.loadtxt(p, delimiter=",", skiprows=1)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"External grid file {p} must produce an (N, 2) array of "
                f"(x, y); got shape {arr.shape}"
            )
        return arr
    raise ValueError(f"Unknown grid source: {g.source!r}")
