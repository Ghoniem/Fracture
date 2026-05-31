"""
_build_template.py -- generate the starter Excel template for VCEM_4_0 case
inputs.

Run from this directory:
    python _build_template.py [case_name]

Produces ``<case_name>.xlsx`` (default: ``disk_compression_2.xlsx``) with five
sheets that ``excel_io.load_case`` understands:

  * BEM                  -- material data, BEM mesh, boundary loading, boundary
                            conditions (the latter are implicit for the disk:
                            derived from P_total + arc).
  * GEOMETRY             -- outside-boundary spec + stress-grid spec
                            (grid_source = builtin_rect | builtin_polar |
                             external_file).
  * CRACK_NETWORK        -- initial crack network: vertices (id, x_m, y_m) in
                            columns A-C and edges (v0, v1) in columns E-F.
  * CONFIGURATION_LOGIC  -- non-numerical run-mode knobs (booleans + string
                            choices: engine, coupling_method, parametrization,
                            d_mode, cmap, output_dir_name, ...).
  * CONFIGURATION_PARAM  -- numerical knobs (ints + floats: outer_cycles,
                            crack-growth lengths/angles, augmented ridges,
                            plot DPI / level counts, ...).

Re-run this script after defaults change in the underlying dataclasses;
hand edits to the .xlsx will be overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Common option strings ------------------------------------------------
BOOL  = "True | False"
FLOAT = ""            # any real number
INT   = ""            # any integer
TEXT  = ""            # free-form text

# ---------------------------------------------------------------- BEM
BEM_ROWS = [
    # key, value, description, options
    ("R_m",                 0.0127,    "Disk radius [m] (= 25.4 mm / 2)",                          FLOAT),
    ("P_total_N",           3800.0,    "Total compressive force per platen [N]",                   FLOAT),
    ("E_Pa",                2.3152e11, "Young's modulus [Pa]",                                     FLOAT),
    ("nu",                  0.3,       "Poisson's ratio",                                          FLOAT),
    ("h_thickness_m",       6.35e-3,   "Disk thickness [m]",                                       FLOAT),
    ("plane_strain",        True,      "Plane strain (True) / plane stress (False)",               BOOL),
    ("n_boundary_elements", 120,       "Number of boundary elements on the outer circle",          INT),
    ("gauss_n",             4,         "Gauss-Legendre points per BEM element",                    INT),
    ("arc_half_angle_deg",  15.0,      "Half-angle [deg] of the loaded arc (top + bottom)",        FLOAT),
    ("boundary_mesh",         "uniform", "Boundary-node distribution on the outer circle. "
                                         "'clustered' concentrates nodes at the +/-90 deg platens "
                                         "and tapers toward the equator.",                          "uniform | clustered"),
    ("boundary_concentration", 8.0,     "(clustered) Peak/baseline density ratio at the platens",   FLOAT),
    ("boundary_taper_exponent",4.0,     "(clustered) Taper rate; larger = sharper concentration",   FLOAT),
    ("max_steps",            20,        "Total number of grow + solve STEPS. STEP_00 = initial state, "
                                         "STEP_01..STEP_<max_steps> = grow increments.",              INT),
    ("BEM_correction_frequency", 1,    "Fire BEM correction every K steps. 0 disables (no_coupling).", INT),
    # BCs for the disk pipeline are derived inside ensure_bem_field from
    # P_total + arc (top/bottom pressure_normal). Future support for
    # explicit BCSpec blocks will live in a dedicated BC_BLOCKS sheet.
]

# ----------------------------------------------------------- GEOMETRY
GEO_ROWS = [
    ("boundary_type",       "circle",        "Outer boundary type",
        "circle | circle_graded | rectangle | polygon"),
    ("boundary_R_m",        0.0127,          "Radius [m] (used by circle / circle_graded; must equal BEM.R_m for the disk pipeline)",
        FLOAT),
    ("boundary_n_boundary", 120,             "Number of boundary segments on the outer boundary",
        INT),
    ("boundary_center_x",   0.0,             "Outer boundary center x [m]",
        FLOAT),
    ("boundary_center_y",   0.0,             "Outer boundary center y [m]",
        FLOAT),
    ("grid_source",         "builtin_rect",  "Stress-grid source",
        "builtin_rect | builtin_polar | external_file"),
    ("grid_n_grid",         120,             "(builtin_rect) Grid points per side",
        INT),
    ("grid_pad_frac",       0.03,            "(builtin_rect / builtin_polar) Inside-mask / outer-padding fraction of R",
        FLOAT),
    ("grid_n_r",            60,              "(builtin_polar) Radial points",
        INT),
    ("grid_n_theta",        120,             "(builtin_polar) Angular points",
        INT),
    ("grid_external_file",  "",              "(external_file) Path to (N,2) (x,y) nodes file (.csv or .npy); relative paths resolve to this input/ directory",
        TEXT),
]

# ----------------------------------------------------- CRACK_NETWORK
# Two crack segments meeting at the origin (Y-junction):
#   e1: vertex 0 (0,0) -> vertex 1 ( 2 mm,  2 mm)
#   e2: vertex 0 (0,0) -> vertex 2 (-2 mm, -2 mm)
mm = 1e-3
VERTICES = np.array(
    [
        [0,  0.0,        0.0       ],
        [1,  2.0 * mm,   2.0 * mm  ],
        [2, -2.0 * mm,  -2.0 * mm  ],
    ],
    dtype=float,
)
EDGES = np.array(
    [
        [0, 1],
        [0, 2],
    ],
    dtype=int,
)

# --------------------------------------------- CONFIGURATION_LOGIC
# Non-numerical knobs: bool toggles + string choices / free text.
CFG_LOGIC_ROWS = [
    # ----- run mode --------------------------------------------------
    ("engine",                                "cpp",         "Solver backend",
        "python | cpp"),
    ("coupling_method",                       "iterative",   "Outer-cycle update strategy",
        "no_coupling | iterative | direct"),
    ("dry_run",                               False,         "If True, define the run function but do not invoke it",
        BOOL),
    ("skip_bem_solve",                        False,         "If True, load BEM arrays from disk instead of recomputing",
        BOOL),
    ("output_dir_name",                       "BEM_Brazilian_disk_iterative", "Subdir name under repo_root/output/",
        TEXT),

    # ----- crack growth (bool/string subset of CrackGrowthParams) ----
    ("crack_growth.simultaneous_tip_growth",  True,          "Grow all tips simultaneously each step",
        BOOL),
    ("crack_growth.simplify_each_step",       False,         "Run per-polyline segment-merge simplifier at the end of each outer cycle (before the outer solve). Currently a no-op; kept for workbook back-compat.",
        BOOL),
    ("crack_growth.intersection_detect_mode", "single_pass", "Intersection detection mode",
        "single_pass | iterative"),

    # ----- one-way base solver_kwargs (string choice) ---------------
    ("solver_kwargs.parametrization",         "polyline",    "Per-polyline geometry the BEM solver assumes. 'polyline' keeps piecewise-linear segments with kink refinement at every interior vertex. 'cspline' fits a natural cubic spline (heavier; kept for opt-in experiments).",
        "polyline | cspline"),
    ("solver_kwargs.crack_mode",              "auto",        "Crack DOF formulation. 'full' bonds COD across junctions (K is full column rank, well conditioned). 'half' allocates per-side COD DOFs at junctions (creates a structural column-rank deficit that the KKT constraints must absorb; needed when branches open independently at a junction). 'auto' picks per outer cycle: 'full' while every vertex is deg<=1, 'half' once any deg>=2 vertex appears (first intersection / junction).",
        "auto | full | half"),
    ("solver_kwargs.junction_model",          "strict",      "Junction DOF model (used only when crack_mode='half'). 'strict' shares one jump DOF per junction vertex (smallest column count). 'core' allocates per-branch-end DOFs with hard continuity rows in C. 'soft' appends weighted-LS continuity rows to K instead of C (no column-rank deficit, best conditioning).",
        "strict | core | soft"),

    # ----- augmented (direct-coupling) bools + mode -----------------
    ("augmented.d_mode",                      "correction_zero", "Augmented displacement mode",
        "correction_zero | external_total"),
    ("augmented.enforce_crack_equilibrium",   True,          "Enforce crack-equilibrium row in augmented KKT",
        BOOL),
    ("augmented.equilibrate",                 True,          "Row-equilibration of augmented block",
        BOOL),
    ("augmented.equilibrate_cols",            False,         "Column-equilibration of augmented block",
        BOOL),
    ("augmented.physical_scaling",            True,          "Physical pre-scaling of augmented block",
        BOOL),
    ("augmented.scale_traction_rows",         True,          "Scale traction rows in augmented block",
        BOOL),
    ("augmented.keep_bem_applied",            True,          "Keep BEM-applied loads when full-KKT",
        BOOL),

    # ----- plotting (string choice + bool toggles) ------------------
    ("plot.cmap",                             "jet",         "Matplotlib colormap",
        "any matplotlib colormap (jet, viridis, plasma, magma, RdBu_r, coolwarm, ...)"),
    ("plot.match_limits_to_crack",            True,          "Match contour limits to crack-region values",
        BOOL),
    ("plot.robust",                           True,          "Use robust (percentile) clipping",
        BOOL),
    ("plot.symmetric",                        True,          "Symmetric colorbar about zero",
        BOOL),
    ("plot.show_initial",                     True,          "Show the initial-state plot",
        BOOL),
    ("plot.show_cycles",                      True,          "Show per-cycle plots",
        BOOL),
    ("plot.show_final",                       True,          "Show the final-state plot",
        BOOL),
]

# --------------------------------------------- CONFIGURATION_PARAM
# Numerical knobs only: ints + floats.
CFG_PARAM_ROWS = [
    # ----- run mode --------------------------------------------------
    ("outer_cycles",                          2,             "Number of outer coupling cycles",
        INT),

    # ----- crack growth (numeric subset of CrackGrowthParams) -------
    ("crack_growth.max_cycles",               5,             "Number of growth steps per outer coupling cycle (caps the inner growth loop)",
        INT),
    ("crack_growth.L_limit_mm",               20.0,          "Max total network length [mm]",
        FLOAT),
    ("crack_growth.vertex_high",              18,            "Max vertex id reserved (network growth cap)",
        INT),
    ("crack_growth.f_disk_radius",            0.05,          "Growth step size as fraction of disk radius (Δa_ref = f_disk_radius × R, constant in absolute terms)",
        FLOAT),
    ("crack_growth.Kc_demo",                  1e6,           "Demo fracture toughness [Pa*sqrt(m)]",
        FLOAT),
    ("crack_growth.rmax_frac",                0.25,          "Max growth radius / current length fraction",
        FLOAT),
    ("crack_growth.min_pts",                  8,             "Min sampling points along a crack segment",
        INT),
    ("crack_growth.max_kink_deg",             20.0,          "Soft clamp on segment-to-segment kink angle [deg]; bypassed on each tip's first emission. <= 0 disables.",
        FLOAT),

    # ----- end-of-outer-cycle simplifier knobs (currently unused) ---
    # Retained for workbook back-compat after the per-polyline simplifier
    # was removed. Values are read but ignored by the runner.
    ("crack_growth.simp_max_angle_deg",       3.0,           "Max kink angle [deg] below which two adjacent segments are considered collinear and merged.",
        FLOAT),
    ("crack_growth.simp_min_edge_mm",         0.2,           "Edges shorter than this [mm] are candidates for merging with a neighbour.",
        FLOAT),
    ("crack_growth.simp_max_merged_edge_mm",  5.0,           "Cap on the length [mm] of any merged edge; protects against over-aggressive coalescence on long branches.",
        FLOAT),
    ("crack_growth.simp_merge_tolerance_m",   1e-6,          "Vertex-coincidence merge tolerance [m].",
        FLOAT),

    # ----- one-way base solver_kwargs (numeric) ---------------------
    ("solver_kwargs.n_crack_elements",        40,            "Elements per crack segment (one-way KKT)",
        INT),
    ("solver_kwargs.soft_eta",                1.0,           "Penalty weight for junction_model='soft' (ignored otherwise). Larger eta -> tighter junction continuity at the cost of conditioning; ~1.0 is a sensible neutral default.",
        FLOAT),

    # ----- augmented (direct-coupling) numeric ----------------------
    ("augmented.gauss_n",                     4,             "Gauss-Legendre points for augmented payload",
        INT),
    ("augmented.ridge_q",                     1e-4,          "Ridge regularization on q block",
        FLOAT),
    ("augmented.ridge_y",                     1e-8,          "Ridge regularization on y block",
        FLOAT),

    # ----- plotting (numeric) ---------------------------------------
    ("plot.robust_pct",                       97.0,          "Robust percentile cutoff",
        FLOAT),
    ("plot.vmax_factor",                      2.0,           "Multiplier on auto vmax",
        FLOAT),
    ("plot.n_levels",                         60,            "Number of filled contour bands",
        INT),
    ("plot.n_line_levels",                    15,            "Number of overlaid contour lines",
        INT),
    ("plot.dpi",                              300,           "Figure DPI",
        INT),
]


def _df(rows, cols=("key", "value", "description", "options")):
    return pd.DataFrame(rows, columns=list(cols))


def build(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_bem       = _df(BEM_ROWS)
    df_geo       = _df(GEO_ROWS)
    df_cfg_logic = _df(CFG_LOGIC_ROWS)
    df_cfg_param = _df(CFG_PARAM_ROWS)
    df_v = pd.DataFrame(VERTICES, columns=["id", "x_m", "y_m"])
    df_v["id"] = df_v["id"].astype(int)
    df_e = pd.DataFrame(EDGES, columns=["v0", "v1"]).astype(int)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_bem.to_excel(xw, sheet_name="BEM", index=False)
        df_geo.to_excel(xw, sheet_name="GEOMETRY", index=False)
        df_v.to_excel(xw, sheet_name="CRACK_NETWORK", index=False, startcol=0)
        df_e.to_excel(xw, sheet_name="CRACK_NETWORK", index=False, startcol=4)
        df_cfg_logic.to_excel(xw, sheet_name="CONFIGURATION_LOGIC", index=False)
        df_cfg_param.to_excel(xw, sheet_name="CONFIGURATION_PARAM", index=False)

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "case_name",
        nargs="?",
        default="disk_compression_2",
        help="Output filename stem (default: disk_compression_2)",
    )
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / f"{args.case_name}.xlsx"
    build(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
