"""
_build_simulations_excel.py -- emit SimulationsCpp.ipynb (Excel-driven version).

Re-run after editing any of the cell-source blocks below.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "SimulationsCpp.ipynb"

# ---------------------------------------------------------------- cells

CELL0_MD = """\
# Simulations (Excel-driven, C++/Python switchable) -- VCEM_4_0

Brazilian-disk + crack-network coupling, driven entirely by an Excel
workbook in `VCEM_4_0/input/`. Replaces the hand-coded
`run_disk_compression_2()` that used to live in this notebook.

## What this notebook does
1. Reads `input/<case>.xlsx` (sheets: BEM, GEOMETRY, CRACK_NETWORK,
   CONFIGURATION_LOGIC, CONFIGURATION_PARAM).
2. Builds typed parameter objects (`BrazilianDiskParams`,
   `CrackGrowthParams`, `PlotParams`, a boundary spec, a grid spec).
3. Runs the disk-compression workflow:
   - BEM baseline solve on the disk
   - One-way crack growth + outer-cycle coupling (iterative or direct)

## Regenerating the template
After defaults change in any dataclass in `fracture_utils.Ubem`, regenerate
the starter workbook with:

    python input/_build_template.py

This will overwrite `input/disk_compression_2.xlsx` -- back up first if
you have hand edits to preserve.

## Grid mesh options (`GEOMETRY.grid_source`)
- `builtin_rect`   : rectangular grid built from `n_grid` + `pad_frac`
                     (this is what `ensure_bem_field` uses for the saved
                     `xs.npy` / `ys.npy` arrays).
- `builtin_polar`  : polar grid (r x theta) for out-of-band stress sampling.
- `external_file`  : load `(N, 2)` `(x, y)` nodes from a `.csv` or `.npy`
                     file (point `grid_external_file` to it).

Only `builtin_rect` is consumed by the BEM contour arrays in the current
disk pipeline. The other two are exposed via
`excel_io.build_grid_points(cfg)` for downstream / postprocessing use.
"""

CELL1_MD = "## 1. Pick the case workbook"

CELL2_CODE = """\
# Point this at the .xlsx that describes your run. Relative paths are
# resolved against the repo root (= the VCEM_4_0/ directory).
CASE_XLSX = "input/disk_compression_2.xlsx"
"""

CELL3_MD = """\
## 2. Preamble, Excel load, engine setup

Loads the workbook, captures `ENGINE` as a notebook global so the
monkey-patches below pick it up, and reloads / re-patches every
solver-side module that has an `engine='cpp' | 'python'` hook.
"""

CELL4_CODE = '''\
from __future__ import annotations

from pathlib import Path
import os, sys
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

nb_dir = Path(os.getcwd()).resolve()
repo_root = nb_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Hoist preamble.py at module scope so the run function inherits its names.
from preamble import *  # noqa: F401,F403

# --- Load case configuration from Excel ------------------------------
from input.excel_io import load_case, build_grid_points

_xlsx_path = (repo_root / CASE_XLSX).resolve()
cfg = load_case(_xlsx_path)

CASE    = Path(CASE_XLSX).stem
ENGINE  = cfg.engine          # consumed by the monkey-patch block below
DRY_RUN = cfg.dry_run

print(f'Workbook:  {_xlsx_path}')
print(f'Case:      {CASE}')
print(f'Engine:    {ENGINE}')
print(f'Coupling:  {cfg.coupling_method}  |  outer_cycles={cfg.outer_cycles}')
print(f'Network:   {cfg.vertices.shape[0]} vertices, {cfg.connectivity.shape[0]} edges')
print(f'Grid src:  {cfg.grid.source}')

# --- Backend dispatch (re-runnable: no Kernel-Restart needed) --------
import sys as _sys, importlib as _il

# Add VCEM_4_0/cpp/python so `import bem_cpp` resolves.
_cpp_pkg_dir = str(repo_root / 'cpp' / 'python')
if _cpp_pkg_dir not in _sys.path:
    _sys.path.insert(0, _cpp_pkg_dir)

# Drop any half-baked bem_cpp import from a previous run.
for _stale in [m for m in list(_sys.modules) if m == 'bem_cpp' or m.startswith('bem_cpp.')]:
    del _sys.modules[_stale]
_CPP_AVAILABLE = False
_bem_cpp_err = None
try:
    import bem_cpp  # noqa: F401
    _CPP_AVAILABLE = True
except Exception as _e:
    _bem_cpp_err = _e
    if ENGINE == 'cpp':
        ENGINE = 'python'

# Reload patched-module leaves before their consumers.
import fracture_utils.Usolver.build_discretize    as _bdisc_mod
import fracture_utils.Usolver.constraints         as _con_mod
import fracture_utils.Usolver.KKT                 as _kkt_mod
import fracture_utils.Usolver.cpp_dispatch        as _disp_mod
import fracture_utils.Usolver.parametrization     as _para_mod
import fracture_utils.Ugenerator.generator        as _gen_mod
import fracture_utils.Ugenerator.crack_network_simplifier as _simp_mod
import fracture_utils.Ubem.brazilian_disk_bem     as _bdb_mod
import fracture_utils.Ubem.disk_iterative_coupling as _dic_mod
import fracture_utils.Ubem.disk_network_propagation as _dnp_mod
import fracture_utils.Upropagation.direction       as _dir_mod
import fracture_utils.Upropagation.tip_state       as _tip_mod
import fracture_utils.Upropagation.geometry_update as _gu_mod
import fracture_utils.Upropagation.step_control    as _sc_mod
import fracture_utils.Upropagation.toughness       as _tough_mod
import fracture_utils.Upropagation.evaluate        as _eval_mod
import fracture_utils.Upropagation.network_growth  as _ng_mod
import fracture_utils.Upropagation.propagator      as _prop_mod
for _mod in (_bdisc_mod, _con_mod, _kkt_mod, _disp_mod, _para_mod,
             _simp_mod, _gen_mod,
             _dir_mod, _tip_mod, _gu_mod, _sc_mod, _tough_mod,
             _eval_mod, _ng_mod, _prop_mod,
             _bdb_mod, _dic_mod, _dnp_mod):
    _il.reload(_mod)
# Re-bind after reload.
import fracture_utils.Usolver.parametrization     as _para_mod
import fracture_utils.Ugenerator.generator        as _gen_mod
import fracture_utils.Ubem.brazilian_disk_bem     as _bdb_mod
import fracture_utils.Ubem.disk_iterative_coupling as _dic_mod

_orig_solve = _para_mod.DCENetworkStaticV4.solve
def _solve_with_engine(self, *args, **kwargs):
    kwargs.setdefault('engine', ENGINE)
    return _orig_solve(self, *args, **kwargs)
_para_mod.DCENetworkStaticV4.solve = _solve_with_engine

_orig_det = _gen_mod.CrackNetworkGenerator.detect_and_split_intersections
def _det_with_engine(self, *args, **kwargs):
    kwargs.setdefault('engine', ENGINE)
    return _orig_det(self, *args, **kwargs)
_gen_mod.CrackNetworkGenerator.detect_and_split_intersections = _det_with_engine

_orig_ensure  = _bdb_mod.ensure_bem_field
_orig_compute = _bdb_mod.compute_bem_brazilian_disk_field
def _ensure_with_engine(*args, **kwargs):
    kwargs.setdefault('engine', ENGINE)
    return _orig_ensure(*args, **kwargs)
def _compute_with_engine(*args, **kwargs):
    kwargs.setdefault('engine', ENGINE)
    return _orig_compute(*args, **kwargs)
_bdb_mod.ensure_bem_field = _ensure_with_engine
_bdb_mod.compute_bem_brazilian_disk_field = _compute_with_engine

_orig_sbwet = _dic_mod.solve_bem_with_extra_boundary_tractions
def _sbwet_with_engine(*args, **kwargs):
    kwargs.setdefault('engine', ENGINE)
    return _orig_sbwet(*args, **kwargs)
_dic_mod.solve_bem_with_extra_boundary_tractions = _sbwet_with_engine

# Banner so the user can't miss which backend is actually live.
print()
print('=' * 62)
print(f' ACTIVE ENGINE = {ENGINE.upper():<6s} '
      + ('(C++/OpenMP, %d threads)' % bem_cpp.openmp_max_threads() if _CPP_AVAILABLE and ENGINE == 'cpp'
         else '(pure Python)'))
if ENGINE == 'cpp':
    print(' Patched: DCENetworkStaticV4.solve, detect_and_split_intersections,')
    print('          ensure_bem_field, compute_bem_brazilian_disk_field,')
    print('          solve_bem_with_extra_boundary_tractions')
    print(f' bem_cpp v{bem_cpp.__version__} from {bem_cpp.__file__}')
else:
    if _bem_cpp_err is not None:
        print(f' Note: bem_cpp import failed: {_bem_cpp_err!r}')
        print(f'       (looked in {_cpp_pkg_dir})')
print('=' * 62)
'''

CELL5_MD = """\
## 3. Run

`run_disk_compression_2(cfg)` is the Excel-driven version of the
per-case workflow. Outer-cycle coupling, growth params, augmented
payload, and plotting are all sourced from `cfg`.

Set `dry_run = True` in the CONFIGURATION sheet of the workbook to
define the function without invoking it.
"""

CELL6_CODE = '''\
from dataclasses import replace as _dc_replace

from fracture_utils.Ubem.brazilian_disk_bem import (
    ensure_bem_field, compute_bem_brazilian_disk_field,
)
from fracture_utils.Ubem.disk_network_propagation import (
    CrackGrowthParams, run_network_growth_uncoupled,
)
from input.excel_io import save_supplementary_grid
from fracture_utils.Ubem.disk_crack_plotting import (
    make_standard_plot_hook, plot_total_field,
)
from fracture_utils.Ubem.coupling_blocks_v2 import (
    BEMBlockV2, compose_solver_kwargs_v2,
)
from fracture_utils.Ubem.disk_iterative_coupling import (
    compute_crack_boundary_tractions_direct,
    solve_bem_with_extra_boundary_tractions,
)
from fracture_utils.Ubem.boundary_conditions import build_boundary


def _network_to_arrays(net):
    V = np.array([[int(v.id), float(v.x), float(v.y)] for v in net.vertices], float)
    if len(net.edges):
        C = np.array([[int(e.v0), int(e.v1)] for e in net.edges], int)
    else:
        C = np.zeros((0, 2), int)
    return V, C


def _save_total_as_bem_arrays(bem_dir, Sxx_tot, Syy_tot, Sxy_tot):
    xs = np.load(bem_dir / "xs.npy")
    ys = np.load(bem_dir / "ys.npy")
    np.save(bem_dir / "xs.npy", xs)
    np.save(bem_dir / "ys.npy", ys)
    np.save(bem_dir / "Sxx.npy", np.asarray(Sxx_tot, float))
    np.save(bem_dir / "Syy.npy", np.asarray(Syy_tot, float))
    np.save(bem_dir / "Sxy.npy", np.asarray(Sxy_tot, float))


def run_disk_compression_2(cfg):
    """Brazilian-disk + crack-network coupling, Excel-driven."""
    from datetime import datetime as _datetime
    _ts = _datetime.now().strftime("%Y%m%d_%H%M%S")
    bem_dir = repo_root / "output" / f"{_ts}_{cfg.output_dir_name}"
    bem_dir.mkdir(parents=True, exist_ok=True)

    # (1) BEM baseline -- engine kwarg is filled in by the monkey-patch.
    # If a non-rect supplementary grid was requested, run the BEM solve
    # via the lower-level entry that returns the solver, so we can sample
    # the supplementary grid without a second BEM solve.
    if cfg.grid.source == "builtin_rect":
        ensure_bem_field(
            cfg.disk,
            bem_dir,
            recompute=(not cfg.skip_bem_solve),
            show=True,
            save_contours=True,
        )
    else:
        print(f"[grid] cfg.grid.source = {cfg.grid.source!r}: "
              "rect arrays remain the primary grid; supplementary polar/external "
              "arrays will be saved alongside.")
        solver, _field = compute_bem_brazilian_disk_field(
            cfg.disk, bem_dir,
            show=True, save_arrays=True, save_contours=True,
        )
        save_supplementary_grid(cfg, bem_dir, solver=solver, show=True)

    # (2) outer boundary mesh (used by iterative coupling step)
    boundary_mesh = build_boundary(cfg.boundary_spec)

    # (3) plot hook
    hook = make_standard_plot_hook(
        bem_dir=bem_dir,
        combined_out_dir=bem_dir,
        plot_params=cfg.plot,
        show_initial=cfg.plot_show_initial,
        show_cycles=cfg.plot_show_cycles,
        show_final=cfg.plot_show_final,
    )

    # (4) solver kwargs (one-way + direct/full-KKT)
    # cfg.parametrization (Excel: solver_kwargs.parametrization) controls
    # whether each polyline is fit as a cubic spline ('cspline') or kept as
    # piecewise-linear panels ('polyline') at solve time.
    one_way_kwargs = compose_solver_kwargs_v2(
        "one_way",
        base_solver_kwargs={
            "n_crack_elements": cfg.n_crack_elements,
            "parametrization": cfg.parametrization,
        },
    )
    bem_block = BEMBlockV2(disk=cfg.disk, out_dir=bem_dir)
    aug_payload = bem_block.build_augmented_payload(
        d_mode=cfg.aug_d_mode, gauss_n=cfg.aug_gauss_n,
    )
    direct_kwargs = compose_solver_kwargs_v2(
        "full_kkt",
        base_solver_kwargs={
            "n_crack_elements": cfg.n_crack_elements,
            "parametrization": cfg.parametrization,
            "augmented_enforce_crack_equilibrium": cfg.aug_enforce_crack_equilibrium,
            "augmented_equilibrate": cfg.aug_equilibrate,
            "augmented_equilibrate_cols": cfg.aug_equilibrate_cols,
            "augmented_physical_scaling": cfg.aug_physical_scaling,
            "augmented_scale_traction_rows": cfg.aug_scale_traction_rows,
        },
        augmented_payload=aug_payload,
        augmented_ridge_q=cfg.aug_ridge_q,
        augmented_ridge_y=cfg.aug_ridge_y,
        augmented_keep_bem_applied=cfg.aug_keep_bem_applied,
    )

    # (5) outer cycle loop
    V_curr = cfg.vertices.copy()
    C_curr = cfg.connectivity.copy()
    # Pin the original initial-crack vertex ids so the kink clamp's
    # "first emission" bypass only fires for tips on the truly original
    # crack -- not for tips grown in previous outer cycles whose ids end
    # up in net.vertices when run_network_growth_uncoupled is re-entered.
    original_initial_vids = set(int(v) for v in cfg.vertices[:, 0])
    res_last = None

    for cyc in range(1, int(cfg.outer_cycles) + 1):
        print(f"\\n[outer-cycle] {cyc}/{cfg.outer_cycles} method={cfg.coupling_method}")

        grow = _dc_replace(cfg.crack_growth, solver_kwargs=one_way_kwargs)
        out_growth = bem_dir / f"outer_{cyc:02d}_growth"
        out_growth.mkdir(parents=True, exist_ok=True)

        res_growth = run_network_growth_uncoupled(
            bem_dir=bem_dir,
            out_dir=out_growth,
            vertices=V_curr,
            connectivity=C_curr,
            params=grow,
            plot_hook=hook,
            original_initial_vertex_ids=original_initial_vids,
            plot_params=cfg.plot,
        )

        if cfg.coupling_method == "iterative":
            tx_cr, ty_cr = compute_crack_boundary_tractions_direct(
                res=res_growth, boundary_mesh=boundary_mesh,
            )
            solve_bem_with_extra_boundary_tractions(
                disk_params=cfg.disk,
                bem_dir=bem_dir,
                tx_extra=-tx_cr,
                ty_extra=-ty_cr,
                show=False,
            )
            res_last = res_growth

        elif cfg.coupling_method == "direct":
            # max_cycles=0 -> solve direct KKT on current network, no growth
            grow_direct = _dc_replace(
                cfg.crack_growth, max_cycles=0, solver_kwargs=direct_kwargs,
            )
            out_direct = bem_dir / f"outer_{cyc:02d}_direct"
            out_direct.mkdir(parents=True, exist_ok=True)

            Vg, Cg = _network_to_arrays(res_growth.calc.network)
            res_direct = run_network_growth_uncoupled(
                bem_dir=bem_dir,
                out_dir=out_direct,
                vertices=Vg,
                connectivity=Cg,
                params=grow_direct,
                plot_hook=None,
                original_initial_vertex_ids=original_initial_vids,
                plot_params=cfg.plot,
            )
            Sxx_tot, Syy_tot, Sxy_tot = plot_total_field(
                bem_dir=bem_dir,
                res=res_direct,
                out_dir=out_direct,
                tag=f"outer_{cyc:02d}_direct",
                params=cfg.plot,
                show=True,
            )
            _save_total_as_bem_arrays(bem_dir, Sxx_tot, Syy_tot, Sxy_tot)
            res_last = res_direct

        elif cfg.coupling_method == "no_coupling":
            res_last = res_growth

        else:
            raise ValueError(
                f"Unknown coupling_method: {cfg.coupling_method!r}. "
                f"Expected one of: no_coupling, iterative, direct."
            )

        # Advance network for next outer cycle.
        V_curr, C_curr = _network_to_arrays(res_growth.calc.network)

    print("\\n[done] outer-cycle run completed")

    # (6) End-of-run videos: walk outer_*_growth/cycle_*/ and stitch the
    # four per-cycle PNG sequences (deformed_network + 3 contours) into
    # <bem_dir>/videos/*.mp4. Failures are non-fatal: the run is already
    # done; we just log and continue.
    try:
        import sys as _sys
        _tools_dir = str(repo_root / "tools")
        if _tools_dir not in _sys.path:
            _sys.path.insert(0, _tools_dir)
        from make_run_videos import make_videos as _make_videos
        _make_videos(bem_dir, fps=5)
    except Exception as _e:
        print(f"[videos] skipped (error: {_e})")

    return {"bem_dir": bem_dir, "res_last": res_last,
            "vertices": V_curr, "connectivity": C_curr}


# --- dispatch -------------------------------------------------------
if DRY_RUN:
    print(f"DRY RUN -- run_disk_compression_2 defined but not executed.")
    print(f"Set dry_run = False in the CONFIGURATION sheet to invoke it.")
    result = None
else:
    result = run_disk_compression_2(cfg)
    print(f"Finished -- result keys: {list(result)}")
'''


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def main():
    cells = [
        md_cell(CELL0_MD),
        md_cell(CELL1_MD),
        code_cell(CELL2_CODE),
        md_cell(CELL3_MD),
        code_cell(CELL4_CODE),
        md_cell(CELL5_MD),
        code_cell(CELL6_CODE),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {NB_PATH}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
