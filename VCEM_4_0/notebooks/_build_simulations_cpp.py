"""Build SimulationsCpp.ipynb by cloning Simulations.ipynb and inserting

  - a backend-selector cell ('python' vs 'cpp') near the top
  - a monkey-patch block in the preamble that injects engine=ENGINE
    into DCENetworkStaticV4.solve and into
    CrackNetworkGenerator.detect_and_split_intersections

Case function bodies are kept byte-for-byte identical. The patches affect
only the kwargs that reach those calls at runtime.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path


SRC = Path(__file__).parent / "Simulations.ipynb"
DST = Path(__file__).parent / "SimulationsCpp.ipynb"


SELECTOR_CELL = {
    "cell_type": "markdown",
    "id": "backend-selector-md",
    "metadata": {},
    "source": [
        "## 0. Backend selector (Python vs C++)\n",
        "\n",
        "Pick the computation backend used for the crack-side KKT solve and the\n",
        "topology intersection detector. The BEM-side (`ensure_bem_field`)\n",
        "currently runs Python in both configurations; only the parts that have\n",
        "been ported to C++/OpenMP via the `bem_cpp` extension are affected.\n",
        "\n",
        "| ENGINE  | What runs in C++                                                   |\n",
        "|---------|--------------------------------------------------------------------|\n",
        "| 'python'| Nothing -- pure-Python reference (same as Simulations.ipynb)       |\n",
        "| 'cpp'   | KKT crack-solve assemblers + KKT solve + segment-intersection detect |\n",
        "\n",
        "The two engines produce results that agree to machine precision\n",
        "(~1e-12 relative) on every case verified so far. The C++ engine\n",
        "gives 16-27x full-solve speedup on networks with 100-1000 panels\n",
        "(see VCEM_4_0/cpp/python/bench_kkt_scaling.py)."
    ],
}

ENGINE_CELL = {
    "cell_type": "code",
    "id": "backend-selector-code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Backend for the crack-side KKT solve and intersection detection.\n",
        "ENGINE = 'cpp'    # options: 'python' | 'cpp'\n",
    ],
}

# Inserted at the end of the preamble cell (cell index 4 in the source).
ENGINE_PATCHES = """

# ── Backend dispatch (re-runnable: no Kernel-Restart needed after edits) ──
# This block force-reloads every module we monkey-patch and re-installs the
# patches every time the cell runs. That way, after any code change to the
# patched modules you can just re-run the preamble cell instead of
# Restart-Kernel-and-Run-All.
import sys as _sys, importlib as _il

# Add VCEM_4_0/cpp/python to sys.path so `import bem_cpp` resolves. The
# original preamble only adds repo_root (= VCEM_4_0).
_cpp_pkg_dir = str(repo_root / 'cpp' / 'python')
if _cpp_pkg_dir not in _sys.path:
    _sys.path.insert(0, _cpp_pkg_dir)

# Try to import bem_cpp, fall back to engine='python' if missing.
# Drop any half-baked import from a previous run so reload() can't fail
# on a missing-submodule cache entry.
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

# Force-reload every patched module AND its non-trivial dependencies
# (build_discretize, constraints, KKT, cpp_dispatch) -- reloading
# parametrization alone is not enough because its `from .build_discretize
# import discretize_polylines` re-runs but gets the cached build_discretize
# from sys.modules. We have to reload the *leaves* first, in dependency
# order, before reloading parametrization that re-binds from them.
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
# Reload leaves before their consumers; parametrization last among Usolver.
for _mod in (_bdisc_mod, _con_mod, _kkt_mod, _disp_mod, _para_mod,
              _simp_mod, _gen_mod, _bdb_mod, _dic_mod, _dnp_mod):
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

# Loud banner so the user can't miss which backend is actually live.
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
        # Detailed diagnostic so the root cause is obvious.
        print()
        print(' --- diagnostic ---')
        print(f'   sys.executable = {_sys.executable}')
        print(f'   sys.version    = {_sys.version.split(chr(10))[0]}')
        _pkg = _cpp_pkg_dir + '/bem_cpp'
        import os as _os
        if _os.path.isdir(_pkg):
            _files = sorted(_os.listdir(_pkg))
            print(f'   {_pkg} contents:')
            for _f in _files:
                print(f'     {_f}')
            _pyd = [f for f in _files if f.endswith('.pyd') or f.endswith('.so')]
            if _pyd:
                # cp310-win_amd64 -> Python 3.10. Check vs running interp.
                _v = _sys.version_info
                _tag = f'cp{_v.major}{_v.minor}'
                _matching = [f for f in _pyd if _tag in f]
                if not _matching:
                    print(f'   *** ABI MISMATCH: extension is {_pyd[0]}, '
                          f'but running interpreter is {_tag} (Python '
                          f'{_v.major}.{_v.minor}). ***')
                    print('   *** This usually means Jupyter is using a '
                          'different Python than the conda vcem_4_0 env. ***')
                    print('   *** Pick the vcem_4_0 kernel in Jupyter '
                          '(Kernel -> Change Kernel) or rebuild the '
                          'extension for your running Python ***')
                    print('   *** (cd VCEM_4_0/cpp/build && cmake --build . '
                          '--config Release --target bem_cpp && cmake '
                          '--install . --config Release). ***')
        else:
            print(f'   *** directory missing: {_pkg}')
            print('   *** rebuild via VCEM_4_0/cpp/README.md ***')
print('=' * 62)
import sys as _sys2; _sys2.stdout.flush()
"""


def main():
    with SRC.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Patch title cell (cell 0)
    title_src = "".join(cells[0]["source"])
    title_src = title_src.replace(
        "# Simulations",
        "# Simulations (C++/Python switchable)",
    )
    # Add a one-line summary at the top.
    title_src = title_src.replace(
        "Production simulation runs:",
        "Variant of Simulations.ipynb with a Python/C++ backend selector "
        "(see section 0). Production simulation runs:",
    )
    cells[0]["source"] = [title_src]

    # Insert selector cells right after cell 0 (before "## 1. Case selector").
    new_cells = [cells[0], SELECTOR_CELL, ENGINE_CELL] + cells[1:]

    # Append engine-dispatch monkey-patches to the end of the preamble cell.
    # The preamble was cells[4] in the source; after our insertions it's now
    # at index 6 in new_cells (0:title, 1:selector_md, 2:selector_code,
    # 3:case_md, 4:case_code, 5:preamble_md, 6:preamble_code).
    preamble_idx = 6
    assert new_cells[preamble_idx]["cell_type"] == "code"
    src = "".join(new_cells[preamble_idx]["source"])
    assert "from preamble import *" in src, (
        "Preamble cell signature changed; rebuild script needs an update.")
    new_cells[preamble_idx]["source"] = [src + ENGINE_PATCHES]
    # Clear stale outputs / execution_count so the cell runs fresh.
    new_cells[preamble_idx]["outputs"] = []
    new_cells[preamble_idx]["execution_count"] = None

    nb["cells"] = new_cells

    # Clear all execution outputs / counts so the new notebook is reproducible.
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None

    with DST.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {DST}")
    print(f"  cells: {len(nb['cells'])} (was {len(cells)} in source; +2 for backend selector)")


if __name__ == "__main__":
    main()
