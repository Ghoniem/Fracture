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

# ── Backend dispatch: inject ENGINE into every KKT solve and intersection
# ── detector call without rewriting the case functions. Module-level
# ── monkey-patches set kwargs.setdefault('engine', ENGINE), so case code
# ── that already passes its own engine value still wins.
print(f'Engine backend: {ENGINE!r}')

# The notebook's existing sys.path only has repo_root (= VCEM_4_0); the
# C++ extension lives under VCEM_4_0/cpp/python/. Add it here so engine='cpp'
# actually runs C++ instead of silently falling back to Python.
import sys as _sys
_cpp_pkg_dir = str(repo_root / 'cpp' / 'python')
if _cpp_pkg_dir not in _sys.path:
    _sys.path.insert(0, _cpp_pkg_dir)

try:
    import bem_cpp  # noqa: F401
    _CPP_AVAILABLE = True
except Exception as _e:
    _CPP_AVAILABLE = False
    if ENGINE == 'cpp':
        print(f'  WARNING: bem_cpp not importable ({_e!r}); falling back to python.')
        print(f'           (looked in {_cpp_pkg_dir}; build with VCEM_4_0/cpp/README.md)')
        ENGINE = 'python'

if ENGINE == 'cpp':
    print('  C++ OpenMP threads:', bem_cpp.openmp_max_threads())

import fracture_utils.Usolver.parametrization as _para_mod
if not getattr(_para_mod.DCENetworkStaticV4, '_engine_patched', False):
    _orig_solve = _para_mod.DCENetworkStaticV4.solve

    def _solve_with_engine(self, *args, **kwargs):
        kwargs.setdefault('engine', ENGINE)
        return _orig_solve(self, *args, **kwargs)

    _para_mod.DCENetworkStaticV4.solve = _solve_with_engine
    _para_mod.DCENetworkStaticV4._engine_patched = True

import fracture_utils.Ugenerator.generator as _gen_mod
if not getattr(_gen_mod.CrackNetworkGenerator, '_engine_patched', False):
    _orig_det = _gen_mod.CrackNetworkGenerator.detect_and_split_intersections

    def _det_with_engine(self, *args, **kwargs):
        kwargs.setdefault('engine', ENGINE)
        return _orig_det(self, *args, **kwargs)

    _gen_mod.CrackNetworkGenerator.detect_and_split_intersections = _det_with_engine
    _gen_mod.CrackNetworkGenerator._engine_patched = True

# BEM-side: route engine into ensure_bem_field (and compute_bem_brazilian_disk_field
# which it calls internally) and into solve_bem_with_extra_boundary_tractions.
# These swap BEMSolver2D for bem_cpp.BEMSolver2D under the hood.
import fracture_utils.Ubem.brazilian_disk_bem as _bdb_mod
if not getattr(_bdb_mod, '_engine_patched', False):
    _orig_ensure = _bdb_mod.ensure_bem_field
    _orig_compute = _bdb_mod.compute_bem_brazilian_disk_field

    def _ensure_with_engine(*args, **kwargs):
        kwargs.setdefault('engine', ENGINE)
        return _orig_ensure(*args, **kwargs)

    def _compute_with_engine(*args, **kwargs):
        kwargs.setdefault('engine', ENGINE)
        return _orig_compute(*args, **kwargs)

    _bdb_mod.ensure_bem_field = _ensure_with_engine
    _bdb_mod.compute_bem_brazilian_disk_field = _compute_with_engine
    _bdb_mod._engine_patched = True

import fracture_utils.Ubem.disk_iterative_coupling as _dic_mod
if not getattr(_dic_mod, '_engine_patched', False):
    _orig_sbwet = _dic_mod.solve_bem_with_extra_boundary_tractions

    def _sbwet_with_engine(*args, **kwargs):
        kwargs.setdefault('engine', ENGINE)
        return _orig_sbwet(*args, **kwargs)

    _dic_mod.solve_bem_with_extra_boundary_tractions = _sbwet_with_engine
    _dic_mod._engine_patched = True
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
