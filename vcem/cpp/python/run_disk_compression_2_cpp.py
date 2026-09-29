"""Run the disk_compression_2 case with engine='cpp' end-to-end.

Drives the same code path SimulationsCpp.ipynb does (extracts the case
function body from Simulations.ipynb verbatim, applies the same engine
monkey-patches, executes). Sets cwd to vcem/notebooks/ so the
case's `nb_dir = Path(os.getcwd())` resolves correctly and outputs
land under vcem/output/BEM_Brazilian_disk_iterative/.

Case parameters (unchanged from disk_compression_2 in the notebook):
  n_boundary_elements = 120, n_grid = 120
  n_crack_elements    = 40
  OUTER_CYCLES        = 2     (iterative BEM coupling)
  max_cycles          = 5     (network growth per outer cycle)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))
sys.path.insert(0, str(REPO / "vcem" / "notebooks"))   # for `preamble`


def _install_cpp_patches():
    import fracture_utils.Usolver.parametrization as para_mod
    import fracture_utils.Ugenerator.generator as gen_mod

    orig_solve = para_mod.DCENetworkStaticV4.solve

    def _solve_with_engine(self, *args, **kwargs):
        kwargs.setdefault("engine", "cpp")
        return orig_solve(self, *args, **kwargs)

    para_mod.DCENetworkStaticV4._orig_solve = orig_solve
    para_mod.DCENetworkStaticV4.solve = _solve_with_engine

    orig_det = gen_mod.CrackNetworkGenerator.detect_and_split_intersections

    def _det_with_engine(self, *args, **kwargs):
        kwargs.setdefault("engine", "cpp")
        return orig_det(self, *args, **kwargs)

    gen_mod.CrackNetworkGenerator._orig_det = orig_det
    gen_mod.CrackNetworkGenerator.detect_and_split_intersections = _det_with_engine


def _extract_case_source(case_name: str) -> str:
    nb_path = REPO / "vcem" / "notebooks" / "Simulations.ipynb"
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    target = f"def run_{case_name}"
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.startswith(target):
            return src
    raise RuntimeError(f"case {case_name!r} not found")


def main():
    case = "disk_compression_2"

    # cwd = notebooks/ so the case's repo_root resolves to vcem/
    notebooks_dir = REPO / "vcem" / "notebooks"
    os.chdir(notebooks_dir)
    print(f"cwd: {os.getcwd()}")

    # Quick sanity check on the C++ extension
    import bem_cpp
    print(f"bem_cpp: __version__={bem_cpp.__version__}  "
          f"openmp_max_threads={bem_cpp.openmp_max_threads()}")

    _install_cpp_patches()
    print("engine='cpp' patches installed on DCENetworkStaticV4.solve "
          "and CrackNetworkGenerator.detect_and_split_intersections\n")

    src = _extract_case_source(case)
    ns: dict = {}
    exec("from preamble import *", ns)
    exec(src, ns)
    fn = ns[f"run_{case}"]

    print(f"=== running run_{case}() with engine='cpp' ===")
    t0 = time.perf_counter()
    # Let the case's own prints flow through so progress is visible.
    result = fn()
    t = time.perf_counter() - t0

    print()
    print(f"=== finished in {t:.2f} s ({t/60:.2f} min) ===")

    # Quick descriptor of the final network state
    res_last = result.get("res_last") if isinstance(result, dict) else None
    if res_last is not None:
        net = getattr(getattr(res_last, "calc", None), "network", None)
        if net is not None:
            print(f"final network: {len(net.vertices)} vertices, "
                  f"{len(net.edges)} edges")

    bem_dir = REPO / "vcem" / "output" / "BEM_Brazilian_disk_iterative"
    if bem_dir.exists():
        files = sorted(bem_dir.iterdir())
        print(f"output dir: {bem_dir}")
        print(f"  files: {len(files)}")
        for sub in sorted(p for p in bem_dir.iterdir() if p.is_dir()):
            n = len(list(sub.iterdir()))
            print(f"    {sub.name}/  ({n} files)")


if __name__ == "__main__":
    main()
