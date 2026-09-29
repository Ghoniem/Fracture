"""Drive the disk_compression_2 case from SimulationsCpp.ipynb directly,
once with engine='python' and once with engine='cpp', and report:

  - wall-clock for the run
  - final crack-network state (vertex + edge counts; bounding box)

This is the same dispatch the notebook does -- we exec the case function
that lives in Simulations.ipynb (the body is unchanged across notebooks)
after applying the same module-level monkey-patches that the cpp
notebook installs in its preamble cell.
"""

from __future__ import annotations
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))
sys.path.insert(0, str(REPO / "vcem" / "notebooks"))  # for `preamble`


def _install_engine_patches(engine: str):
    """Same monkey-patches the cpp notebook installs in its preamble."""
    import fracture_utils.Usolver.parametrization as para_mod
    import fracture_utils.Ugenerator.generator as gen_mod

    # Reinstall every call so the captured `engine` string is fresh.
    # If we already patched once, restore the originals first to avoid stacking.
    if getattr(para_mod.DCENetworkStaticV4, "_engine_patched", False):
        para_mod.DCENetworkStaticV4.solve = para_mod.DCENetworkStaticV4._orig_solve
        del para_mod.DCENetworkStaticV4._orig_solve
        del para_mod.DCENetworkStaticV4._engine_patched
    if getattr(gen_mod.CrackNetworkGenerator, "_engine_patched", False):
        gen_mod.CrackNetworkGenerator.detect_and_split_intersections = (
            gen_mod.CrackNetworkGenerator._orig_det)
        del gen_mod.CrackNetworkGenerator._orig_det
        del gen_mod.CrackNetworkGenerator._engine_patched

    orig_solve = para_mod.DCENetworkStaticV4.solve

    def _solve_with_engine(self, *args, **kwargs):
        kwargs.setdefault("engine", engine)
        return orig_solve(self, *args, **kwargs)

    para_mod.DCENetworkStaticV4._orig_solve = orig_solve
    para_mod.DCENetworkStaticV4.solve = _solve_with_engine
    para_mod.DCENetworkStaticV4._engine_patched = True

    orig_det = gen_mod.CrackNetworkGenerator.detect_and_split_intersections

    def _det_with_engine(self, *args, **kwargs):
        kwargs.setdefault("engine", engine)
        return orig_det(self, *args, **kwargs)

    gen_mod.CrackNetworkGenerator._orig_det = orig_det
    gen_mod.CrackNetworkGenerator.detect_and_split_intersections = _det_with_engine
    gen_mod.CrackNetworkGenerator._engine_patched = True


def _extract_case_source(case_name: str) -> str:
    """Pull the body of run_<case>() out of Simulations.ipynb verbatim."""
    import json
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
    raise RuntimeError(f"Case {case_name!r} not found in notebook.")


def run_case(case_name: str, engine: str):
    _install_engine_patches(engine)
    src = _extract_case_source(case_name)
    # Run the case in a fresh namespace seeded with the same wildcard imports
    # the notebook's preamble cell injects.
    ns: dict = {}
    exec("from preamble import *", ns)
    exec(src, ns)
    fn = ns[f"run_{case_name}"]

    t0 = time.perf_counter()
    # Mute stdout to keep our timing output readable. Re-enable if debugging.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    t = time.perf_counter() - t0
    return t, result, buf.getvalue()


def summarize_network(res):
    """Pull a few stable descriptors of the final crack network for comparison."""
    if not isinstance(res, dict):
        return None
    res_last = res.get("res_last")
    if res_last is None:
        return None
    net = getattr(getattr(res_last, "calc", None), "network", None)
    if net is None:
        return None
    V = np.array([[v.x, v.y] for v in net.vertices], dtype=float)
    nv = len(net.vertices)
    ne = len(net.edges)
    bbox = (float(V[:, 0].min()), float(V[:, 0].max()),
            float(V[:, 1].min()), float(V[:, 1].max()))
    return {"nv": nv, "ne": ne, "bbox": bbox, "V": V}


def compare_networks(a, b):
    if a is None or b is None:
        return None
    if a["nv"] != b["nv"] or a["ne"] != b["ne"]:
        return f"vertex/edge counts differ: {a['nv']}/{a['ne']} vs {b['nv']}/{b['ne']}"
    # Hausdorff-style metric on vertex positions
    if a["V"].shape == b["V"].shape:
        diff = np.max(np.abs(a["V"] - b["V"]))
        return f"max vertex coord diff = {diff:.3e}"
    return "shape mismatch"


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "disk_compression_2"
    print(f"\nCase: {case}\n")

    print("--- engine='python' (baseline) ---")
    t_py, res_py, log_py = run_case(case, "python")
    s_py = summarize_network(res_py)
    print(f"  wall-clock: {t_py:8.2f} s")
    if s_py:
        print(f"  final network: {s_py['nv']} vertices, {s_py['ne']} edges, "
              f"bbox = {s_py['bbox']}")

    print("\n--- engine='cpp' ---")
    t_cpp, res_cpp, log_cpp = run_case(case, "cpp")
    s_cpp = summarize_network(res_cpp)
    print(f"  wall-clock: {t_cpp:8.2f} s   (speedup = {t_py/max(t_cpp,1e-9):.2f}x)")
    if s_cpp:
        print(f"  final network: {s_cpp['nv']} vertices, {s_cpp['ne']} edges, "
              f"bbox = {s_cpp['bbox']}")

    print("\n--- comparison ---")
    print(f"  {compare_networks(s_py, s_cpp)}")


if __name__ == "__main__":
    main()
