"""Minimal diagnostic: run ONE growth cycle of disk_compression_inclined
with engine='cpp', confirming each solve() call is actually using the
C++ path (not silently falling back to Python). Prints per-solve timing
to expose whether one specific solve is hanging vs all of them being
silently Python.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))


def main():
    # Confirm bem_cpp imports
    import bem_cpp
    print(f"bem_cpp: v{bem_cpp.__version__}  openmp_threads={bem_cpp.openmp_max_threads()}",
          flush=True)

    # cwd at notebooks so the case's repo_root resolves to VCEM
    os.chdir(REPO / "vcem" / "notebooks")

    # Install engine='cpp' patch with INSTRUMENTATION: every solve() prints
    # how long it took and whether it dispatched to C++ (the cpp dispatch
    # silently falls back to Python on missing bem_cpp -- we want to see
    # which branch fires).
    import fracture_utils.Usolver.parametrization as para_mod
    import fracture_utils.Usolver.cpp_dispatch as disp_mod
    orig_solve = para_mod.DCENetworkStaticV4.solve
    _call_id = {"n": 0}

    def _traced_solve(self, *args, **kwargs):
        kwargs.setdefault("engine", "cpp")
        _call_id["n"] += 1
        cid = _call_id["n"]
        n_unk_hint = kwargs.get("n_crack_elements", "?")
        print(f"  [solve #{cid:>3d}] enter  engine={kwargs.get('engine')!r} "
              f"n_crack_elements={n_unk_hint!r} kwargs_keys={list(kwargs)[:5]}",
              flush=True)
        t0 = time.perf_counter()
        try:
            out = orig_solve(self, *args, **kwargs)
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  [solve #{cid:>3d}] RAISED after {dt*1000:.1f} ms: {type(e).__name__}: {e}",
                  flush=True)
            raise
        dt = time.perf_counter() - t0
        ncol = "?"
        try:
            ncol = sum(len(p["x_col"]) for p in out["polyline_solutions"])
        except Exception:
            pass
        print(f"  [solve #{cid:>3d}] exit  in {dt*1000:>8.1f} ms  ncol_tot={ncol}", flush=True)
        return out

    para_mod.DCENetworkStaticV4.solve = _traced_solve

    # Build the inclined-crack network from disk_compression_inclined
    from fracture_utils.Ubem.brazilian_disk_bem import (
        BrazilianDiskParams, ensure_bem_field)
    from fracture_utils.Ubem.disk_network_propagation import (
        CrackGrowthParams, run_network_growth_uncoupled)

    bem_dir = REPO / "vcem" / "output" / "BEM_Brazilian_disk"
    bem_dir.mkdir(parents=True, exist_ok=True)

    mm = 1e-3
    vertices = np.array([
        [0, -2.0*mm, -3.0*mm],
        [1,  2.0*mm,  3.0*mm],
    ], float)
    connectivity = np.array([[0, 1]], int)

    disk = BrazilianDiskParams(R=25.4e-3/2, P_total=3.8e3, E=231.52e9, nu=0.3,
                                n_boundary_elements=60, n_grid=120)

    # ensure_bem_field with engine='cpp' (uses bem_cpp.BEMSolver2D)
    print("\nensure_bem_field (engine='cpp')...", flush=True)
    t0 = time.perf_counter()
    ensure_bem_field(disk, bem_dir, recompute=True, show=False,
                     save_contours=False, engine='cpp')
    print(f"  ensure_bem_field done in {time.perf_counter()-t0:.1f} s", flush=True)

    # ONE growth cycle, no plot hook
    print("\nrun_network_growth_uncoupled (max_cycles=1, engine=cpp)...", flush=True)
    grow = CrackGrowthParams(max_cycles=1, L_limit_mm=20.0, vertex_high=18)
    t0 = time.perf_counter()
    res = run_network_growth_uncoupled(
        bem_dir=bem_dir, out_dir=bem_dir,
        vertices=vertices, connectivity=connectivity,
        params=grow, plot_hook=None,
    )
    dt = time.perf_counter() - t0
    print(f"\nrun_network_growth_uncoupled done in {dt:.1f} s "
          f"({_call_id['n']} solve calls total)", flush=True)


if __name__ == "__main__":
    main()
