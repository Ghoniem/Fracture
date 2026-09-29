"""Compare C++ assemble_operator vs Python on a small straight-crack
benchmark. Same fixture as test_panel_merge_guard.py to keep the build
path self-contained.

Both implementations should produce K, rhs that match to ~1e-12 relative
(arithmetic order is identical; the only deviations come from OpenMP
reduction order in the per-row inner loops).
"""

import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.build_discretize import (
    discretize_polylines, allocate_unknowns, assemble_operator as py_assemble,
)
from fracture_utils.Usolver.build_dipolar_polyline import build_polylines_full
from fracture_utils.Usolver.material import Material, AppliedStress


def make_problem(L=1.0e-3, n_crack_elements=15):
    net = CrackNetworkV4(
        vertices=[VertexV4(id=0, x=-L/2, y=0.0), VertexV4(id=1, x=L/2, y=0.0)],
        edges=[EdgeV4(id=0, v0=0, v1=1)],
    )
    polylines, _ = build_polylines_full(net)
    panels = discretize_polylines(
        net, polylines,
        n_crack_elements=n_crack_elements,
        node_distribution="uniform",
        collocation_mode="midpoint",
        representation="regular",
        nq_stress=4,
        tip_cluster="power", tip_cluster_power=2.0,
        endpoint_min_nodes=4,
        refine_junction_endpoints=True,
        refine_kinks=False,
        min_panel_length_ratio=1e-3,
    )
    deg = {0: 1, 1: 1}
    offsets, _jdof, _bdof, nunk = allocate_unknowns(
        panels, deg, crack_mode="full", junction_model="strict")

    material = Material(E=70e9, nu=0.3, plane_stress=False)
    applied  = AppliedStress(sigma_xx=0.0, sigma_yy=-1.0e6, sigma_xy=0.0)
    return panels, offsets, nunk, material, applied


def main():
    panels, offsets, nunk, material, applied = make_problem(
        L=1.0e-3, n_crack_elements=15)
    print(f"problem: {len(panels)} polylines, total panels = "
          f"{sum(p['Np'] for p in panels)}, nunk = {nunk}, "
          f"ncol_tot = {sum(len(p['x_col']) for p in panels)}")

    # ── Python reference ────────────────────────────────────────────────
    t0 = time.perf_counter()
    K_py, rhs_py = py_assemble(
        material=material, applied=applied,
        poly_panels=panels, offsets=list(offsets), nunk=nunk, nq_col=4)
    t_py = time.perf_counter() - t0
    print(f"  python assemble_operator: {t_py*1000:8.2f} ms")

    # ── Precompute sigma_at_col for the C++ call ────────────────────────
    cols = np.vstack([p["x_col"] for p in panels])      # (ncol_tot, 2)
    sig_full = applied.tensor_at(cols)                  # (ncol_tot, 2, 2)
    sigma_at_col = np.column_stack([
        sig_full[:, 0, 0],
        sig_full[:, 1, 1],
        sig_full[:, 0, 1],
    ])                                                   # (ncol_tot, 3)

    # ── C++ implementation ──────────────────────────────────────────────
    t0 = time.perf_counter()
    K_cpp, rhs_cpp = bem_cpp.assemble_operator(
        list(panels), list(offsets), int(nunk),
        material.E, material.nu, bool(material.plane_stress),
        sigma_at_col)
    t_cpp = time.perf_counter() - t0
    print(f"  cpp    assemble_operator: {t_cpp*1000:8.2f} ms  "
          f"(speedup = {t_py/t_cpp:.1f}x)")

    # ── Numeric agreement ───────────────────────────────────────────────
    Kscale   = float(np.max(np.abs(K_py)))
    rhsscale = float(np.max(np.abs(rhs_py)))
    K_err    = float(np.max(np.abs(K_py - K_cpp)))
    rhs_err  = float(np.max(np.abs(rhs_py - rhs_cpp)))
    print(f"  K   scale={Kscale:.3e}  max|err|={K_err:.3e}  "
          f"rel={K_err/max(Kscale,1e-30):.3e}")
    print(f"  rhs scale={rhsscale:.3e}  max|err|={rhs_err:.3e}  "
          f"rel={rhs_err/max(rhsscale,1e-30):.3e}")

    assert K_err   / max(Kscale, 1e-30)   < 1e-10, "K mismatch"
    assert rhs_err / max(rhsscale, 1e-30) < 1e-12, "rhs mismatch"
    print("\nASSEMBLE_OPERATOR TEST PASSED")


if __name__ == "__main__":
    main()
