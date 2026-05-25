"""Compare C++ edge_dislocation_u / edge_dislocation_stress against the
Python reference in fracture_utils/Usolver/solve_kernels.py at a panel of
representative (dx, dy, dBx, dBy) tuples.

Both implementations follow the same formulas line-for-line; expected
max|err| = 0 (IEEE-754 deterministic for identical op sequence).
"""

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))
sys.path.insert(0, str(REPO / "VCEM_4_0" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Usolver import solve_kernels as py_kern

E, NU, MU = 70e9, 0.3, 70e9 / (2.0 * (1.0 + 0.3))

# ── displacement kernel ────────────────────────────────────────────────────
def check_u(dx, dy, dBx, dBy, plane_stress, name):
    ux_p, uy_p = py_kern.edge_dislocation_u(np.array(dx), np.array(dy),
                                             dBx, dBy, NU,
                                             plane_stress=plane_stress)
    ux_p = float(ux_p); uy_p = float(uy_p)
    ux_c, uy_c = bem_cpp.edge_dislocation_u(dx, dy, dBx, dBy, NU, plane_stress)
    err = max(abs(ux_p - ux_c), abs(uy_p - uy_c))
    assert err < 1e-14, (f"u {name}: err={err:.3e}\n"
                          f"  PY=({ux_p:.6e}, {uy_p:.6e})\n  CPP=({ux_c:.6e}, {uy_c:.6e})")
    print(f"[OK] u  {name:30s}  max|err| = {err:.2e}")

check_u(1.0, 0.0, 1.0, 0.0, False, "horizontal bx, plane_strain")
check_u(0.5, 0.866, 1.0, 0.0, False, "60-deg bx")
check_u(1.0, 2.0, 0.0, 1.0, False, "off-origin by")
check_u(1e-3, 0.0, 1.0, 0.0, False, "near-singular")
check_u(0.7, -0.3, 0.4, -0.6, True,  "oblique mixed bx+by, plane_stress")

# ── stress kernel ──────────────────────────────────────────────────────────
def check_s(dx, dy, dBx, dBy, plane_stress, name):
    sxx_p, syy_p, sxy_p = py_kern.stress_edge_dislocation(
        np.array(dx), np.array(dy), dBx, dBy, MU, NU, plane_stress=plane_stress)
    sxx_p = float(sxx_p); syy_p = float(syy_p); sxy_p = float(sxy_p)
    sxx_c, syy_c, sxy_c = bem_cpp.edge_dislocation_stress(
        dx, dy, dBx, dBy, MU, NU, plane_stress)
    err = max(abs(sxx_p - sxx_c), abs(syy_p - syy_c), abs(sxy_p - sxy_c))
    scale = max(abs(sxx_p), abs(syy_p), abs(sxy_p), 1e-30)
    assert err / scale < 1e-12, (f"s {name}: rel err={err/scale:.3e}\n"
                                  f"  PY=({sxx_p:.6e}, {syy_p:.6e}, {sxy_p:.6e})\n"
                                  f"  CPP=({sxx_c:.6e}, {syy_c:.6e}, {sxy_c:.6e})")
    print(f"[OK] s  {name:30s}  max|err| = {err:.2e}  rel = {err/scale:.2e}")

check_s(1.0, 0.0, 1.0, 0.0, False, "horizontal bx, plane_strain")
check_s(0.5, 0.866, 1.0, 0.0, False, "60-deg bx")
check_s(1.0, 2.0, 0.0, 1.0, False, "off-origin by")
check_s(1.0, 1.0, 0.5, -0.3, False, "mixed bx+by")
check_s(0.7, -0.3, 0.4, -0.6, True,  "plane_stress")
check_s(1e-3, 0.0, 1.0, 0.0, False, "near-singular")

print("\nALL EDGE-DISLOCATION KERNEL TESTS PASSED")
