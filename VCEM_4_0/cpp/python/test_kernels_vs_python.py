"""Unit-test C++ Kelvin kernels and quadrature against the Python originals.

Run from the conda env `vcem_4_0`:
    python VCEM_4_0/cpp/python/test_kernels_vs_python.py

Compares C++ bem_cpp.* against fracture_utils.Ubem.bem_solver.* on a handful
of representative (field, source, normal) tuples. Passing tolerance: <1e-12
absolute (both implementations use double precision, identical formulas).
"""

import sys
from pathlib import Path

import numpy as np

# Make both implementations importable.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))            # for fracture_utils
sys.path.insert(0, str(REPO / "VCEM_4_0" / "cpp" / "python"))  # for bem_cpp

import bem_cpp
from fracture_utils.Ubem import bem_solver as py_bem

PS = True
E, nu = 70e9, 0.3

# ── 1. Gauss-Legendre nodes/weights vs numpy.polynomial.legendre.leggauss ──
# C++ side: there's no direct accessor for the rule, but we can probe it via
# add_element + a known-quadrature smoke test if needed. For now check via
# numpy's leggauss as the reference (which the Python side uses directly).
for n in (4, 8, 12, 16):
    xs, ws = np.polynomial.legendre.leggauss(n)
    # Sanity: nodes sum to ~0, weights sum to 2.
    assert abs(xs.sum()) < 1e-12, f"leggauss({n}) nodes don't sum to 0"
    assert abs(ws.sum() - 2.0) < 1e-12, f"leggauss({n}) weights don't sum to 2"
print("[OK] numpy.leggauss self-consistency (reference)")

# ── 2. kelvin_U at a generic (field, source) ───────────────────────────────
def check_U(xf, yf, xs_, ys_, name):
    U_py  = py_bem.kelvin_U((xf, yf), (xs_, ys_), E=E, nu=nu, plane_strain=PS)
    U_cpp = bem_cpp.kelvin_U(xf, yf, xs_, ys_, E, nu, PS)
    err = np.max(np.abs(U_py - U_cpp))
    assert err < 1e-14, f"kelvin_U {name}: max err = {err:.3e}\nPY=\n{U_py}\nCPP=\n{U_cpp}"
    print(f"[OK] kelvin_U {name:20s}  max|err| = {err:.2e}")

check_U(0.0, 0.0, 1.0, 0.0,   "horizontal")
check_U(0.0, 0.0, 0.5, 0.866, "60-deg")
check_U(1.0, 2.0, 3.0, 5.0,   "off-origin")
check_U(0.0, 0.0, 1e-3, 0.0,  "near-singular")

# ── 3. kelvin_T at a generic (field, source, normal) ──────────────────────
def check_T(xf, yf, xs_, ys_, nx, ny, name):
    T_py  = py_bem.kelvin_T((xf, yf), (xs_, ys_), (nx, ny),
                             E=E, nu=nu, plane_strain=PS)
    T_cpp = bem_cpp.kelvin_T(xf, yf, xs_, ys_, nx, ny, E, nu, PS)
    err = np.max(np.abs(T_py - T_cpp))
    assert err < 1e-14, f"kelvin_T {name}: max err = {err:.3e}\nPY=\n{T_py}\nCPP=\n{T_cpp}"
    print(f"[OK] kelvin_T {name:20s}  max|err| = {err:.2e}")

check_T(0.0, 0.0, 1.0, 0.0,  0.0, 1.0, "tangent-normal")
check_T(0.0, 0.0, 0.7, 0.3,  0.5, np.sqrt(1-0.25), "oblique-normal")
check_T(1.0, 2.0, 3.0, 5.0, -1.0, 0.0, "negative-x normal")

# ── 4. kelvin_dU_dfield ───────────────────────────────────────────────────
def check_dU(xf, yf, xs_, ys_, name):
    dUdx_py, dUdy_py = py_bem.kelvin_dU_dfield((xf, yf), (xs_, ys_),
                                               E=E, nu=nu, plane_strain=PS)
    dUdx_cpp, dUdy_cpp = bem_cpp.kelvin_dU_dfield(xf, yf, xs_, ys_, E, nu, PS)
    ex = max(np.max(np.abs(dUdx_py - dUdx_cpp)),
             np.max(np.abs(dUdy_py - dUdy_cpp)))
    assert ex < 1e-12, f"kelvin_dU {name}: max err = {ex:.3e}"
    print(f"[OK] kelvin_dU {name:20s} max|err| = {ex:.2e}")

check_dU(0.0, 0.0, 1.0, 0.0,   "horizontal")
check_dU(1.0, 2.0, 3.0, 5.0,   "off-origin")

# ── 5. kelvin_dT_dfield ───────────────────────────────────────────────────
def check_dT(xf, yf, xs_, ys_, nx, ny, name):
    dTdx_py, dTdy_py = py_bem.kelvin_dT_dfield((xf, yf), (xs_, ys_), (nx, ny),
                                               E=E, nu=nu, plane_strain=PS)
    dTdx_cpp, dTdy_cpp = bem_cpp.kelvin_dT_dfield(xf, yf, xs_, ys_, nx, ny,
                                                   E, nu, PS)
    ex = max(np.max(np.abs(dTdx_py - dTdx_cpp)),
             np.max(np.abs(dTdy_py - dTdy_cpp)))
    assert ex < 1e-12, f"kelvin_dT {name}: max err = {ex:.3e}"
    print(f"[OK] kelvin_dT {name:20s} max|err| = {ex:.2e}")

check_dT(0.0, 0.0, 1.0, 0.0,  0.0, 1.0, "tangent-normal")
check_dT(1.0, 2.0, 3.0, 5.0, -1.0, 0.0, "negative-x normal")
check_dT(0.0, 0.0, 0.7, 0.3,  0.5, np.sqrt(1-0.25), "oblique-normal")

print("\nALL KERNEL TESTS PASSED")
