"""Compare C++ solve_kkt_lsq_eq + compress_constraints vs Python on a
handful of representative shapes:
  - unconstrained least squares
  - well-posed equality constraint
  - rank-deficient constraint (dependent rows -- exercises the SVD compression)
  - ridge regularization
  - non-zero d vector

Tolerance: ~1e-9 relative. Both implementations form the SAME KKT block
matrix and call LAPACK via either numpy or Eigen, so deviations should
only come from BDCSVD-vs-numpy-gelsd, never from arithmetic order.
"""

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Usolver import KKT as py_kkt


def check(K, rhs, C, d, ridge, name):
    x_py  = py_kkt.solve_kkt_lsq_eq(K, rhs, C, d, ridge=ridge)
    x_cpp = bem_cpp.solve_kkt_lsq_eq(K, rhs, C, d if d is not None else np.zeros((C.shape[0],)),
                                      ridge=ridge,
                                      backend=bem_cpp.KKTBackend.AutoLU,
                                      constraint_tol=1e-12)
    err = float(np.max(np.abs(x_py - x_cpp)))
    scale = max(float(np.max(np.abs(x_py))), 1e-30)
    rel = err / scale
    assert rel < 1e-9, (f"{name}: rel err={rel:.3e}\n"
                         f"  PY  = {x_py}\n  CPP = {x_cpp}\n  DIFF = {x_py - x_cpp}")
    print(f"[OK] {name:38s} n={K.shape[1]:>3d}  m={C.shape[0]:>2d}  "
          f"max|err|={err:.2e}  rel={rel:.2e}")


def make_random(m, n, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((m, n))
    rhs = rng.standard_normal(m)
    return K, rhs


# 1. Unconstrained least squares (m_c = 0)
K, rhs = make_random(20, 10, seed=1)
C = np.zeros((0, 10))
d = np.zeros((0,))
check(K, rhs, C, d, ridge=0.0, name="unconstrained LS")

# 2. Simple equality constraint, full row rank
K, rhs = make_random(30, 12, seed=2)
rng = np.random.default_rng(7)
C = rng.standard_normal((3, 12))
d = rng.standard_normal(3)
check(K, rhs, C, d, ridge=0.0, name="full-rank constraints, nonzero d")

# 3. Rank-deficient constraints: 4 rows, but row[3] = row[0] + row[1]
K, rhs = make_random(40, 15, seed=3)
C0 = rng.standard_normal((3, 15))
C = np.vstack([C0, (C0[0] + C0[1])[None, :]])
d = np.array([1.0, -2.0, 0.5, 1.0 - 2.0])   # consistent with dependent row
check(K, rhs, C, d, ridge=0.0, name="rank-deficient constraints (SVD compression)")

# 4. Ridge regularization
K, rhs = make_random(15, 25, seed=4)   # underdetermined (m < n)
C = np.zeros((0, 25))
d = np.zeros((0,))
check(K, rhs, C, d, ridge=1e-3, name="ridge-regularized underdetermined")

# 5. Tall thin K with mild ridge + small constraint set
K, rhs = make_random(50, 8, seed=5)
C = rng.standard_normal((2, 8))
d = np.zeros((2,))
check(K, rhs, C, d, ridge=1e-6, name="ridge + zero-RHS constraints")

# 6. Small system end-to-end check (sanity)
K = np.array([[1, 2], [3, 4], [5, 6]], float)
rhs = np.array([1.0, 0.5, -0.5])
C = np.array([[1.0, -1.0]])
d = np.array([0.5])
check(K, rhs, C, d, ridge=0.0, name="3x2 small explicit case")

print("\nALL KKT SOLVE TESTS PASSED")
