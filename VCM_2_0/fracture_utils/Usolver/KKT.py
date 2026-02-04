"""Small KKT solve utility for constrained least squares."""
from __future__ import annotations
import numpy as np

import numpy as np

def _compress_constraints(C: np.ndarray, tol: float = 1e-12):
    """
    Return a full-row-rank equivalent constraint matrix Cc such that:
      C x = 0  <=>  Cc x = 0
    Uses an SVD-based row-space compression.
    """
    C = np.asarray(C, float)
    if C.size == 0 or C.shape[0] == 0:
        return C

    # SVD on C to find its row-rank
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    if s.size == 0:
        return np.zeros((0, C.shape[1]), float)

    # Rank threshold relative to largest singular value
    r = int(np.sum(s > tol * s[0]))
    if r >= C.shape[0]:
        return C  # already full row rank

    # Compress: keep independent row-space basis
    Cc = (U[:, :r].T @ C)  # shape (r, n)
    return Cc


def solve_kkt_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float)

    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0])
    b = K.T @ rhs

    # --- KEY FIX: remove dependent constraints (prevents singular KKT)
    Cc = _compress_constraints(C, tol=1e-12)

    m = Cc.shape[0]
    Z = np.zeros((m, m), float)

    # Small diagonal stabilization on A (helps conditioning, not rank(C))
    eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-12
    A2 = A + eps * np.eye(A.shape[0])

    KKT = np.block([[A2, Cc.T],
                    [Cc, Z]])
    bb = np.concatenate([b, np.zeros(m)])

    # --- KEY FIX: never assume KKT is invertible; use solve then lstsq fallback
    try:
        sol = np.linalg.solve(KKT, bb)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, bb, rcond=1e-12)[0]

    return sol[:A.shape[0]]
