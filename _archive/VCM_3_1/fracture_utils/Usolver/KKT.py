"""Small KKT solve utility for constrained least squares."""
from __future__ import annotations
import numpy as np

def _compress_constraints(C: np.ndarray, d: np.ndarray | None = None, tol: float = 1e-12):
    """
    Return a full-row-rank equivalent constraint matrix Cc such that:
      C x = 0  <=>  Cc x = 0
    Uses an SVD-based row-space compression.
    """
    C = np.asarray(C, float)
    if C.size == 0 or C.shape[0] == 0:
        if d is None:
            return C
        return C, np.asarray(d, float).reshape(-1)

    # SVD on C to find its row-rank
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    if s.size == 0:
        C0 = np.zeros((0, C.shape[1]), float)
        if d is None:
            return C0
        return C0, np.zeros((0,), float)

    # Rank threshold relative to largest singular value
    r = int(np.sum(s > tol * s[0]))
    if r >= C.shape[0]:
        if d is None:
            return C  # already full row rank
        return C, np.asarray(d, float).reshape(-1)

    # Compress: keep independent row-space basis
    UrT = U[:, :r].T
    Cc = UrT @ C  # shape (r, n)
    if d is None:
        return Cc
    dc = UrT @ np.asarray(d, float).reshape(-1)
    return Cc, dc


def solve_kkt_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    return solve_kkt_lsq_eq(K=K, rhs=rhs, C=C, d=None, ridge=ridge, ridge_diag=None)


def solve_kkt_lsq_eq(
    K: np.ndarray,
    rhs: np.ndarray,
    C: np.ndarray,
    d: np.ndarray | None = None,
    ridge: float = 0.0,
    ridge_diag: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve constrained least-squares with linear equality constraints:

        min_x ||K x - rhs||_2^2 + ridge*||x||_2^2 + ||diag(sqrt(ridge_diag)) x||_2^2
        s.t.  C x = d

    If d is None, zero RHS constraints are used.
    """
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float)
    if d is None:
        d = np.zeros((C.shape[0],), float)
    else:
        d = np.asarray(d, float).reshape(-1)
        if d.shape[0] != C.shape[0]:
            raise ValueError("d must have one entry per constraint row in C.")

    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0])
    if ridge_diag is not None:
        rd = np.asarray(ridge_diag, float).reshape(-1)
        if rd.shape[0] != A.shape[0]:
            raise ValueError("ridge_diag must have one entry per unknown.")
        A = A + np.diag(np.maximum(rd, 0.0))
    b = K.T @ rhs

    # --- KEY FIX: remove dependent constraints (prevents singular KKT)
    Cc, dc = _compress_constraints(C, d=d, tol=1e-12)

    m = Cc.shape[0]
    Z = np.zeros((m, m), float)

    # Small diagonal stabilization on A (helps conditioning, not rank(C))
    eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-12
    A2 = A + eps * np.eye(A.shape[0])

    KKT = np.block([[A2, Cc.T],
                    [Cc, Z]])
    bb = np.concatenate([b, dc])

    # --- KEY FIX: never assume KKT is invertible; use solve then lstsq fallback
    try:
        sol = np.linalg.solve(KKT, bb)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, bb, rcond=1e-12)[0]

    return sol[:A.shape[0]]
