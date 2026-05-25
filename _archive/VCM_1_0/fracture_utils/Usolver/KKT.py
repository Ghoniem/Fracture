"""Small KKT solve utility for constrained least squares."""
from __future__ import annotations
import numpy as np

def solve_kkt_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float)

    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0])
    b = K.T @ rhs

    m = C.shape[0]
    Z = np.zeros((m, m), float)
    KKT = np.block([[A, C.T],
                    [C, Z]])
    bb = np.concatenate([b, np.zeros(m)])
    try:
        sol = np.linalg.solve(KKT, bb)
    except np.linalg.LinAlgError:
        eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-12
        A2 = A + eps * np.eye(A.shape[0])
        KKT = np.block([[A2, C.T],
                        [C, Z]])
        sol = np.linalg.solve(KKT, bb)
    return sol[:A.shape[0]]
