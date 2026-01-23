"""
KKT_v1.py

Constrained least-squares solvers (v1).

Provides:
- solve_kkt_lsq: equality-constrained least squares (legacy API compatible).
- solve_kkt_lsq_active_set: equality + inequality constrained LS via primal active-set.

Problem form (least squares):
    minimize   0.5 || K q - rhs ||^2 + 0.5 * ridge * ||q||^2
    subject to C q = 0
              G q >= h

We convert to a QP in normal equations:
    A = K^T K + ridge I
    b = K^T rhs
    minimize 0.5 q^T A q - b^T q

Then solve KKT systems for the active set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _as_2d(M: np.ndarray, ncols: int) -> np.ndarray:
    M = np.asarray(M, float)
    if M.size == 0:
        return np.zeros((0, ncols), float)
    if M.ndim == 1:
        return M.reshape(1, -1)
    return M


def solve_kkt_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """
    Legacy equality-constrained least squares.

    Parameters
    ----------
    K : (m,n) operator
    rhs : (m,) target
    C : (p,n) equality constraints (hard), representing C q = 0
    ridge : float >= 0

    Returns
    -------
    q : (n,)
    """
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float)

    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0])
    b = K.T @ rhs

    C = _as_2d(C, A.shape[0])
    m = C.shape[0]

    if m == 0:
        # unconstrained
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=1e-12)[0]

    Z = np.zeros((m, m), float)
    KKT = np.block([[A, C.T],
                    [C, Z]])
    bb = np.concatenate([b, np.zeros(m)])

    try:
        sol = np.linalg.solve(KKT, bb)
    except np.linalg.LinAlgError:
        # Stabilize A and retry; if still singular, go to lstsq.
        eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-12
        A2 = A + eps * np.eye(A.shape[0])
        KKT2 = np.block([[A2, C.T],
                         [C, Z]])
        try:
            sol = np.linalg.solve(KKT2, bb)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(KKT2, bb, rcond=1e-12)[0]

    return sol[:A.shape[0]]


@dataclass
class ActiveSetInfo:
    iterations: int
    added: int
    dropped: int
    final_max_violation: float
    status: str


def solve_kkt_lsq_active_set(
    K: np.ndarray,
    rhs: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    *,
    ridge: float = 0.0,
    max_iter: int = 200,
    tol_violation: float = 1e-10,
    tol_lambda: float = 1e-12,
    rcond: float = 1e-12,
    return_info: bool = False,
) -> np.ndarray | Tuple[np.ndarray, ActiveSetInfo]:
    """
    Active-set solver for:
        minimize 0.5||Kq-rhs||^2 + 0.5*ridge||q||^2
        subject to Cq = 0
                   Gq >= h

    Active set A enforces selected inequalities as equalities:
        G_A q = h_A

    Returns
    -------
    q or (q, info)
    """
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)

    n = int(K.shape[1])
    C = _as_2d(np.asarray(C, float), n)
    G = _as_2d(np.asarray(G, float), n)
    h = np.asarray(h, float).reshape(-1)

    if G.shape[0] != h.shape[0]:
        raise ValueError("G and h row counts must match.")

    # Normal equations
    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(n)
    b = K.T @ rhs

    m_eq = C.shape[0]
    p_ineq = G.shape[0]

    active: List[int] = []
    added = 0
    dropped = 0
    status = "unknown"

    def solve_qp_with_eq(Ceq: np.ndarray, deq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve:
          minimize 0.5 q^T A q - b^T q
          s.t. Ceq q = deq
        Returns q, lambda
        """
        Ceq = _as_2d(Ceq, n)
        deq = np.asarray(deq, float).reshape(-1)
        me = Ceq.shape[0]
        if me == 0:
            try:
                q = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                q = np.linalg.lstsq(A, b, rcond=rcond)[0]
            return q, np.zeros((0,), float)

        Z = np.zeros((me, me), float)
        KKT = np.block([[A, Ceq.T],
                        [Ceq, Z]])
        bb = np.concatenate([b, deq])
        try:
            sol = np.linalg.solve(KKT, bb)
        except np.linalg.LinAlgError:
            # Stabilize the primal block and retry, then lstsq.
            eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
            if not np.isfinite(eps) or eps <= 0:
                eps = 1e-12
            A2 = A + eps * np.eye(n)
            KKT2 = np.block([[A2, Ceq.T],
                             [Ceq, Z]])
            try:
                sol = np.linalg.solve(KKT2, bb)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(KKT2, bb, rcond=rcond)[0]
        q = sol[:n]
        lam = sol[n:]
        return q, lam

    # Initial solve with equalities only
    q, lam = solve_qp_with_eq(C, np.zeros((m_eq,), float))

    for it in range(max_iter):
        # compute violation: Gq - h should be >=0
        if p_ineq == 0:
            status = "no_inequalities"
            break
        v = (G @ q) - h
        min_v = float(np.min(v)) if v.size else 0.0  # most negative = most violated
        if min_v >= -tol_violation:
            # feasible; check multipliers for active inequalities
            if not active:
                status = "optimal_feasible"
                break
            # resolve to get consistent multipliers
            Ceq = np.vstack([C, G[active, :]]) if m_eq else G[active, :]
            deq = np.concatenate([np.zeros((m_eq,), float), h[active]]) if m_eq else h[active]
            q, lam = solve_qp_with_eq(Ceq, deq)
            lam_active = lam[-len(active):] if len(active) else np.zeros((0,), float)
            if lam_active.size and float(np.min(lam_active)) < -tol_lambda:
                j = int(np.argmin(lam_active))
                active.pop(j)
                dropped += 1
                continue
            status = "optimal_active_set"
            break

        # add most violated inequality
        j_add = int(np.argmin(v))  # most negative
        if j_add not in active:
            active.append(j_add)
            added += 1

        Ceq = np.vstack([C, G[active, :]]) if m_eq else G[active, :]
        deq = np.concatenate([np.zeros((m_eq,), float), h[active]]) if m_eq else h[active]
        q, lam = solve_qp_with_eq(Ceq, deq)

    else:
        status = "max_iter_reached"

    # final violation magnitude
    if p_ineq:
        v = (G @ q) - h
        final_max_violation = float(max(0.0, -np.min(v))) if v.size else 0.0
    else:
        final_max_violation = 0.0

    info = ActiveSetInfo(
        iterations=int(min(max_iter, it + 1)) if p_ineq else 0,
        added=int(added),
        dropped=int(dropped),
        final_max_violation=float(final_max_violation),
        status=str(status),
    )
    return (q, info) if return_info else q
