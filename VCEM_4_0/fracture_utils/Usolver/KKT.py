"""Constrained least-squares solver using a null-space (range-space) method.

We solve

    min_x ||K x - rhs||^2 + ridge*||x||^2 + ||diag(sqrt(ridge_diag)) x||^2
    s.t.  C x = d

via the null-space method:

    1. Compress C to full row rank Cc (SVD), with the corresponding dc.
    2. Build a particular solution xp in range(Cc^T) with Cc xp = dc.
    3. Build an orthonormal basis Z for null(Cc) via complete QR on Cc^T.
       Then any x = xp + Z y, and Cc x = dc holds for every y.
    4. Solve the unconstrained ridge LS in the reduced variable y of size
       (n - m).  Because xp ∈ range(Cc^T) ⟂ range(Z), the ridge cross-term
       vanishes:  ||x||^2 = ||xp||^2 + ||y||^2.
    5. Reconstruct x = xp + Z y.

Why this matters: the previous saddle-point assembly formed
    [[A2, Cc^T], [Cc, 0]] x_aug = [b; dc]
with A2 = K^T K + eps*I.  When K has more columns than rows (the half-COD
formulation allocates per-side junction DOFs that have no collocation
rows), A is rank-deficient by construction and the saddle KKT is
numerically singular (cond ~ 1e+49 observed on a 1058x1058 case).
Null-space reduction sidesteps the K^T K squaring entirely: the reduced
problem ||K Z y - (rhs - K xp)||^2 has cond(K Z) ≈ cond(K), about 1e4 on
the same case.

For the common m == 0 case (no constraints) this collapses to a plain
ridge-LS via np.linalg.lstsq.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import lsmr


# Iterative-LS tolerances and cap. LSMR's atol/btol bound the relative
# residual; conlim caps the condition estimate (acts like rcond=1/conlim).
# Defaults match np.linalg.lstsq(rcond=1e-12) behaviour for the worst case
# while letting well-conditioned problems converge in ~tens of iterations.
_LSMR_ATOL = 1.0e-10
_LSMR_BTOL = 1.0e-10
_LSMR_CONLIM = 1.0e12
_LSMR_MAXITER_MULT = 4   # cap = mult * min(rows, cols); LSMR scipy default is 4*N


def _compress_constraints(C: np.ndarray, d: np.ndarray | None = None, tol: float = 1e-12):
    """Return a full-row-rank (Cc, dc) equivalent to (C, d).

    Cc x = dc has the same solution set as C x = d, with rank(Cc) = rank(C).
    Uses SVD-based row-space compression.
    """
    C = np.asarray(C, float)
    if C.size == 0 or C.shape[0] == 0:
        if d is None:
            return C
        return C, np.asarray(d, float).reshape(-1)

    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    if s.size == 0:
        C0 = np.zeros((0, C.shape[1]), float)
        if d is None:
            return C0
        return C0, np.zeros((0,), float)

    r = int(np.sum(s > tol * s[0]))
    if r >= C.shape[0]:
        if d is None:
            return C
        return C, np.asarray(d, float).reshape(-1)

    UrT = U[:, :r].T
    Cc = UrT @ C
    if d is None:
        return Cc
    dc = UrT @ np.asarray(d, float).reshape(-1)
    return Cc, dc


def _build_nullspace_basis(Cc: np.ndarray) -> np.ndarray:
    """Return Z (n × (n − m)) orthonormal with Cc Z = 0.

    Complete QR on Cc^T: Cc^T = Q R, Q ∈ O(n).  Then the trailing
    columns Q[:, m:] span null(Cc).  Cc has full row rank m by assumption
    (caller compresses first).
    """
    n = Cc.shape[1]
    m = Cc.shape[0]
    if m == 0:
        return np.eye(n)
    if m >= n:
        return np.zeros((n, 0), float)
    Q, _ = np.linalg.qr(Cc.T, mode="complete")     # Q is n x n
    return Q[:, m:]


def _particular_solution(Cc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Min-norm particular solution xp with Cc xp = dc.

    xp = Cc^T (Cc Cc^T)^{-1} dc.  Cc has full row rank so Cc Cc^T is
    SPD m × m and the solve is cheap.
    """
    if Cc.size == 0 or Cc.shape[0] == 0 or np.all(dc == 0.0):
        return np.zeros((Cc.shape[1] if Cc.size else 0,), float)
    G = Cc @ Cc.T                                   # m x m, SPD
    return Cc.T @ np.linalg.solve(G, dc)


def solve_kkt_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray,
                  ridge: float = 0.0) -> np.ndarray:
    return solve_kkt_lsq_eq(K=K, rhs=rhs, C=C, d=None, ridge=ridge,
                            ridge_diag=None)


def solve_kkt_lsq_eq(
    K: np.ndarray,
    rhs: np.ndarray,
    C: np.ndarray,
    d: np.ndarray | None = None,
    ridge: float = 0.0,
    ridge_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Null-space solve for the constrained ridge least-squares problem.

    See module docstring for the formulation.
    """
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float) if C is not None and np.asarray(C).size \
        else np.zeros((0, K.shape[1]), float)
    if d is None:
        d = np.zeros((C.shape[0],), float)
    else:
        d = np.asarray(d, float).reshape(-1)
        if d.shape[0] != C.shape[0]:
            raise ValueError("d must have one entry per constraint row in C.")

    n = int(K.shape[1])
    if ridge_diag is not None:
        rd = np.asarray(ridge_diag, float).reshape(-1)
        if rd.shape[0] != n:
            raise ValueError("ridge_diag must have one entry per unknown.")
        rd = np.maximum(rd, 0.0)
    else:
        rd = None

    ridge = float(ridge) if (ridge and ridge > 0) else 0.0

    # 1) Compress C to full row rank.
    Cc, dc = _compress_constraints(C, d=d, tol=1e-12)
    m = int(Cc.shape[0])

    # Fast path: no constraints -> plain ridge LS via lstsq on the
    # augmented matrix [K; sqrt(ridge) I; diag(sqrt(rd))].
    if m == 0:
        return _ridge_lstsq(K, rhs, ridge, rd)

    # 2) Particular solution and 3) null-space basis.
    xp = _particular_solution(Cc, dc)
    Z = _build_nullspace_basis(Cc)                  # n x (n - m), orthonormal
    k = Z.shape[1]                                  # reduced dimension
    if k == 0:
        # Constraints fully determine x; no free directions left.
        return xp

    # 4) Reduced ridge LS:  min ||K Z y - (rhs - K xp)||^2
    #                       + ridge ||y||^2
    #                       + ||diag(sqrt(rd)) (xp + Z y)||^2
    # ||xp + Z y||^2 = ||xp||^2 + ||y||^2 because xp ⟂ range(Z), so the
    # scalar-ridge cross-term vanishes; for the diagonal weight rd we
    # stack the full term [diag(sqrt(rd)) Z] y ≈ -diag(sqrt(rd)) xp.
    KZ = K @ Z
    rhs_red = rhs - K @ xp

    # Fold scalar ridge into LSMR's `damp` (it solves min ||Ax-b||^2 +
    # damp^2 ||x||^2 directly, no augmented stacking needed).  Only the
    # per-DOF ridge_diag term gets stacked, since it doesn't have an
    # equivalent damp form.
    damp = float(np.sqrt(ridge)) if ridge > 0.0 else 0.0

    rows = [KZ]
    rhs_rows = [rhs_red]
    if rd is not None and np.any(rd > 0.0):
        W = np.sqrt(rd)                             # length n
        rows.append(W[:, None] * Z)                 # shape n x k
        rhs_rows.append(-(W * xp))

    A_aug = np.vstack(rows) if len(rows) > 1 else rows[0]
    b_aug = np.concatenate(rhs_rows) if len(rhs_rows) > 1 else rhs_rows[0]

    y = _solve_ls_iter(A_aug, b_aug, damp=damp)

    # 5) Reconstruct.
    return xp + Z @ y


def _ridge_lstsq(K: np.ndarray, rhs: np.ndarray,
                 ridge: float, rd: np.ndarray | None) -> np.ndarray:
    """Unconstrained ridge LS: min ||K x - rhs||^2 + ridge ||x||^2 + ||diag(sqrt(rd)) x||^2."""
    damp = float(np.sqrt(ridge)) if ridge > 0.0 else 0.0
    if rd is None or not np.any(rd > 0.0):
        return _solve_ls_iter(K, rhs, damp=damp)
    # Per-DOF ridge_diag: append diag(sqrt(rd)) as a block since damp only
    # supports the uniform-ridge case.
    n = K.shape[1]
    A_aug = np.vstack([K, np.diag(np.sqrt(np.maximum(rd, 0.0)))])
    b_aug = np.concatenate([rhs, np.zeros(n)])
    return _solve_ls_iter(A_aug, b_aug, damp=damp)


def _solve_ls_iter(A: np.ndarray, b: np.ndarray, damp: float = 0.0) -> np.ndarray:
    """Iterative LS via LSMR.

    Solves  min ||A x - b||^2 + damp^2 ||x||^2  using scipy's LSMR (a
    Golub-Kahan bidiagonalisation variant of LSQR with better numerics on
    rectangular and ill-conditioned A).  Falls back to np.linalg.lstsq if
    LSMR hits the iteration cap without converging — never returns a
    silently-bad answer.
    """
    rows, cols = A.shape
    maxit = _LSMR_MAXITER_MULT * min(rows, cols)
    out = lsmr(A, b, damp=float(damp),
               atol=_LSMR_ATOL, btol=_LSMR_BTOL,
               conlim=_LSMR_CONLIM, maxiter=maxit)
    x, istop = out[0], out[1]
    # istop 1..6 are "converged"; 7 is "maxiter reached".
    if istop == 7 or not np.all(np.isfinite(x)):
        if damp > 0.0:
            # Stack ridge row block for lstsq fallback.
            A_aug = np.vstack([A, damp * np.eye(cols)])
            b_aug = np.concatenate([b, np.zeros(cols)])
            x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=1.0 / _LSMR_CONLIM)
        else:
            x, *_ = np.linalg.lstsq(A, b, rcond=1.0 / _LSMR_CONLIM)
    return x
