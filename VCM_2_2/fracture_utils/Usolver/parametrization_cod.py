"""
COD non-negativity active-set enforcement for HALF mode.

This is isolated so the main solver stays readable.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Set
import numpy as np

from .KKT import solve_kkt_lsq
from .constraints import build_cod_rows


def enforce_cod_nonnegative_active_set(
    *,
    K: np.ndarray,
    rhs: np.ndarray,
    C_base: np.ndarray,
    ridge: float,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    crack_mode: str,
    junction_model: str,
    junction_dof: Dict[int, int],
    branch_end_dof: Dict[Tuple[int, str], int],
    cod_tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Active-set loop:
    - Start with base constraints C_base
    - Add COD=0 constraints on violated midpoints (cod < -cod_tol)
    - Resolve until no new violations or max_iter
    """
    if str(crack_mode).lower().strip() != "half":
        q = solve_kkt_lsq(K, rhs, C_base, ridge=float(ridge))
        return q, C_base

    junction_model = str(junction_model).lower().strip()
    max_iter = max(1, int(max_iter))
    cod_tol = float(cod_tol)
    if not np.isfinite(cod_tol) or cod_tol <= 0:
        cod_tol = 1e-10

    active: Set[Tuple[int, int]] = set()
    q = None
    C_work = C_base

    for _it in range(max_iter):
        C_work = C_base
        if active:
            C_cod = build_cod_rows(
                active_pairs=sorted(active),
                poly_panels=poly_panels,
                offsets=offsets,
                nunk=nunk,
                crack_mode="half",
                junction_model=junction_model,
                junction_dof=junction_dof,
                branch_end_dof=branch_end_dof,
            )
            if getattr(C_cod, "size", 0):
                C_work = np.vstack([C_base, C_cod])

        q = solve_kkt_lsq(K, rhs, C_work, ridge=float(ridge))

        newly_violated: List[Tuple[int, int]] = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[int(pid)]
            Np = int(pp["Np"])
            ds = np.asarray(pp["ds"], float).reshape(-1)
            tmid = np.asarray(pp["t_mid"], float)
            nmid = np.asarray(pp["n_mid"], float)

            # J0 at polyline start
            if junction_model == "strict":
                v0 = int(pp["v_start"])
                if v0 in junction_dof:
                    joff = int(junction_dof[v0])
                    J0 = np.array([float(q[joff + 0]), float(q[joff + 1])], float)
                else:
                    J0 = np.array([0.0, 0.0], float)
            else:
                key = (int(pid), "start")
                if key in branch_end_dof:
                    joff = int(branch_end_dof[key])
                    J0 = np.array([float(q[joff + 0]), float(q[joff + 1])], float)
                else:
                    J0 = np.array([0.0, 0.0], float)

            Bj = np.zeros((Np, 2), float)
            for j in range(Np):
                bI = float(q[off + 2 * j + 0])
                bII = float(q[off + 2 * j + 1])
                tj = np.array([float(tmid[j, 0]), float(tmid[j, 1])], float)
                nj = np.array([float(nmid[j, 0]), float(nmid[j, 1])], float)
                Bj[j, :] = (bII * tj + bI * nj) * float(ds[j])

            prefix = np.cumsum(Bj, axis=0)

            for k in range(Np):
                integ = (prefix[k - 1] if k > 0 else 0.0) + 0.5 * Bj[k]
                Jk = J0 - integ
                nk = np.array([float(nmid[k, 0]), float(nmid[k, 1])], float)
                cod = float(np.dot(nk, Jk))
                if cod < -cod_tol:
                    pair = (int(pid), int(k))
                    if pair not in active:
                        newly_violated.append(pair)

        if not newly_violated:
            return q, C_work
        active.update(newly_violated)

    if q is None:
        q = solve_kkt_lsq(K, rhs, C_base, ridge=float(ridge))
        return q, C_base
    return q, C_work
