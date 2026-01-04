"""
utils.dce_solver_v3

Multi-crack (array of cracks) static mixed-mode DCE solver.

Key features
-----------
* Fully-coupled linear system (global solve across all cracks).
* Supports:
    - "cheb_quad"
    - "cheb_spectral"
* Uses the verified v2 single-crack stress kernel to assemble coupling columns:
    unit DOF on crack j -> traction (tn, ts) on crack i collocation points.

This avoids analytical LEFM fields and preserves numerical verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Reuse your working V2 single-crack implementation and helper functions
from utils.dce_solver_v2 import (
    Material,
    Crack,
    AppliedStress,
    DCESingleCrackStatic,
    element_nodes,
    gauss_legendre,
    tip_weight,
)


# -------------------------
# Utility: constrained LSQ via KKT
# -------------------------
def solve_constrained_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """
    Minimize ||K u - rhs||^2 subject to C u = 0.

    Uses KKT system:
        [K^T K + ridge I,  C^T][u] = [K^T rhs]
        [C            ,   0 ][λ]   [   0   ]
    """
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float)

    A = K.T @ K
    if ridge and ridge > 0:
        A = A + ridge * np.eye(A.shape[0])
    b = K.T @ rhs

    m = C.shape[0]
    n = A.shape[0]
    Z = np.zeros((m, m), float)

    KKT = np.block([[A, C.T],
                    [C, Z]])
    bb = np.concatenate([b, np.zeros(m)])

    sol = np.linalg.solve(KKT, bb)
    u = sol[:n]
    return u


# -------------------------
# Multi-crack calc
# -------------------------
class DCEMultiCrackStatic:
    """
    Fully coupled multi-crack solver.

    Parameters
    ----------
    material : Material
    cracks   : list[Crack]
    applied  : AppliedStress (global)
    """

    def __init__(self, material: Material, cracks: list[Crack], applied: AppliedStress):
        self.material = material
        self.cracks = list(cracks)
        self.applied = applied

        # Build one single-crack calculator per crack (reuses v2 kernel/evaluators)
        self._single = [DCESingleCrackStatic(material, ck, applied) for ck in self.cracks]

    @property
    def ncracks(self) -> int:
        return len(self.cracks)

    # ---- DOF layout helpers ----
    def _dof_count_one(self, ne_half: int, representation: str) -> int:
        rep = representation.lower()
        if rep == "cheb_quad":
            ne = 2 * int(ne_half)
            # v2: xnodes has size (2*ne + 1)
            return 2 * ne + 1
        if rep == "cheb_spectral":
            N = max(4, 2 * int(ne_half))
            return int(N)
        raise ValueError(f"Unknown representation '{representation}'")

    def _build_collocation_points_local(self, a: float, ne_half: int, node_distribution: str, nq_col: int = 3) -> np.ndarray:
        """
        Collocation points along crack line in local coordinates (y=0).
        Uses Gauss points per element for stability.
        """
        ne = 2 * int(ne_half)
        xe = element_nodes(a, ne, node_distribution=node_distribution)
        xg, _ = gauss_legendre(int(nq_col))

        xcol_list = []
        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm = 0.5 * (xR + xL)
            xcol_list.append(xm + J * xg)
        return np.concatenate(xcol_list)

    def _closure_row_cheb_quad(self, calc_j: DCESingleCrackStatic, ne_half: int, node_distribution: str, nq_src: int = 6) -> np.ndarray:
        """
        Build closure row C for cheb_quad:
            ∫_{-a}^{a} b(x) dx = ∫ g(x)/sqrt(a^2-x^2) dx = 0
        Approximated with quadrature consistent with v2.
        """
        a = calc_j.crack.half_length
        ne = 2 * int(ne_half)
        xe = element_nodes(a, ne, node_distribution=node_distribution)
        xm = 0.5 * (xe[:-1] + xe[1:])

        xnodes = np.empty(2 * ne + 1, float)
        xnodes[0::2] = xe
        xnodes[1::2] = xm
        nd = xnodes.size

        xg, wg = gauss_legendre(int(nq_src))
        C = np.zeros((nd,), float)

        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm_e = 0.5 * (xR + xL)
            xs = xm_e + J * xg
            ws = J * wg

            # quadratic shape on parent [-1,1]
            eta = xg
            N1 = 0.5 * eta * (eta - 1.0)
            N2 = 1.0 - eta * eta
            N3 = 0.5 * eta * (eta + 1.0)
            Ne_loc = np.vstack([N1, N2, N3]).T

            wtip = tip_weight(xs, a)  # 1/sqrt(a^2-x^2)
            idxe = np.array([2 * e, 2 * e + 1, 2 * e + 2], int)

            C[idxe] += (ws[:, None] * (wtip[:, None] * Ne_loc)).sum(axis=0)

        return C

    def _closure_row_cheb_spectral(self, N: int) -> np.ndarray:
        """
        For b(x) = g(s)/sqrt(1-s^2), integral corresponds to pi*c0 (up to scaling).
        Enforcing closure => c0 = 0 is the standard spectral closure.
        """
        C = np.zeros((N,), float)
        C[0] = 1.0
        return C

    # ---- influence-column builder ----
    def _traction_column_from_unit_dof(
        self,
        i_crack: int,
        xcol_i: np.ndarray,
        j_crack: int,
        rep: str,
        mode: str,
        k: int,
        ne_half: int,
        node_distribution: str,
        nq_stress: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (tn_col, ts_col) on crack i collocation points due to unit DOF 'k'
        on crack j (in either mode I or II), using v2 stress kernel.
        """
        calc_i = self._single[i_crack]
        calc_j = self._single[j_crack]

        # Collocation points in GLOBAL coords on crack i (y=0 in i-local)
        a_i = calc_i.crack.half_length
        center_i = np.asarray(calc_i.crack.center, float).reshape(2,)
        Ri = calc_i.R
        Pg = center_i.reshape(2, 1) + (Ri @ np.vstack([xcol_i, np.zeros_like(xcol_i)]))

        # Convert those points to crack j LOCAL coords
        center_j = np.asarray(calc_j.crack.center, float).reshape(2,)
        Rj = calc_j.R
        pl = (Rj.T @ (Pg - center_j.reshape(2, 1)))
        Xl_j = pl[0].reshape(-1)
        Yl_j = pl[1].reshape(-1)

        # Build a "unit" solution dictionary for crack j
        rep = rep.lower()
        mode = mode.upper()

        if rep == "cheb_quad":
            a = calc_j.crack.half_length
            ne = 2 * int(ne_half)
            xe = element_nodes(a, ne, node_distribution=node_distribution)
            xm = 0.5 * (xe[:-1] + xe[1:])
            xnodes = np.empty(2 * ne + 1, float)
            xnodes[0::2] = xe
            xnodes[1::2] = xm
            nd = xnodes.size

            qI = np.zeros(nd, float)
            qII = np.zeros(nd, float)
            if mode == "I":
                qI[int(k)] = 1.0
            else:
                qII[int(k)] = 1.0

            sol_j = dict(
                representation="cheb_quad",
                node_distribution=str(node_distribution),
                ne_half=int(ne_half),
                xnodes=xnodes,
                qI=qI,
                qII=qII,
            )

        elif rep == "cheb_spectral":
            N = max(4, 2 * int(ne_half))
            cI = np.zeros(N, float)
            cII = np.zeros(N, float)
            if mode == "I":
                cI[int(k)] = 1.0
            else:
                cII[int(k)] = 1.0
            sol_j = dict(
                representation="cheb_spectral",
                ne_half=int(ne_half),
                N=int(N),
                coeffs_I=cI,
                coeffs_II=cII,
            )
        else:
            raise ValueError(f"Unknown representation '{rep}'")

        # Evaluate stresses in crack j LOCAL frame at those points
        X = Xl_j.reshape(1, -1)
        Y = Yl_j.reshape(1, -1)
        s11, s22, s12 = calc_j.evaluate_stress_field_local(X, Y, sol_j, add_remote=False, nq_stress=int(nq_stress))

        # Convert j-local stresses to GLOBAL
        s11 = s11.reshape(-1); s22 = s22.reshape(-1); s12 = s12.reshape(-1)
        Sg00 = (Rj[0,0]**2)*s11 + (Rj[0,1]**2)*s22 + 2*Rj[0,0]*Rj[0,1]*s12
        Sg11 = (Rj[1,0]**2)*s11 + (Rj[1,1]**2)*s22 + 2*Rj[1,0]*Rj[1,1]*s12
        Sg01 = (Rj[0,0]*Rj[1,0])*s11 + (Rj[0,1]*Rj[1,1])*s22 + (Rj[0,0]*Rj[1,1]+Rj[0,1]*Rj[1,0])*s12
        Sg10 = Sg01

        # Convert GLOBAL -> i-local
        Rt = Ri.T
        # Si = Rt * Sg * Ri
        Si00 = Rt[0,0]*(Sg00*Ri[0,0] + Sg01*Ri[1,0]) + Rt[0,1]*(Sg10*Ri[0,0] + Sg11*Ri[1,0])
        Si11 = Rt[1,0]*(Sg00*Ri[0,1] + Sg01*Ri[1,1]) + Rt[1,1]*(Sg10*Ri[0,1] + Sg11*Ri[1,1])
        Si01 = Rt[0,0]*(Sg00*Ri[0,1] + Sg01*Ri[1,1]) + Rt[0,1]*(Sg10*Ri[0,1] + Sg11*Ri[1,1])
        # traction on plane y=0 with normal n=(0,1): t = Si @ n => [Si01, Si11]
        ts = Si01
        tn = Si11
        return tn, ts

    # ---- main solve ----
    def solve(
        self,
        ne_half: int,
        representation: str = "cheb_quad",
        *,
        node_distribution: str = "tip_dense",
        nq_col: int = 3,
        nq_stress: int = 6,
        ridge: float = 0.0,
    ) -> dict:
        """
        Fully coupled solve for all cracks.

        Returns solution dict with per-crack blocks and global layout.
        """
        rep = representation.lower()
        nC = self.ncracks

        # Collocation points for each crack (local coords)
        xcol = []
        ncol_each = []
        for i in range(nC):
            a = self._single[i].crack.half_length
            xci = self._build_collocation_points_local(a, ne_half, node_distribution, nq_col=nq_col)
            xcol.append(xci)
            ncol_each.append(xci.size)

        ncol_total = sum(ncol_each)

        # Unknown DOF layout
        ndof_one = [self._dof_count_one(ne_half, rep) for _ in range(nC)]
        # two modes per crack (I and II)
        offsets_I = np.cumsum([0] + ndof_one[:-1])
        offsets_II = offsets_I + sum(ndof_one)
        nunk = 2 * sum(ndof_one)

        # Build global K and rhs (two equations per collocation point: tn, ts)
        n_eq = 2 * ncol_total
        K = np.zeros((n_eq, nunk), float)
        rhs = np.zeros((n_eq,), float)

        # RHS: -remote tractions on each crack in its local frame
        row0 = 0
        for i in range(nC):
            ts0, tn0 = self._single[i].applied_tractions_local()
            ni = ncol_each[i]
            # tn equations
            rhs[row0:row0 + ni] = -tn0
            # ts equations
            rhs[row0 + ncol_total: row0 + ncol_total + ni] = -ts0
            row0 += ni

        # Fill K by influence columns
        # Row indexing: first block = tn rows for all cracks concatenated, second block = ts rows.
        tn_row_start = 0
        ts_row_start = ncol_total

        # Precompute row ranges per crack i
        row_ranges = []
        acc = 0
        for i in range(nC):
            row_ranges.append((acc, acc + ncol_each[i]))
            acc += ncol_each[i]

        for j in range(nC):
            ndj = ndof_one[j]
            for k in range(ndj):
                # Mode I column
                col_I = offsets_I[j] + k
                # Mode II column
                col_II = offsets_II[j] + k

                # compute traction contribution on all cracks i
                for i in range(nC):
                    r0, r1 = row_ranges[i]
                    tn, ts = self._traction_column_from_unit_dof(
                        i_crack=i,
                        xcol_i=xcol[i],
                        j_crack=j,
                        rep=rep,
                        mode="I",
                        k=k,
                        ne_half=ne_half,
                        node_distribution=node_distribution,
                        nq_stress=nq_stress,
                    )
                    K[tn_row_start + r0: tn_row_start + r1, col_I] = tn
                    K[ts_row_start + r0: ts_row_start + r1, col_I] = ts

                    tn, ts = self._traction_column_from_unit_dof(
                        i_crack=i,
                        xcol_i=xcol[i],
                        j_crack=j,
                        rep=rep,
                        mode="II",
                        k=k,
                        ne_half=ne_half,
                        node_distribution=node_distribution,
                        nq_stress=nq_stress,
                    )
                    K[tn_row_start + r0: tn_row_start + r1, col_II] = tn
                    K[ts_row_start + r0: ts_row_start + r1, col_II] = ts

        # Constraints (closure) per crack per mode
        Crows = []
        for j in range(nC):
            ndj = ndof_one[j]
            if rep == "cheb_quad":
                Crow = self._closure_row_cheb_quad(self._single[j], ne_half, node_distribution)
            elif rep == "cheb_spectral":
                Crow = self._closure_row_cheb_spectral(ndj)
            else:
                raise ValueError(f"Unknown representation '{rep}'")

            # Mode I constraint
            Ci = np.zeros((nunk,), float)
            Ci[offsets_I[j]: offsets_I[j] + ndj] = Crow
            Crows.append(Ci)

            # Mode II constraint
            Cii = np.zeros((nunk,), float)
            Cii[offsets_II[j]: offsets_II[j] + ndj] = Crow
            Crows.append(Cii)

        C = np.vstack(Crows)

        # Solve constrained LSQ
        u = solve_constrained_lsq(K, rhs, C, ridge=ridge)

        # Pack solution per crack in the same format v2 evaluators expect
        per_crack = []
        for j in range(nC):
            ndj = ndof_one[j]
            uI = u[offsets_I[j]: offsets_I[j] + ndj]
            uII = u[offsets_II[j]: offsets_II[j] + ndj]

            if rep == "cheb_quad":
                a = self._single[j].crack.half_length
                ne = 2 * int(ne_half)
                xe = element_nodes(a, ne, node_distribution=node_distribution)
                xm = 0.5 * (xe[:-1] + xe[1:])
                xnodes = np.empty(2 * ne + 1, float)
                xnodes[0::2] = xe
                xnodes[1::2] = xm

                solj = dict(
                    representation="cheb_quad",
                    node_distribution=str(node_distribution),
                    ne_half=int(ne_half),
                    xnodes=xnodes,
                    qI=uI.copy(),
                    qII=uII.copy(),
                )
            else:
                solj = dict(
                    representation="cheb_spectral",
                    ne_half=int(ne_half),
                    N=int(ndj),
                    coeffs_I=uI.copy(),
                    coeffs_II=uII.copy(),
                )

            per_crack.append(solj)

        return dict(
            representation=str(rep),
            ne_half=int(ne_half),
            node_distribution=str(node_distribution),
            nq_col=int(nq_col),
            nq_stress=int(nq_stress),
            ridge=float(ridge),
            ncracks=int(nC),
            xcol=xcol,
            per_crack=per_crack,
            u=u,
            layout=dict(
                ndof_one=ndof_one,
                offsets_I=offsets_I.tolist(),
                offsets_II=offsets_II.tolist(),
            ),
        )
