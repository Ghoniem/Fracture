"""
DCE Calculations Module - Version 1 (CLEAN VARIATIONAL)
======================================================

This file is a drop-in replacement for CrackUtils/dce_calculations_v1.py.

Key properties:
  * Keeps the public API used by your notebook:
        Material, Crack, AppliedStress, DCECalculatorV1
  * Uses a stabilized constrained variational formulation (least-squares + Tikhonov)
    to solve for Burgers magnitudes at prescribed dislocation nodes.
  * Removes the common null/drift modes with TWO hard constraints per mode:
        sum(b)     = 0        (closure / removes constant offset)
        sum(x*b)   = 0        (centering / removes linear drift that shifts COD peak)
  * Designed as a foundation for large systems (matrix-free acceleration can replace
    dense assembly later without changing the variational structure).

Notes:
  - This module computes stresses in LOCAL crack coordinates using standard edge
    dislocation fields for isotropic elasticity (plane strain convention).
  - Current implementation assembles dense matrices (suitable for development and
    moderate N). For large crack networks, replace the dense A-assembly with
    operator application (A@x and A.T@y) and use iterative solvers.

Author: Clean rebuild by ChatGPT (GPT-5.2 Thinking) for Nasr Ghoniem
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy.linalg import solve


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class Material:
    mu: float               # shear modulus [Pa]
    nu: float               # Poisson's ratio [-]


@dataclass
class Crack:
    length: float           # total crack length 2a [m]
    center: np.ndarray      # global center position [m] (2,)
    angle: float = 0.0      # crack angle in global frame [rad]


@dataclass
class AppliedStress:
    sigma_xx: float         # [Pa]
    sigma_yy: float         # [Pa]
    sigma_xy: float         # [Pa]


class DCEPolarity(Enum):
    NEGATIVE = -1
    POSITIVE = +1


class DCEElement:
    """
    DCE element representing one HALF crack.
    The element carries a set of dislocation nodes along the crack line.
    """

    def __init__(self, polarity: DCEPolarity, n_dislocations: int,
                 tip_position: float, base_position: float):
        self.polarity = polarity
        self.n_dislocations = int(n_dislocations)
        self.tip_position = float(tip_position)
        self.base_position = float(base_position)

        # Nodes along x1 (local crack axis). Note: ordering can be tip->base.
        self.positions_x1 = np.linspace(self.tip_position, self.base_position, self.n_dislocations)

        # Unknown Burgers magnitudes (signed) for each node
        self.b_mode_I = np.zeros(self.n_dislocations, dtype=float)    # b2 component
        self.b_mode_II = np.zeros(self.n_dislocations, dtype=float)   # b1 component

    def get_burgers_vector_local(self, i: int) -> np.ndarray:
        """
        Burgers vector in LOCAL coordinates (b1 along x1, b2 along x2).
        """
        return np.array([self.b_mode_II[i], self.b_mode_I[i]], dtype=float)


# ---------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------

class DCECalculatorV1:
    def __init__(self, material: Material, crack: Crack, stress: AppliedStress):
        self.material = material
        self.crack = crack
        self.stress = stress

        self.a = 0.5 * float(crack.length)

        # Crack local basis in global coordinates
        c = float(np.cos(crack.angle))
        s = float(np.sin(crack.angle))
        # x1: tangent, x2: normal
        self.crack_direction_global = np.array([c, s], dtype=float)
        self.crack_normal_global = np.array([-s, c], dtype=float)

    # ----------------------------
    # Coordinate transforms
    # ----------------------------
    def global_to_local(self, x_global: np.ndarray) -> np.ndarray:
        xg = np.asarray(x_global, dtype=float) - np.asarray(self.crack.center, dtype=float)
        x1 = float(np.dot(xg, self.crack_direction_global))
        x2 = float(np.dot(xg, self.crack_normal_global))
        return np.array([x1, x2], dtype=float)

    def local_to_global(self, x_local: np.ndarray) -> np.ndarray:
        xl = np.asarray(x_local, dtype=float)
        return (np.asarray(self.crack.center, dtype=float)
                + xl[0] * self.crack_direction_global
                + xl[1] * self.crack_normal_global)

    def stress_local_to_global(self, sigma_local: np.ndarray) -> np.ndarray:
        """
        Rotate local stress tensor to global basis.
        """
        Q = np.column_stack([self.crack_direction_global, self.crack_normal_global])  # columns are basis vectors
        return Q @ sigma_local @ Q.T

    def get_applied_stress_local(self) -> np.ndarray:
        """
        Applied stress tensor in LOCAL crack coordinates.
        """
        sxx = float(self.stress.sigma_xx)
        syy = float(self.stress.sigma_yy)
        sxy = float(self.stress.sigma_xy)
        Sg = np.array([[sxx, sxy],
                       [sxy, syy]], dtype=float)

        Q = np.column_stack([self.crack_direction_global, self.crack_normal_global])
        # local = Q^T * global * Q
        return Q.T @ Sg @ Q

    # ----------------------------
    # Kernel: edge dislocation stress (LOCAL)
    # ----------------------------
    def stress_from_edge_dislocation(self, r: np.ndarray, b_vec: np.ndarray, disl_pos: np.ndarray) -> np.ndarray:
        """
        Stress from a single edge dislocation in LOCAL coordinates.

        r: evaluation point (x1,x2)
        disl_pos: dislocation position (x1,x2)
        b_vec: Burgers vector (b1,b2) in local coordinates
        """
        x1 = float(r[0] - disl_pos[0])
        x2 = float(r[1] - disl_pos[1])
        r2 = x1*x1 + x2*x2
        if r2 < (1e-12)**2:
            return np.zeros((2, 2), dtype=float)

        D = self.material.mu / (2.0 * np.pi * (1.0 - self.material.nu))
        r4 = r2 * r2
        b1 = float(b_vec[0])
        b2 = float(b_vec[1])

        sig = np.zeros((2, 2), dtype=float)

        # contribution from b1 (Mode-II basis)
        if abs(b1) > 1e-30:
            sig[0, 0] += -D * b1 * x2 * (3.0*x1*x1 + x2*x2) / r4
            sig[1, 1] +=  D * b1 * x2 * (x1*x1 - x2*x2) / r4
            sig[0, 1] +=  D * b1 * x1 * (x1*x1 - x2*x2) / r4

        # contribution from b2 (Mode-I basis)
        if abs(b2) > 1e-30:
            sig[0, 0] +=  D * b2 * x1 * (x1*x1 - x2*x2) / r4
            sig[1, 1] += -D * b2 * x1 * (x1*x1 + 3.0*x2*x2) / r4
            sig[0, 1] += -D * b2 * x2 * (x1*x1 - x2*x2) / r4

        sig[1, 0] = sig[0, 1]
        return sig

    # ----------------------------
    # Element generation
    # ----------------------------
    def create_symmetric_crack_elements(self, n_dislocations_per_element: int, tip_fraction: float) -> List[DCEElement]:
        """
        Create two HALF-crack elements that meet at x1=0:
            NEGATIVE: [-a*tip_fraction, 0]
            POSITIVE: [0, +a*tip_fraction]
        """
        a_eff = self.a * float(tip_fraction)

        neg = DCEElement(DCEPolarity.NEGATIVE, n_dislocations_per_element, tip_position=-a_eff, base_position=0.0)
        pos = DCEElement(DCEPolarity.POSITIVE, n_dislocations_per_element, tip_position=+a_eff, base_position=0.0)

        return [neg, pos]

    # ----------------------------
    # Variational solve (per mode) with constraints
    # ----------------------------
    @staticmethod
    def _second_difference_L(n: int) -> np.ndarray:
        """
        Second-difference operator for stabilization (size (n-2) x n).
        Penalizes curvature of b(x) and suppresses checkerboard modes.
        """
        if n <= 2:
            return np.zeros((0, n), dtype=float)
        L = np.zeros((n-2, n), dtype=float)
        for i in range(n-2):
            L[i, i] = 1.0
            L[i, i+1] = -2.0
            L[i, i+2] = 1.0
        return L

    def _assemble_A(self, x_eval: np.ndarray, x_src: np.ndarray, mode: str) -> np.ndarray:
        """
        Build dense influence matrix A where:
          mode = "I"  -> A[i,j] = sigma_22 at eval i due to unit b2 at source j
          mode = "II" -> A[i,j] = sigma_12 at eval i due to unit b1 at source j
        """
        n = x_eval.size
        A = np.zeros((n, n), dtype=float)

        if mode == "I":
            b_unit = np.array([0.0, 1.0], dtype=float)
            comp = (1, 1)
        elif mode == "II":
            b_unit = np.array([1.0, 0.0], dtype=float)
            comp = (0, 1)
        else:
            raise ValueError("mode must be 'I' or 'II'")

        for i in range(n):
            ri = np.array([x_eval[i], 0.0], dtype=float)
            for j in range(n):
                if i == j:
                    # core-regularized self-term: evaluate at a small offset
                    # (avoids singularity and improves conditioning)
                    rj = np.array([x_src[j], 0.0], dtype=float)
                    r_core = np.array([x_eval[i], 1e-12], dtype=float)
                    sig = self.stress_from_edge_dislocation(r_core, b_unit, rj)
                else:
                    rj = np.array([x_src[j], 0.0], dtype=float)
                    sig = self.stress_from_edge_dislocation(ri, b_unit, rj)
                A[i, j] = sig[comp[0], comp[1]]
        return A

    def _solve_mode_variational(self, A: np.ndarray, rhs: np.ndarray, x_sorted: np.ndarray,
                               reg_alpha: float) -> np.ndarray:
        """
        Solve:  min ||A b - rhs||^2 + reg_alpha ||L b||^2
               s.t. sum(b)=0 and sum(x*b)=0

        Unknowns are in the SAME ordering as x_sorted.
        """
        n = A.shape[0]
        L = self._second_difference_L(n)
        K = A.T @ A
        if L.shape[0] > 0 and reg_alpha > 0:
            K = K + float(reg_alpha) * (L.T @ L)
        f = A.T @ rhs

        # Hard constraints (2): closure + centering
        C = np.column_stack([
            np.ones(n, dtype=float),
            x_sorted.astype(float)
        ])  # (n,2)

        M = np.block([
            [K, C],
            [C.T, np.zeros((2, 2), dtype=float)]
        ])
        rhs_kkt = np.concatenate([f, np.zeros(2, dtype=float)])

        sol = solve(M, rhs_kkt)
        return np.asarray(sol[:n], dtype=float)

    def solve_mixed_mode(self, dce_elements: List[DCEElement], reg_alpha: float = 1e-2) -> List[DCEElement]:
        """
        Variational stabilized mixed-mode solve for a joined crack made of half elements.

        Enforces traction-free in a least-squares sense at collocation points (currently chosen
        as the dislocation nodes), with stabilization and two hard constraints per mode:
            sum(b)=0 and sum(x*b)=0.

        reg_alpha: stabilization strength (dimensionless). Try 1e-3 to 1e-1.
        """
        sigma_local = self.get_applied_stress_local()
        sigma_22 = float(sigma_local[1, 1])
        sigma_12 = float(sigma_local[0, 1])

        # Build global node list
        disl_map: List[Tuple[int, int]] = []
        x1_global = []
        for eidx, elem in enumerate(dce_elements):
            for lidx in range(elem.n_dislocations):
                disl_map.append((eidx, lidx))
                x1_global.append(float(elem.positions_x1[lidx]))

        x1_global = np.asarray(x1_global, dtype=float)

        # Sort nodes left->right for stable regularization and consistent constraints
        idx_sort = np.argsort(x1_global)
        idx_unsort = np.argsort(idx_sort)
        x_sorted = x1_global[idx_sort]

        # Collocation points = evaluation points = node points
        x_eval = x_sorted.copy()
        x_src = x_sorted.copy()

        # Assemble influence matrices (in sorted ordering)
        A_I = self._assemble_A(x_eval, x_src, mode="I")
        A_II = self._assemble_A(x_eval, x_src, mode="II")

        rhs_I = -sigma_22 * np.ones_like(x_eval)
        rhs_II = -sigma_12 * np.ones_like(x_eval)

        bI_sorted = self._solve_mode_variational(A_I, rhs_I, x_sorted=x_sorted, reg_alpha=reg_alpha)
        bII_sorted = self._solve_mode_variational(A_II, rhs_II, x_sorted=x_sorted, reg_alpha=reg_alpha)

        # Map solution back to original global ordering
        bI = bI_sorted[idx_unsort]
        bII = bII_sorted[idx_unsort]

        # Scatter back into elements
        for gidx, (eidx, lidx) in enumerate(disl_map):
            dce_elements[eidx].b_mode_I[lidx] = float(bI[gidx])
            dce_elements[eidx].b_mode_II[lidx] = float(bII[gidx])

        return dce_elements

    # ----------------------------
    # SIF calculation (PK-based, as in your original file)
    # ----------------------------
    def calculate_K_I_K_II(self, dce_elements: List[DCEElement]):
        sigma_local = self.get_applied_stress_local()

        # Find rightmost node as "tip" proxy
        tip_elem = None
        tip_elem_idx = None
        tip_local_idx = None
        max_x1 = -np.inf
        for elem_idx, elem in enumerate(dce_elements):
            for local_idx, x1 in enumerate(elem.positions_x1):
                if x1 > max_x1:
                    max_x1 = float(x1)
                    tip_elem = elem
                    tip_elem_idx = elem_idx
                    tip_local_idx = local_idx

        x1_tip = float(tip_elem.positions_x1[tip_local_idx])
        pos_tip = np.array([x1_tip, 0.0], dtype=float)

        sigma_app = np.array([[0.0, sigma_local[0, 1]],
                              [sigma_local[0, 1], sigma_local[1, 1]]], dtype=float)
        sigma_total = sigma_app.copy()

        for eidx, elem in enumerate(dce_elements):
            for lidx in range(elem.n_dislocations):
                if eidx == tip_elem_idx and lidx == tip_local_idx:
                    continue
                pos = np.array([float(elem.positions_x1[lidx]), 0.0], dtype=float)
                b = elem.get_burgers_vector_local(lidx)
                sigma_total += self.stress_from_edge_dislocation(pos_tip, b, pos)

        b_tip = tip_elem.get_burgers_vector_local(tip_local_idx)
        sigma_dot_b = sigma_total @ b_tip

        f1 = sigma_dot_b[1]
        f2 = -sigma_dot_b[0]

        K_I = np.sqrt(2.0 * self.material.mu * abs(f2) / (1.0 - self.material.nu))
        K_II = np.sqrt(2.0 * self.material.mu * abs(f1) / (1.0 - self.material.nu))

        K_I_analytical = abs(float(sigma_local[1, 1])) * np.sqrt(np.pi * self.a)
        K_II_analytical = abs(float(sigma_local[0, 1])) * np.sqrt(np.pi * self.a)

        return K_I, K_II, K_I_analytical, K_II_analytical

    # ----------------------------
    # Driver used by your notebook
    # ----------------------------
    def solve(self, n_dislocations_per_element: int = 30, tip_fraction: float = 0.95,
              reg_alpha: float = 1e-2) -> Dict[str, object]:
        """
        Convenience driver that:
          1) builds two half elements,
          2) solves mixed-mode variational problem,
          3) computes SIFs.

        Returns dict compatible with your plotter.
        """
        dce_elements = self.create_symmetric_crack_elements(
            n_dislocations_per_element=n_dislocations_per_element,
            tip_fraction=tip_fraction
        )

        dce_elements = self.solve_mixed_mode(dce_elements, reg_alpha=reg_alpha)

        K_I, K_II, K_I_ana, K_II_ana = self.calculate_K_I_K_II(dce_elements)
        err_I = abs(K_I - K_I_ana) / K_I_ana * 100.0 if K_I_ana > 0 else 0.0
        err_II = abs(K_II - K_II_ana) / K_II_ana * 100.0 if K_II_ana > 0 else 0.0

        return {
            "dce_elements": dce_elements,
            "K_I": K_I,
            "K_II": K_II,
            "K_I_analytical": K_I_ana,
            "K_II_analytical": K_II_ana,
            "error_I_percent": err_I,
            "error_II_percent": err_II,
            "reg_alpha": float(reg_alpha),
            "tip_fraction": float(tip_fraction),
            "n_dislocations_per_element": int(n_dislocations_per_element),
        }
