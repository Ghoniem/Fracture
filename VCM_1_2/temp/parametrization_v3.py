"""
Refactored half/full parametrized polyline solver (utils_half).

Key goals:
- Keep public API (DCENetworkStaticV4.solve) stable.
- Split implementation into:
  * solver_parametrized_half_build.py        (polyline construction + discretization + operator assembly)
  * solver_parametrized_half_constraints.py  (KKT constraint construction)
- Fix prior indentation / stray-else issues by rewriting the junction/closure logic coherently.

Half mode features:
- Branch polylines (junction-to-tip / junction-to-junction).
- Shared junction DOFs J_vx,J_vy for every endpoint vertex with degree>=2.
- Along-polyline endpoint constraints enforce J_end = J_start - ∫(bII t + bI n) ds when end is a junction.
- Dipole neutrality:
    * deg>=3 vertices: full vector closure (2 constraints, x & y) using signed branch incidence
    * deg==2 vertices: one projected closure along bisector normal (1 constraint), skip near-collinear
- Option A gauge: fix J at ONE degree-1 tip (2 constraints).
"""

from __future__ import annotations
from typing import Dict
import numpy as np

from .material import Material, AppliedStress
from .network import CrackNetworkV4
from .KKT_v1 import solve_kkt_lsq, solve_kkt_lsq_active_set

from .build_v1 import (
    vertex_degrees,
    build_polylines_full,
    build_polylines_half_branches,
    discretize_polylines,
    assemble_operator,
    allocate_unknowns,
)

from .constraints_v1 import (
    build_constraints_full,
    build_constraints_half,
    build_constraints_half_option_a,
    build_cod_inequalities,
)


class DCENetworkStaticV4:
    def __init__(self, material: Material, network: CrackNetworkV4, applied: AppliedStress):
        self.material = material
        self.network = network
        self.applied = applied

    def solve(
        self,
        ne_half: int,
        representation: str = "panel",
        *,
        solver_option: str = "parametrized_crack",
        parametrization: str = "polyline",
        node_distribution: str = "tip_dense",
        nq_col: int = 3,
        nq_stress: int = 6,
        ridge: float = 0.0,
        add_vertex_constraints: bool = False,
        junction_option: int = 0,
        r0_factor: float = 0.0,
        n_int: int = 0,
        crack_mode: str = "full",
        enforce_cod_nonneg: bool = False,
        cod_nonneg_eps: float = 0.0,
        **_ignored,
    ) -> Dict:
        # Keep signature compatibility with the full solver facade
        _ = (add_vertex_constraints, junction_option, r0_factor, n_int)

        opt = str(solver_option).lower().strip()
        if opt not in ("parametrized_crack", "parameterized_crack", "param_crack"):
            raise ValueError("v4 solver supports solver_option='parametrized_crack' only.")
        if str(parametrization).lower().strip() not in ("polyline", "segmented", "kinked"):
            raise ValueError("v4 solver currently supports parametrization='polyline' only.")

        rep_in = str(representation).lower().strip()
        dist_in = str(node_distribution).lower().strip()

        # Chebyshev aliases (same pattern as full solver)
        collocation_mode = "mid"
        if rep_in in ("cheb_quad", "cheb_spectral", "cheb_quad_singular", "cheb_spectral_singular"):
            if rep_in.startswith("cheb_spectral"):
                collocation_mode = "nodes"
            dist_in = "tip_dense"
            rep_in = "singular" if rep_in.endswith("_singular") else "panel"

        if rep_in not in ("panel", "singular"):
            raise ValueError(
                f"Unknown representation={representation!r}. Use 'panel', 'singular', 'cheb_quad', or 'cheb_spectral'."
            )
        if dist_in not in ("uniform", "tip_dense"):
            raise ValueError(f"Unknown node_distribution={node_distribution!r}. Use 'uniform' or 'tip_dense'.")

        crack_mode = str(crack_mode).lower().strip()
        if crack_mode not in ("full", "half"):
            raise ValueError("crack_mode must be 'full' or 'half'.")

        # Junction model for half mode:

        #   'strict' (legacy Option A), 'core' (open junctions), 'soft' (penalized coupling)

        junction_model = str(_ignored.get("junction_model", "strict")).lower().strip()

        if junction_model not in ("strict", "core", "soft"):

            junction_model = "strict"

        soft_eta = float(_ignored.get("soft_eta", 1.0))

        if not np.isfinite(soft_eta) or soft_eta < 0:

            soft_eta = 1.0

        ne_half = max(2, int(ne_half))

        deg = vertex_degrees(self.network)

        # ------------------------------------------------------------
        # Build polylines + edge_to_polyline
        # ------------------------------------------------------------
        if crack_mode == "full":
            polylines, edge_to_polyline = build_polylines_full(self.network)
        else:
            polylines, edge_to_polyline = build_polylines_half_branches(self.network, deg)

        # ------------------------------------------------------------
        # Discretize polylines and assemble operator
        # ------------------------------------------------------------
        poly_panels = discretize_polylines(
            self.network,
            polylines,
            ne_half=ne_half,
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            representation=rep_in,
            nq_stress=int(nq_stress),
        )

        offsets, junction_dof, branch_end_dof, nunk = allocate_unknowns(poly_panels, deg, crack_mode=crack_mode, junction_model=junction_model)

        K, rhs = assemble_operator(
            material=self.material,
            applied=self.applied,
            poly_panels=poly_panels,
            offsets=offsets,
            nunk=nunk,
            nq_col=int(nq_col),
        )

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------
        if crack_mode == "full":
            C = build_constraints_full(
                poly_panels=poly_panels,
                offsets=offsets,
                nunk=nunk,
                use_tip_singular=(rep_in == "singular"),
            )
        else:
            theta_min_flat = float(_ignored.get("theta_min_flat", 5.0 * np.pi / 180.0))

            C, P = build_constraints_half(

                network=self.network,

                deg=deg,

                poly_panels=poly_panels,

                offsets=offsets,

                junction_dof=junction_dof,

                branch_end_dof=branch_end_dof,

                nunk=nunk,

                junction_model=junction_model,

                theta_min_flat=theta_min_flat,

                soft_eta=soft_eta,

            )

        # ------------------------------------------------------------
        # Solve KKT least squares
        # ------------------------------------------------------------
        if crack_mode == "half" and junction_model == "soft" and 'P' in locals() and P is not None and getattr(P, "size", 0):

            # Add soft coupling as additional LS equations: sqrt(eta) * P q ≈ 0

            w = float(np.sqrt(max(soft_eta, 0.0)))

            K = np.vstack([K, w * P])

            rhs = np.concatenate([rhs, np.zeros((P.shape[0],), float)])

        # ------------------------------------------------------------
        # Optional COD >= 0 inequalities (active-set)
        # ------------------------------------------------------------
        if bool(enforce_cod_nonneg):
            G, h, _meta_ineq = build_cod_inequalities(
                poly_panels=poly_panels,
                offsets=offsets,
                nunk=nunk,
                deg=deg,
                crack_mode=crack_mode,
                representation=rep_in,
                eps=float(cod_nonneg_eps),
            )
            if G.shape[0] > 0:
                q = solve_kkt_lsq_active_set(K, rhs, C, G, h, ridge=float(ridge))
            else:
                q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))
        else:
            q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))


        # ------------------------------------------------------------
        # Pack solution
        # ------------------------------------------------------------
        poly_solutions = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[pid]
            Np = int(pp["Np"])

            bI_hat = np.array([q[off + 2 * k + 0] for k in range(Np)], float)
            bII_hat = np.array([q[off + 2 * k + 1] for k in range(Np)], float)

            if rep_in == "singular":
                s_mid = np.array(pp["s_mid"], float)
                Lp = float(pp["L"])
                eps = 1e-14 * Lp if Lp > 0 else 1e-14
                if crack_mode == "full":
                    sing_mid = 1.0 / np.sqrt(np.maximum(s_mid, eps) * np.maximum(Lp - s_mid, eps))
                else:
                    sing_mid = 1.0 / np.sqrt(np.maximum(Lp - s_mid, eps))
                bI = bI_hat * sing_mid
                bII = bII_hat * sing_mid
            else:
                bI = bI_hat.copy()
                bII = bII_hat.copy()

            solp = dict(
                pid=int(pid),
                bI=bI, bII=bII,
                bI_hat=bI_hat, bII_hat=bII_hat,
                s_nodes=np.array(pp["s_nodes"], float),
                s_mid=np.array(pp["s_mid"], float),
                x_mid=np.array(pp["x_mid"], float),
                t_mid=np.array(pp["t_mid"], float),
                n_mid=np.array(pp["n_mid"], float),
                x_col=np.array(pp["x_col"], float),
                t_col=np.array(pp["t_col"], float),
                n_col=np.array(pp["n_col"], float),
                ds=np.array(pp["ds"], float),
                v_start=int(pp["v_start"]),
                v_end=int(pp["v_end"]),
            )

            if crack_mode == "half":


                # Store J0 as the jump at the polyline start endpoint.


                # - strict: shared per-vertex junction_dof


                # - core/soft: per-branch-end DOF for (pid,'start') if present


                if junction_model == "strict":


                    v0 = int(pp["v_start"])


                    if v0 in junction_dof:


                        joff = int(junction_dof[v0])


                        solp["J0"] = np.array([float(q[joff + 0]), float(q[joff + 1])], float)


                    else:


                        solp["J0"] = np.array([0.0, 0.0], float)


                else:


                    key = (int(pid), "start")


                    if key in branch_end_dof:


                        joff = int(branch_end_dof[key])


                        solp["J0"] = np.array([float(q[joff + 0]), float(q[joff + 1])], float)


                    else:


                        solp["J0"] = np.array([0.0, 0.0], float)

            poly_solutions.append(solp)

        sol = dict(
            representation=rep_in,
            solver_option="parametrized_crack",
            parametrization="polyline",
            ne_half=int(ne_half),
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            crack_mode=crack_mode,
            n_polylines=int(len(polylines)),
            parametrized_polylines=polylines,
            parametrized_edge_to_polyline=edge_to_polyline,
            polyline_solutions=poly_solutions,
            meta=dict(nq_stress=int(nq_stress), ridge=float(ridge), ndof=int(nunk)),
            constraints=dict(n_constraints=int(C.shape[0]), junction_model=junction_model, soft_eta=float(soft_eta)),
        )
        return sol
