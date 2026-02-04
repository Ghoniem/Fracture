"""
Parametrized polyline solver (v2.0), refactored and consistent with split build modules.

Public API:
- DCENetworkStaticV4.solve(...)

Build stack (new split):
- build_geometry.vertex_degrees
- build_dipolar_polyline.build_polylines_full
- build_polar_polyline.build_polylines_half_branches
- build_discretize.discretize_polylines / allocate_unknowns / assemble_operator

Curved conversion:
- build_curved.make_cspline_polyline_from_path (via preprocess.convert_polylines_to_csplines)
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from .material import Material, AppliedStress
from .network import CrackNetworkV4
from .KKT import solve_kkt_lsq

from .build_geometry import vertex_degrees
from .build_dipolar_polyline import build_polylines_full
from .build_polar_polyline import build_polylines_half_branches
from .build_discretize import discretize_polylines, allocate_unknowns, assemble_operator

from .constraints import (
    build_constraints_full,
    build_constraints_half,
)

from .parametrization_preprocess import (
    normalize_representation,
    convert_polylines_to_csplines,
)

from .parametrization_cod import enforce_cod_nonnegative_active_set


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
        **_ignored,
    ) -> Dict:
        # Keep signature compatibility with the full solver facade
        _ = (add_vertex_constraints, junction_option, r0_factor, n_int)

        opt = str(solver_option).lower().strip()
        if opt not in ("parametrized_crack", "parameterized_crack", "param_crack"):
            raise ValueError("v2.0 solver supports solver_option='parametrized_crack' only.")

        param_kind = str(parametrization).lower().strip()
        if param_kind not in ("polyline", "segmented", "kinked", "cspline", "cubic_spline", "mixed", "arc"):
            raise ValueError("v2.0 solver expects parametrization='polyline'|'cspline'|'mixed' (aliases allowed).")

        crack_mode = str(crack_mode).lower().strip()
        if crack_mode not in ("full", "half"):
            raise ValueError("crack_mode must be 'full' or 'half'.")

        rep_in, dist_in, collocation_mode = normalize_representation(representation, node_distribution)

        # Junction model for half mode:
        #   'strict' (shared vertex jump DOF), 'core'/'soft' (per-branch-end jump DOFs)
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

        # Optional conversion of straight polylines to cubic splines
        branch_param = _ignored.get("branch_param", None)
        polylines = convert_polylines_to_csplines(
            self.network,
            polylines,
            crack_mode=crack_mode,
            parametrization=param_kind,
            branch_param=branch_param if isinstance(branch_param, dict) else None,
        )

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
            tip_min_nodes=int(_ignored.get("tip_min_nodes", 0)),
            endpoint_min_nodes=int(_ignored.get("endpoint_min_nodes", 8)),
            kink_side_nodes=int(_ignored.get("kink_side_nodes", 4)),
            refine_junction_endpoints=bool(_ignored.get("refine_junction_endpoints", True)),
            refine_kinks=bool(_ignored.get("refine_kinks", True)),
            tip_cluster=str(_ignored.get("tip_cluster", "power")),
            tip_cluster_power=float(_ignored.get("tip_cluster_power", 2.0)),
            other_min_panels=int(_ignored.get("other_min_panels", 1)),
        )

        offsets, junction_dof, branch_end_dof, nunk = allocate_unknowns(
            poly_panels, deg, crack_mode=crack_mode, junction_model=junction_model
        )

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
            P = None
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
        # Soft coupling (half+soft): add sqrt(eta) P q ≈ 0 as LS rows
        # ------------------------------------------------------------
        if crack_mode == "half" and junction_model == "soft" and P is not None and getattr(P, "size", 0):
            w = float(np.sqrt(max(soft_eta, 0.0)))
            K = np.vstack([K, w * P])
            rhs = np.concatenate([rhs, np.zeros((P.shape[0],), float)])

        # ------------------------------------------------------------
        # Optional COD >= 0 (active set)
        # ------------------------------------------------------------
        enforce_cod = bool(
            _ignored.get("enforce_cod_nonnegative", False) or _ignored.get("enforce_cod_positive", False)
        )
        cod_tol = float(_ignored.get("cod_tol", 1e-10))
        cod_max_iter = int(_ignored.get("cod_max_iter", 25))

        if crack_mode == "half" and enforce_cod:
            q, C_used = enforce_cod_nonnegative_active_set(
                K=K,
                rhs=rhs,
                C_base=C,
                ridge=float(ridge),
                poly_panels=poly_panels,
                offsets=offsets,
                nunk=int(nunk),
                crack_mode="half",
                junction_model=junction_model,
                junction_dof=junction_dof,
                branch_end_dof=branch_end_dof,
                cod_tol=float(cod_tol),
                max_iter=int(cod_max_iter),
            )
            C = C_used
        else:
            q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))

        # ------------------------------------------------------------
        # Pack solution
        # ------------------------------------------------------------
        poly_solutions = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[int(pid)]
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
                bI=bI,
                bII=bII,
                bI_hat=bI_hat,
                bII_hat=bII_hat,
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
            parametrization=param_kind,
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
