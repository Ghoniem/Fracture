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
from .KKT import solve_kkt_lsq

from .build import (
    vertex_degrees,
    build_polylines_full,
    build_polylines_half_branches,
    discretize_polylines,
    assemble_operator,
    allocate_unknowns,
    make_cspline_polyline_from_path,
)

from .constraints import (
    build_constraints_full,
    build_constraints_half,
    build_constraints_half_option_a,
    build_cod_rows,
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
        **_ignored,
    ) -> Dict:
        # Keep signature compatibility with the full solver facade
        _ = (add_vertex_constraints, junction_option, r0_factor, n_int)

        opt = str(solver_option).lower().strip()
        if opt not in ("parametrized_crack", "parameterized_crack", "param_crack"):
            raise ValueError("v4 solver supports solver_option='parametrized_crack' only.")
        if str(parametrization).lower().strip() not in ("polyline", "segmented", "kinked", "cspline", "cubic_spline", "mixed", "arc"):
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
        param_kind = str(parametrization).lower().strip()

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
        # Convert polyline components to interpolating cubic splines (through vertices)
        # - cspline/cubic_spline: convert ALL polyline components with >=3 vertices
        # - mixed: convert ONLY those whose endpoint key is mapped to 'cspline' in branch_param
        branch_param = _ignored.get("branch_param", None)
        if param_kind in ("cspline", "cubic_spline", "mixed"):
            new_polylines = []
            for meta in polylines:
                if str(meta.get("kind", "polyline")).lower() != "polyline":
                    new_polylines.append(meta)
                    continue
                vids_path = list(meta.get("path_vertex_ids", []))
                eidx_path = list(meta.get("path_edge_indices", []))
                if len(vids_path) < 3:
                    new_polylines.append(meta)
                    continue

                if param_kind == "mixed":
                    v0 = int(meta.get("v_start", vids_path[0]))
                    v1 = int(meta.get("v_end", vids_path[-1]))
                    key = (min(v0, v1), max(v0, v1))
                    mode = None
                    if isinstance(branch_param, dict):
                        mode = branch_param.get(key, branch_param.get((v0, v1), None))
                    if str(mode).lower().strip() not in ("cspline", "cubic_spline"):
                        new_polylines.append(meta)
                        continue

                pts = np.array([self.network.vertex_coords(v) for v in vids_path], float)
                new_polylines.append(
                    make_cspline_polyline_from_path(
                        pts,
                        vids_path,
                        eidx_path,
                        crack_mode=crack_mode,
                        v_start=int(meta.get("v_start", vids_path[0])),
                        v_end=int(meta.get("v_end", vids_path[-1])),
                    )
                )
            polylines = new_polylines

        poly_panels = discretize_polylines(
            self.network,
            polylines,
            ne_half=ne_half,
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            representation=rep_in,
            nq_stress=int(nq_stress),
            tip_min_nodes=int(_ignored.get("tip_min_nodes", 0)),
            tip_cluster=str(_ignored.get("tip_cluster", "power")),
            tip_cluster_power=float(_ignored.get("tip_cluster_power", 2.0)),
            other_min_panels=int(_ignored.get("other_min_panels", 1)),
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
        # Optional: enforce COD >= 0 via active-set on panel midpoints (half mode)
        # ------------------------------------------------------------
        enforce_cod = bool(_ignored.get("enforce_cod_nonnegative", False) or _ignored.get("enforce_cod_positive", False))
        cod_tol = float(_ignored.get("cod_tol", 1e-10))
        cod_max_iter = int(_ignored.get("cod_max_iter", 25))
        if not np.isfinite(cod_tol) or cod_tol <= 0:
            cod_tol = 1e-10
        cod_max_iter = max(1, int(cod_max_iter))

        C_base = C
        q = None  # set below

        if crack_mode == "half" and enforce_cod:
            active: set[tuple[int, int]] = set()

            for _it in range(cod_max_iter):
                C_work = C_base
                if active:
                    C_cod = build_cod_rows(
                        active_pairs=sorted(active),
                        poly_panels=poly_panels,
                        offsets=offsets,
                        nunk=nunk,
                        crack_mode=crack_mode,
                        junction_model=junction_model,
                        junction_dof=junction_dof,
                        branch_end_dof=branch_end_dof,
                    )
                    if getattr(C_cod, "size", 0):
                        C_work = np.vstack([C_base, C_cod])

                # Solve with current active set
                q = solve_kkt_lsq(K, rhs, C_work, ridge=float(ridge))

                # Evaluate COD at panel midpoints and add newly violated constraints
                newly_violated: list[tuple[int, int]] = []
                for pid, pp in enumerate(poly_panels):
                    off = offsets[int(pid)]
                    Np = int(pp["Np"])
                    ds = np.asarray(pp["ds"], float).reshape(-1)
                    tmid = np.asarray(pp["t_mid"], float)
                    nmid = np.asarray(pp["n_mid"], float)

                    # J0 at polyline start
                    if str(junction_model).lower().strip() == "strict":
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

                    # Panel contributions Bj = (bII*t + bI*n)*ds
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
                    # Finalize constraint set for packaging/metadata
                    C = C_work
                    break

                active.update(newly_violated)

            # If loop ends by max_iter, keep latest q/C_work
            if q is None:
                q = solve_kkt_lsq(K, rhs, C_base, ridge=float(ridge))
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