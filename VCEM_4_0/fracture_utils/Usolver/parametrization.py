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
from .KKT import solve_kkt_lsq, solve_kkt_lsq_eq

from .build_geometry import vertex_degrees
from .build_dipolar_polyline import build_polylines_full
from .build_polar_polyline import build_polylines_half_branches
from .build_discretize import (
    discretize_polylines,
    allocate_unknowns,
    assemble_operator,
    assemble_boundary_displacement_operator,
    assemble_boundary_traction_operator,
    assemble_bem_boundary_to_crack_traction_operator,
)

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
        n_crack_elements: int,
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

        n_crack_elements = max(2, int(n_crack_elements))
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
            n_crack_elements=n_crack_elements,
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            representation=rep_in,
            nq_stress=int(nq_stress),
            tip_min_nodes=int(_ignored.get("tip_min_nodes", 6)),
            endpoint_min_nodes=int(_ignored.get("endpoint_min_nodes", 6)),
            kink_side_nodes=int(_ignored.get("kink_side_nodes", 2)),
            refine_junction_endpoints=bool(_ignored.get("refine_junction_endpoints", True)),
            refine_kinks=bool(_ignored.get("refine_kinks", True)),
            tip_cluster=str(_ignored.get("tip_cluster", "power")),
            tip_cluster_power=float(_ignored.get("tip_cluster_power", 2.0)),
            other_min_panels=int(_ignored.get("other_min_panels", 1)),
            min_panel_length_ratio=float(_ignored.get("min_panel_length_ratio", 1.0e-3)),
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

        augmented_coupling = _ignored.get("augmented_coupling", None)
        coupling_meta = dict(
            augmented=False,
            augmented_type="",
        )
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
            if augmented_coupling is None:
                q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))
            else:
                ctype = str(augmented_coupling.get("type", "")).lower().strip()
                if ctype != "bem_traction_only":
                    raise ValueError(
                        "augmented_coupling currently supports type='bem_traction_only' only."
                    )

                Xb = np.asarray(augmented_coupling["boundary_xy"], float)
                Nb = np.asarray(augmented_coupling["boundary_n"], float)
                C_bem = np.asarray(augmented_coupling["C_bem"], float)
                C_bc = np.asarray(augmented_coupling["C_bc"], float)
                d_bc = np.asarray(augmented_coupling["d"], float).reshape(-1)

                # crack-induced boundary displacement/traction maps
                Mu = assemble_boundary_displacement_operator(
                    material=self.material,
                    poly_panels=poly_panels,
                    offsets=offsets,
                    nunk=int(nunk),
                    boundary_xy=Xb,
                )
                # M maps crack unknowns q -> crack-induced boundary traction vector
                Mt = assemble_boundary_traction_operator(
                    material=self.material,
                    poly_panels=poly_panels,
                    offsets=offsets,
                    nunk=int(nunk),
                    boundary_xy=Xb,
                    boundary_n=Nb,
                )

                # N maps boundary unknowns y=[u_bc,t_bc] to crack-collocation
                # traction rows used by K/rhs. This is the monolithic coupling
                # term in crack equilibrium: K q + N y ~= rhs.
                bx1 = np.asarray(augmented_coupling.get("boundary_x1", []), float).reshape(-1)
                by1 = np.asarray(augmented_coupling.get("boundary_y1", []), float).reshape(-1)
                bx2 = np.asarray(augmented_coupling.get("boundary_x2", []), float).reshape(-1)
                by2 = np.asarray(augmented_coupling.get("boundary_y2", []), float).reshape(-1)
                if not (bx1.size == by1.size == bx2.size == by2.size == Xb.shape[0]):
                    raise ValueError(
                        "augmented_coupling boundary endpoints are required and must "
                        "match boundary_xy size for monolithic Kq+Ny coupling."
                    )
                Nbc = assemble_bem_boundary_to_crack_traction_operator(
                    material=self.material,
                    poly_panels=poly_panels,
                    boundary_x1=bx1,
                    boundary_y1=by1,
                    boundary_x2=bx2,
                    boundary_y2=by2,
                    gauss_n=int(_ignored.get("augmented_crack_bc_gauss_n", 4)),
                )

                ny = int(C_bem.shape[1])
                n_total = int(nunk) + ny

                # Monolithic crack equilibrium residual:
                #   K q + N y ~= rhs
                Kz = np.hstack([K, Nbc])

                # Base crack constraints (existing) extended with zero columns for y
                C0 = np.hstack([C, np.zeros((C.shape[0], ny), float)]) if C.size else np.zeros((0, n_total), float)
                d0 = np.zeros((C0.shape[0],), float)

                # New coupling constraints:
                # 1) BEM equation: C_bem y = 0
                C1 = np.hstack([np.zeros((C_bem.shape[0], int(nunk)), float), C_bem])
                d1 = np.zeros((C1.shape[0],), float)

                # 2) Boundary compatibility with split selectors:
                #    [Su*ubc + Su*Mu*q = Su*ubar]
                #    [St*tbc + St*Mt*q = St*tbar]
                # If selectors are not provided, default to traction-only BC.
                nb = int(Xb.shape[0])
                I2 = np.eye(2 * nb, dtype=float)
                C_u = np.zeros((2 * nb, 4 * nb), float)
                C_u[:, : 2 * nb] = I2
                C_t = np.zeros((2 * nb, 4 * nb), float)
                C_t[:, 2 * nb :] = I2

                Su = augmented_coupling.get("S_u", None)
                St = augmented_coupling.get("S_t", None)
                if Su is None or St is None:
                    # Backward compatible traction-only setup
                    Su_m = np.zeros((0, 2 * nb), float)
                    St_m = I2.copy()
                else:
                    Su_m = np.asarray(Su, float)
                    St_m = np.asarray(St, float)
                    if Su_m.ndim != 2 or Su_m.shape[1] != 2 * nb:
                        raise ValueError("S_u must have shape (m_u, 2*n_boundary).")
                    if St_m.ndim != 2 or St_m.shape[1] != 2 * nb:
                        raise ValueError("S_t must have shape (m_t, 2*n_boundary).")

                u_bar = np.asarray(augmented_coupling.get("u_bar", np.zeros((2 * nb,), float)), float).reshape(-1)
                if u_bar.size != 2 * nb:
                    raise ValueError("u_bar must have length 2*n_boundary.")
                t_bar = d_bc.copy()
                if t_bar.size != 2 * nb:
                    raise ValueError("d (t_bar) must have length 2*n_boundary.")

                C2_q = np.vstack([Su_m @ Mu, St_m @ Mt])
                C2_y = np.vstack([Su_m @ C_u, St_m @ C_t])
                C2 = np.hstack([C2_q, C2_y])
                d2 = np.concatenate([Su_m @ u_bar, St_m @ t_bar])

                # Optional traction-row nondimensionalization:
                #   \tilde{t} = (L/G) t,  \tilde{M_t} = (L/G) M_t,  \tilde{d_t} = (L/G) d_t
                n_u_rows = int(Su_m.shape[0])
                n_t_rows = int(St_m.shape[0])
                traction_row_scale = 1.0
                if bool(_ignored.get("augmented_scale_traction_rows", True)) and n_t_rows > 0:
                    xx = np.asarray(Xb[:, 0], float)
                    yy = np.asarray(Xb[:, 1], float)
                    Lbc = max(float(np.max(xx) - np.min(xx)), float(np.max(yy) - np.min(yy)))
                    if (not np.isfinite(Lbc)) or Lbc <= 0.0:
                        Lbc = 1.0
                    G = float(getattr(self.material, "mu", 0.0))
                    if (not np.isfinite(G)) or G <= 0.0:
                        G = 1.0
                    traction_row_scale = float(Lbc / G)
                    C2[n_u_rows : n_u_rows + n_t_rows, :] *= traction_row_scale
                    d2[n_u_rows : n_u_rows + n_t_rows] *= traction_row_scale

                # Optional projection of boundary compatibility constraints to the
                # observable row-space of Mt. This avoids over-constraining the
                # monolithic system in weakly observable crack modes.
                c2_tau = _ignored.get("augmented_c2_project_tau", None)
                c2_rank = -1
                c2_smax = 0.0
                if c2_tau is not None:
                    try:
                        tau = float(c2_tau)
                    except Exception:
                        tau = -1.0
                    if np.isfinite(tau) and tau > 0.0:
                        U_m, s_m, _Vt_m = np.linalg.svd(Mt, full_matrices=False)
                        if s_m.size > 0:
                            c2_smax = float(s_m[0])
                            if c2_smax > 0.0 and np.isfinite(c2_smax):
                                c2_rank = int(np.sum(s_m > tau * c2_smax))
                                if c2_rank > 0:
                                    Pm = U_m[:, :c2_rank].T
                                    C2 = Pm @ C2
                                    d2 = Pm @ d2
                                else:
                                    C2 = np.zeros((0, n_total), float)
                                    d2 = np.zeros((0,), float)

                C_all = np.vstack([C0, C1, C2])
                d_all = np.concatenate([d0, d1, d2])

                ridge_q = float(_ignored.get("augmented_ridge_q", 1e-8))
                if (not np.isfinite(ridge_q)) or ridge_q < 0.0:
                    ridge_q = 1e-8
                ridge_y = float(_ignored.get("augmented_ridge_y", 1e-12))
                if (not np.isfinite(ridge_y)) or ridge_y < 0.0:
                    ridge_y = 1e-12
                rd = np.zeros((n_total,), float)
                if ridge_q > 0:
                    rd[: int(nunk)] = ridge_q
                if ridge_y > 0:
                    rd[int(nunk):] = ridge_y

                # Optional strict monolithic coupling:
                # enforce crack equilibrium rows as hard equalities
                #   K q + N y = rhs
                # instead of weighted least-squares objective.
                enforce_crack_eq = bool(_ignored.get("augmented_enforce_crack_equilibrium", True))
                if enforce_crack_eq:
                    C_all = np.vstack([C_all, Kz])
                    d_all = np.concatenate([d_all, rhs])
                    Kz_obj = np.zeros((0, n_total), float)
                    rhs_obj = np.zeros((0,), float)
                else:
                    Kz_obj = Kz
                    rhs_obj = rhs
                # Physical block scaling for mixed-unit unknowns:
                #   y = [u_bc (m), t_bc (Pa)]
                # Use z = D_phys z_hat with:
                #   u scale ~ L0, t scale ~ E
                # so scaled unknown blocks are closer in magnitude.
                use_phys = bool(_ignored.get("augmented_physical_scaling", True))
                nb_loc = int(Xb.shape[0])
                D_phys = np.ones((n_total,), float)
                L0 = float(_ignored.get("augmented_char_length", 0.0))
                if (not np.isfinite(L0)) or L0 <= 0.0:
                    if nb_loc > 0:
                        xx = np.asarray(Xb[:, 0], float)
                        yy = np.asarray(Xb[:, 1], float)
                        L0 = max(float(np.max(xx) - np.min(xx)), float(np.max(yy) - np.min(yy)))
                if (not np.isfinite(L0)) or L0 <= 0.0:
                    L0 = 1.0
                E0 = float(abs(getattr(self.material, "E", 1.0)))
                if (not np.isfinite(E0)) or E0 <= 0.0:
                    E0 = 1.0
                if use_phys and ny == 4 * nb_loc and nb_loc > 0:
                    D_phys[int(nunk): int(nunk) + 2 * nb_loc] = L0
                    D_phys[int(nunk) + 2 * nb_loc: int(nunk) + 4 * nb_loc] = E0

                Kp = Kz_obj * D_phys[None, :]
                Cp = C_all * D_phys[None, :]

                # Optional equilibration of augmented monolithic system.
                # Unknown transform: z = D_phys D_num z_hat
                use_eq = bool(_ignored.get("augmented_equilibrate", True))
                if use_eq:
                    eps_scale = float(_ignored.get("augmented_equilibrate_eps", 1e-14))
                    if (not np.isfinite(eps_scale)) or eps_scale <= 0.0:
                        eps_scale = 1e-14

                    # Row scaling for objective block
                    rk = np.linalg.norm(Kp, axis=1)
                    Wk = np.ones_like(rk)
                    msk = rk > eps_scale
                    Wk[msk] = 1.0 / rk[msk]
                    K_eq = Wk[:, None] * Kp
                    rhs_eq = Wk * rhs_obj

                    # Row scaling for equality constraints
                    rc = np.linalg.norm(Cp, axis=1)
                    Wc = np.ones_like(rc)
                    msc = rc > eps_scale
                    Wc[msc] = 1.0 / rc[msc]
                    C_eq = Wc[:, None] * Cp
                    d_eq = Wc * d_all

                    # Optional extra numeric column equilibration after physical scaling.
                    use_col_eq = bool(_ignored.get("augmented_equilibrate_cols", False))
                    if use_col_eq:
                        A_stack = np.vstack([K_eq, C_eq])
                        cn = np.linalg.norm(A_stack, axis=0)
                        D_num = np.ones_like(cn)
                        mcol = cn > eps_scale
                        D_num[mcol] = 1.0 / cn[mcol]
                        dnum_min = float(_ignored.get("augmented_equilibrate_cols_min", 1e-6))
                        dnum_max = float(_ignored.get("augmented_equilibrate_cols_max", 1e6))
                        if np.isfinite(dnum_min) and np.isfinite(dnum_max) and dnum_min > 0 and dnum_max > dnum_min:
                            D_num = np.clip(D_num, dnum_min, dnum_max)
                    else:
                        D_num = np.ones((n_total,), float)

                    K_hat = K_eq * D_num[None, :]
                    C_hat = C_eq * D_num[None, :]

                    D_tot = D_phys * D_num
                    # Ridge on z becomes diagonal ridge on z_hat via D_tot^2.
                    rd_hat = rd * (D_tot * D_tot)
                    if float(ridge) > 0.0:
                        rd_hat = rd_hat + float(ridge) * (D_tot * D_tot)
                    ridge_hat = 0.0

                    z_hat = solve_kkt_lsq_eq(
                        K=K_hat,
                        rhs=rhs_eq,
                        C=C_hat,
                        d=d_eq,
                        ridge=float(ridge_hat),
                        ridge_diag=rd_hat,
                    )
                    z = D_tot * z_hat
                else:
                    # Physical scaling only (no equilibration)
                    rd_hat = rd * (D_phys * D_phys)
                    if float(ridge) > 0.0:
                        rd_hat = rd_hat + float(ridge) * (D_phys * D_phys)
                    z_hat = solve_kkt_lsq_eq(
                        K=Kz_obj * D_phys[None, :],
                        rhs=rhs_obj,
                        C=Cp,
                        d=d_all,
                        ridge=0.0,
                        ridge_diag=rd_hat,
                    )
                    z = D_phys * z_hat
                q = z[: int(nunk)]
                y_aug = z[int(nunk):].copy()
                coupling_meta = dict(
                    augmented=True,
                    augmented_type=ctype,
                    y_bc=y_aug,
                    n_boundary=int(Xb.shape[0]),
                    boundary_x1=np.asarray(augmented_coupling.get("boundary_x1", []), float).copy(),
                    boundary_y1=np.asarray(augmented_coupling.get("boundary_y1", []), float).copy(),
                    boundary_x2=np.asarray(augmented_coupling.get("boundary_x2", []), float).copy(),
                    boundary_y2=np.asarray(augmented_coupling.get("boundary_y2", []), float).copy(),
                    d_mode=str(np.asarray(augmented_coupling.get("d_mode", [""])).reshape(-1)[0]),
                    c2_project_tau=(None if c2_tau is None else float(c2_tau)),
                    c2_project_rank=int(c2_rank),
                    c2_project_smax=float(c2_smax),
                    nbc_norm=float(np.linalg.norm(Nbc)),
                    mu_norm=float(np.linalg.norm(Mu)),
                    mt_norm=float(np.linalg.norm(Mt)),
                    traction_row_scale=float(traction_row_scale),
                    equilibrated=bool(use_eq),
                    enforce_crack_equilibrium=bool(enforce_crack_eq),
                    physical_scaling=bool(use_phys),
                    scale_L0=float(L0),
                    scale_E0=float(E0),
                )

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
                # Keep quadrature/source geometry for consistent stress reconstruction.
                src_pts=np.array(pp["src_pts"], float),
                src_t=np.array(pp["src_t"], float),
                src_n=np.array(pp["src_n"], float),
                src_w=np.array(pp["src_w"], float),
                panel_src=list(pp["panel_src"]),
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
            n_crack_elements=int(n_crack_elements),
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            crack_mode=crack_mode,
            n_polylines=int(len(polylines)),
            parametrized_polylines=polylines,
            parametrized_edge_to_polyline=edge_to_polyline,
            polyline_solutions=poly_solutions,
            meta=dict(nq_stress=int(nq_stress), ridge=float(ridge), ndof=int(nunk)),
            constraints=dict(n_constraints=int(C.shape[0]), junction_model=junction_model, soft_eta=float(soft_eta)),
            coupling=coupling_meta,
        )
        return sol
