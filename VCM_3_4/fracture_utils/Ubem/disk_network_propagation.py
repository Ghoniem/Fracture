
"""
disk_network_propagation.py
Clean, indentation-safe version.

Features:
- NO crack-only stress plots
- Intersection pass BEFORE simplification
- Delegates TOTAL plotting to external plot_hook
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Tuple
import numpy as np


# ============================================================
# Parameters
# ============================================================
@dataclass
class CrackGrowthParams:
    material_E: float = 200e9
    material_nu: float = 0.30
    plane_stress: bool = False

    solver_kwargs: Optional[Dict] = None

    f0: float = 0.05
    f_fixed: float = 0.05
    step_mode: str = "fixed_step"
    simultaneous_tip_growth: bool = True

    Kc_demo: float = 1e6
    rmax_frac: float = 0.25
    min_pts: int = 8

    max_cycles: int = 10
    max_inner_cycles_per_outer: Optional[int] = None
    vertex_high: int = 18
    L_limit_mm: float = 20.0
    enable_inner_cycle_plot_save: bool = True
    deformed_network_scale: float = 50.0

    simplification_config: Optional[Dict] = None
    intersection_detect_mode: str = "single_pass"
    intersection_verbose: bool = False
    simplify_each_step: bool = True

    # Optional inner-cycle post-processing metrics
    enable_inner_cycle_metrics_save: bool = True
    enable_inner_cycle_metrics_print: bool = True
    domain_area_m2: Optional[float] = None
    disk_radius_m: Optional[float] = None
    arc_half_angle_deg: float = 15.0
    boundary_band_frac: float = 0.05
    loading_axis: Tuple[float, float] = (0.0, 1.0)
    control_mode: str = "force"          # "force" or "displacement"
    force_P_N: Optional[float] = None    # required for force mode (if energetics needed)
    disp_delta_m: Optional[float] = None # required for displacement mode (if energetics needed)
    disk_thickness_m: float = 1.0
    # Optional fixed inner-cycle count for total-cycle indexing:
    # total_cycle_number = (outer_cycle-1)*inner_cycle_max_for_index + inner_step
    inner_cycle_max_for_index: Optional[int] = None
    # Optional callback to compute bounded-disk compliance for each inner cycle:
    # fn(network=..., outer_cycle=..., inner_step=..., global_step=..., out_dir=...) -> float
    compliance_evaluator: Optional[Callable[..., float]] = None
    check_tip_outside_disk: bool = False
    tip_outside_tolerance_m: float = 0.0
    tip_outside_stop_mode: str = "any"  # "any" or "all"
    stop_on_alpha_stabilization: bool = False
    alpha_rel_change_threshold: float = 0.10
    alpha_rel_change_ref_floor: float = 1e-12
    stop_on_all_keff_below_kc: bool = False


# ============================================================
# Intersection update (safe + local imports)
# ============================================================
def update_network_with_intersections(net, detect_mode="single_pass", verbose=False):
    import numpy as _np
    import networkx as _nx
    from preamble import CrackNetworkV4 as _CrackNetworkV4
    from fracture_utils.Ugenerator.generator import CrackNetworkGenerator

    V = _np.array([[int(v.id), float(v.x), float(v.y)] for v in net.vertices], float)
    C = _np.array([[int(e.v0), int(e.v1)] for e in net.edges], int) if len(net.edges) else _np.zeros((0,2), int)

    xs = V[:,1] if V.size else _np.array([0.0])
    ys = V[:,2] if V.size else _np.array([0.0])
    w = float(xs.max() - xs.min()) if V.shape[0] else 1.0
    h = float(ys.max() - ys.min()) if V.shape[0] else 1.0
    pad = 0.25 * max(w, h) + 1e-6
    domain_size = (w + 2*pad, h + 2*pad)

    gen = CrackNetworkGenerator(domain_size=domain_size, seed=None)
    gen.G = _nx.Graph()

    for row in V:
        vid = int(row[0])
        gen.G.add_node(vid, pos=(float(row[1]), float(row[2])))

    pos = _nx.get_node_attributes(gen.G, "pos")
    for a,b in C:
        pa = _np.array(pos[int(a)], float)
        pb = _np.array(pos[int(b)], float)
        L = float(_np.linalg.norm(pb-pa))
        gen.G.add_edge(int(a), int(b), length=L)

    gen._update_vertex_types()
    gen.detect_and_split_intersections(verbose=verbose, mode=detect_mode)

    Vg, Cg, _ = gen.to_arrays()
    C2 = Cg[:,1:3].astype(int) if Cg.size else _np.zeros((0,2), int)

    net_new = _CrackNetworkV4.from_vertices_connectivity(
        vertices=Vg,
        connectivity=C2,
        Nv_max=4,
        validate=True,
    )
    return net_new


# ============================================================
# Runner
# ============================================================
def run_network_growth_uncoupled(
    *,
    bem_dir: Path,
    out_dir: Path,
    vertices: np.ndarray,
    connectivity: np.ndarray,
    params: CrackGrowthParams,
    plot_hook: Optional[Callable] = None,
):

    from preamble import (
        Material, AppliedStress, CrackNetworkV4,
        DCENetworkStaticV4, DCEResultsNetworkV4,
        DCEPlotterV4, DCEPlotterDeformedV4,
        PropagationConfig, ConstantToughness, MaximumHoopStressLaw,
        CandidateEvaluator, CrackPropagator,
        _call,
    )
    from fracture_utils.Ugenerator.crack_network_simplifier import CrackNetworkSimplifier, SimplificationConfig
    from fracture_utils.Upropagation.network_ops import get_netops
    from fracture_utils.Ubem.crack_energetics import topology_metrics_for_network

    mm = 1e-3
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = CrackNetworkV4.from_vertices_connectivity(
        vertices=np.asarray(vertices,float),
        connectivity=np.asarray(connectivity,int),
        Nv_max=4,
        validate=True,
    )

    # Spatial applied stress from BEM grid
    xs = np.load(Path(bem_dir)/"xs.npy")
    ys = np.load(Path(bem_dir)/"ys.npy")
    Sxx = np.nan_to_num(np.load(Path(bem_dir)/"Sxx.npy"),nan=0.0)
    Syy = np.nan_to_num(np.load(Path(bem_dir)/"Syy.npy"),nan=0.0)
    Sxy = np.nan_to_num(np.load(Path(bem_dir)/"Sxy.npy"),nan=0.0)

    from scipy.interpolate import RegularGridInterpolator
    Ixx = RegularGridInterpolator((ys,xs),Sxx,bounds_error=False,fill_value=0.0)
    Iyy = RegularGridInterpolator((ys,xs),Syy,bounds_error=False,fill_value=0.0)
    Ixy = RegularGridInterpolator((ys,xs),Sxy,bounds_error=False,fill_value=0.0)

    def sigma_func(X):
        X=np.asarray(X,float)
        pts=np.column_stack([X[:,1],X[:,0]])
        S=np.zeros((len(X),2,2))
        S[:,0,0]=Ixx(pts)
        S[:,1,1]=Iyy(pts)
        S[:,0,1]=Ixy(pts)
        S[:,1,0]=Ixy(pts)
        return S

    solver_kwargs = dict(
        ne_half=100,
        representation="cheb_quad",
        node_distribution="tip_dense",
        solver_option="parametrized_crack",
        parametrization="polyline",
        nq_col=12,
        nq_stress=16,
        crack_mode="half",
        junction_model="core",
    )
    if isinstance(params.solver_kwargs, dict):
        solver_kwargs.update(params.solver_kwargs)

    # Allow toughness override via solver_kwargs while keeping solve(**solver_kwargs)
    # clean from non-solver keys.
    Kc_from_kwargs = solver_kwargs.pop("Kc_demo", None)
    _ = solver_kwargs.pop("K_demo", None)  # legacy alias; not used by propagation logic

    # In true augmented coupling mode, the outer boundary load is enforced via
    # KKT constraints (C_bem/C_bc). Using the BEM field as an additional
    # "applied" stress can over-drive crack fields (double counting).
    aug_active = isinstance(solver_kwargs.get("augmented_coupling", None), dict)
    keep_bem_applied = bool(solver_kwargs.get("augmented_keep_bem_applied", True))
    if aug_active and (not keep_bem_applied):
        applied = AppliedStress(sigma_xx=0.0, sigma_yy=0.0, sigma_xy=0.0)
    else:
        applied = AppliedStress(sigma_func=sigma_func)
    material = Material(E=params.material_E,nu=params.material_nu,plane_stress=params.plane_stress)

    cfg = PropagationConfig(f0=params.f0,f_fixed=params.f_fixed,
                            step_mode=params.step_mode,
                            simultaneous_tip_growth=params.simultaneous_tip_growth)

    Kc_eff = float(params.Kc_demo if Kc_from_kwargs is None else Kc_from_kwargs)
    tough = ConstantToughness(Kc_eff)
    dir_law = MaximumHoopStressLaw()

    evaluator = CandidateEvaluator(material=material,applied=applied,
                                   solver_kwargs=solver_kwargs,
                                   direction_law=dir_law,
                                   enable_ne_half_escalation=False)

    prop = CrackPropagator(cfg=cfg,evaluator=evaluator,
                           toughness=tough,direction_law=dir_law)
    netops = get_netops()
    tip_history = []
    inner_cycle_rows = []
    inner_cycle_counter = 0

    # Disk geometry inferred from BEM grid unless explicitly provided.
    R_disk = float(params.disk_radius_m) if params.disk_radius_m is not None else float(max(np.max(np.abs(xs)), np.max(np.abs(ys))))
    A_domain = float(params.domain_area_m2) if params.domain_area_m2 is not None else float(np.pi * R_disk * R_disk)
    band = float(max(params.boundary_band_frac, 0.0)) * R_disk
    arc = float(params.arc_half_angle_deg)

    def _region_from_theta(theta_center_deg: float):
        lo = float(theta_center_deg - arc)
        hi = float(theta_center_deg + arc)

        def _f(node_id, node_data):
            pos = node_data.get("pos", None)
            if pos is None:
                return False
            x, y = float(pos[0]), float(pos[1])
            r = float(np.hypot(x, y))
            if r < (R_disk - band):
                return False
            th = float(np.degrees(np.arctan2(y, x)))
            # wrap into (-180, 180]
            th = ((th + 180.0) % 360.0) - 180.0
            return (th >= lo) and (th <= hi)

        return _f

    top_region = _region_from_theta(90.0)
    bot_region = _region_from_theta(-90.0)

    def _total_network_length(network_state) -> float:
        total = 0.0
        for e in getattr(network_state, "edges", []):
            p0 = np.asarray(network_state.vertex_coords(int(e.v0)), float).reshape(2,)
            p1 = np.asarray(network_state.vertex_coords(int(e.v1)), float).reshape(2,)
            total += float(np.linalg.norm(p1 - p0))
        return float(total)

    def _append_tip_history_row(
        *,
        phase,
        outer_cycle,
        inner_step,
        global_step,
        tip_vid,
        x_tip_m,
        y_tip_m,
        KI,
        KII,
        theta_rad=np.nan,
        grew=False,
        reason="",
        delta_a_m=np.nan,
        total_length_m=np.nan,
    ):
        tip_history.append({
            "phase": str(phase),
            "outer_cycle": int(outer_cycle),
            "inner_step": int(inner_step),
            "global_step": int(global_step),
            "tip_vid": int(tip_vid),
            "x_tip_m": float(x_tip_m),
            "y_tip_m": float(y_tip_m),
            "KI_MPa_sqrt_m": float(KI) * 1e-6,
            "KII_MPa_sqrt_m": float(KII) * 1e-6,
            "theta_rad": float(theta_rad),
            "theta_deg": float(np.degrees(theta_rad)) if np.isfinite(theta_rad) else np.nan,
            "grew": bool(grew),
            "reason": str(reason),
            "delta_a_m": float(delta_a_m),
            "total_length_m": float(total_length_m),
        })

    def _record_inner_cycle_metrics(network_state, *, outer_cycle, inner_step, global_step, reports=None):
        nonlocal inner_cycle_counter
        if params.inner_cycle_max_for_index is not None and int(params.inner_cycle_max_for_index) > 0:
            total_cycle_number = (int(outer_cycle) - 1) * int(params.inner_cycle_max_for_index) + int(inner_step)
        else:
            # Fallback: strictly monotonic count over executed inner cycles (1-based)
            total_cycle_number = int(inner_cycle_counter) + 1
        topo = topology_metrics_for_network(
            network_state,
            domain_area=A_domain,
            region_1=top_region,
            region_2=bot_region,
            loading_axis=tuple(params.loading_axis),
        )
        C_rel = np.nan
        if callable(params.compliance_evaluator):
            try:
                C_rel = float(
                    params.compliance_evaluator(
                        network=network_state,
                        outer_cycle=int(outer_cycle),
                        inner_step=int(inner_step),
                        global_step=int(global_step),
                        out_dir=out_dir,
                    )
                )
            except Exception:
                C_rel = np.nan
        # K_eff from all tip reports in this inner cycle (MPa*sqrt(m))
        keff_vals = []
        for rep in (reports or []):
            KI = float(getattr(rep, "KI", np.nan))
            KII = float(getattr(rep, "KII", np.nan))
            if np.isfinite(KI) and np.isfinite(KII):
                keff_vals.append(float(np.sqrt(KI * KI + KII * KII) * 1e-6))
        if keff_vals:
            keff_arr = np.asarray(keff_vals, float)
            keff_mean = float(np.mean(keff_arr))
            keff_max = float(np.max(keff_arr))
            keff_min = float(np.min(keff_arr))
            n_tips_eval = int(keff_arr.size)
        else:
            keff_mean = np.nan
            keff_max = np.nan
            keff_min = np.nan
            n_tips_eval = 0

        row = dict(
            cycle_index=int(inner_cycle_counter),
            total_cycle_number=int(total_cycle_number),
            outer_cycle=int(outer_cycle),
            inner_step=int(inner_step),
            global_step=int(global_step),
            L_total_m=float(topo.get("L_total_m", np.nan)),
            alpha=float(topo.get("alpha", np.nan)),
            P_infty=float(topo.get("P_infty", np.nan)),
            Q=float(topo.get("Q", np.nan)),
            C_rel_m_per_N=float(C_rel) if np.isfinite(C_rel) else np.nan,
            K_eff_mean_MPa_sqrt_m=float(keff_mean),
            K_eff_max_MPa_sqrt_m=float(keff_max),
            K_eff_min_MPa_sqrt_m=float(keff_min),
            n_tip_reports=int(n_tips_eval),
        )
        if inner_cycle_rows:
            alpha_prev = float(inner_cycle_rows[-1].get("alpha", np.nan))
            alpha_curr = float(row.get("alpha", np.nan))
            if np.isfinite(alpha_prev) and np.isfinite(alpha_curr):
                denom = max(abs(alpha_prev), float(max(params.alpha_rel_change_ref_floor, np.finfo(float).eps)))
                row["alpha_rel_change_from_prev"] = float(abs(alpha_curr - alpha_prev) / denom)
            else:
                row["alpha_rel_change_from_prev"] = np.nan
        else:
            row["alpha_rel_change_from_prev"] = np.nan
        inner_cycle_rows.append(row)
        if bool(params.enable_inner_cycle_metrics_print):
            print(
                "[inner-metrics] "
                f"total={row['total_cycle_number']} outer={row['outer_cycle']} inner={row['inner_step']} "
                f"L={row['L_total_m']:.6e} alpha={row['alpha']:.6e} "
                f"dalpha_rel={row['alpha_rel_change_from_prev']:.6e} "
                f"Pinf={row['P_infty']:.6e} Q={row['Q']:.6e} "
                f"Crel={row['C_rel_m_per_N']:.6e} "
                f"Keff_mean={row['K_eff_mean_MPa_sqrt_m']:.6e} "
                f"Keff_max={row['K_eff_max_MPa_sqrt_m']:.6e} "
                f"ntip={row['n_tip_reports']}"
            )
        inner_cycle_counter += 1

    def _build_inner_cycle_metrics_payload(rows):
        if not rows:
            return None
        metrics = {k: np.asarray([r[k] for r in rows]) for k in rows[0].keys()}

        L_total = np.asarray(metrics["L_total_m"], float)
        C_rel_raw = np.asarray(metrics["C_rel_m_per_N"], float)
        C_rel = np.abs(C_rel_raw)
        metrics["C_rel_m_per_N_signed"] = C_rel_raw
        metrics["C_rel_m_per_N"] = C_rel
        A_crack = 2.0 * float(params.disk_thickness_m) * L_total
        dC_dA = np.full_like(C_rel, np.nan, dtype=float)
        for i in range(1, len(C_rel)):
            dA = float(A_crack[i] - A_crack[i - 1])
            if abs(dA) > 0.0 and np.isfinite(C_rel[i]) and np.isfinite(C_rel[i - 1]):
                dC_dA[i] = (C_rel[i] - C_rel[i - 1]) / dA

        U = np.full_like(C_rel, np.nan, dtype=float)
        W = np.full_like(C_rel, np.nan, dtype=float)
        Phi = np.full_like(C_rel, np.nan, dtype=float)
        G = np.full_like(C_rel, np.nan, dtype=float)

        mode = str(params.control_mode).strip().lower()
        if mode == "force":
            if params.force_P_N is not None:
                P0 = float(params.force_P_N)
                U = 0.5 * (P0 ** 2) * C_rel
                W = (P0 ** 2) * C_rel
                Phi = U - W
                G = 0.5 * (P0 ** 2) * dC_dA
        elif mode == "displacement":
            if params.disp_delta_m is not None:
                d0 = float(params.disp_delta_m)
                with np.errstate(divide="ignore", invalid="ignore"):
                    U = 0.5 * (d0 ** 2) / C_rel
                    Phi = U
                    G = 0.5 * (d0 ** 2) * dC_dA / (C_rel * C_rel)
                W[:] = np.nan

        metrics["A_crack_m2"] = A_crack
        metrics["dC_dA_1_per_Pa"] = dC_dA
        metrics["U_J"] = U
        metrics["W_J"] = W
        metrics["Phi_J"] = Phi
        metrics["G_J_per_m2"] = G
        metrics["control_mode"] = np.asarray([str(params.control_mode)] * len(L_total), dtype=str)
        return metrics

    def _persist_inner_cycle_metrics():
        if not bool(params.enable_inner_cycle_metrics_save):
            return None
        metrics = _build_inner_cycle_metrics_payload(inner_cycle_rows)
        if metrics is None:
            return None
        np.savez(out_dir / "inner_cycle_metrics.npz", **metrics)
        return metrics

    def _persist_tip_history():
        if not tip_history:
            return
        np.savez(
            out_dir / "tip_history.npz",
            x_tip_m=np.asarray([r["x_tip_m"] for r in tip_history], float),
            y_tip_m=np.asarray([r["y_tip_m"] for r in tip_history], float),
            KI_MPa_sqrt_m=np.asarray([r["KI_MPa_sqrt_m"] for r in tip_history], float),
            KII_MPa_sqrt_m=np.asarray([r["KII_MPa_sqrt_m"] for r in tip_history], float),
            theta_rad=np.asarray([r.get("theta_rad", np.nan) for r in tip_history], float),
            theta_deg=np.asarray([r.get("theta_deg", np.nan) for r in tip_history], float),
            grew=np.asarray([bool(r.get("grew", False)) for r in tip_history], bool),
            reason=np.asarray([r.get("reason", "") for r in tip_history], dtype=str),
            delta_a_m=np.asarray([r.get("delta_a_m", np.nan) for r in tip_history], float),
            total_length_m=np.asarray([r.get("total_length_m", np.nan) for r in tip_history], float),
            phase=np.asarray([r["phase"] for r in tip_history], dtype=str),
            outer_cycle=np.asarray([r["outer_cycle"] for r in tip_history], int),
            inner_step=np.asarray([r["inner_step"] for r in tip_history], int),
            global_step=np.asarray([r["global_step"] for r in tip_history], int),
            tip_vid=np.asarray([r["tip_vid"] for r in tip_history], int),
        )

    def _record_inner_step_reports(network_before_growth, reports, *, outer_cycle, inner_step, global_step):
        if not reports:
            return
        base_length_m = _total_network_length(network_before_growth)
        total_growth_m = 0.0
        for rep in reports:
            if bool(getattr(rep, "grew", False)):
                total_growth_m += float(getattr(rep, "delta_a", 0.0))
        length_after_step_m = float(base_length_m + total_growth_m)
        vid_to_xy = {
            int(getattr(v, "id")): (float(getattr(v, "x")), float(getattr(v, "y")))
            for v in getattr(network_before_growth, "vertices", [])
        }
        for rep in reports:
            vid = int(getattr(rep, "tip_vid", -1))
            xy = vid_to_xy.get(vid, (np.nan, np.nan))
            grew_i = bool(getattr(rep, "grew", False))
            _append_tip_history_row(
                phase="inner_step",
                outer_cycle=outer_cycle,
                inner_step=inner_step,
                global_step=global_step,
                tip_vid=vid,
                x_tip_m=xy[0],
                y_tip_m=xy[1],
                KI=float(getattr(rep, "KI", np.nan)),
                KII=float(getattr(rep, "KII", np.nan)),
                theta_rad=float(getattr(rep, "theta", np.nan)),
                grew=grew_i,
                reason=str(getattr(rep, "reason", "")),
                delta_a_m=float(getattr(rep, "delta_a", 0.0)) if grew_i else 0.0,
                total_length_m=length_after_step_m,
            )

    def _record_tip_snapshot(network_state, res_state, *, phase, outer_cycle, inner_step, global_step):
        deg = netops.degree_map(network_state)
        polylines = netops.extract_open_polylines(network_state)
        tips = netops.extract_deg1_tips(network_state, polylines, deg) or []
        total_length_m = _total_network_length(network_state)
        for tip in tips:
            vid = int(getattr(tip, "v_tip", -1))
            x_tip, y_tip = np.asarray(getattr(tip, "x_tip"), float).reshape(2,)
            try:
                tev = evaluator.eval_tip(res_state, tip)
                KI = float(getattr(tev, "KI", np.nan))
                KII = float(getattr(tev, "KII", np.nan))
            except Exception:
                KI = np.nan
                KII = np.nan
            _append_tip_history_row(
                phase=phase,
                outer_cycle=outer_cycle,
                inner_step=inner_step,
                global_step=global_step,
                tip_vid=vid,
                x_tip_m=x_tip,
                y_tip_m=y_tip,
                KI=KI,
                KII=KII,
                theta_rad=np.nan,
                grew=False,
                reason="snapshot",
                delta_a_m=np.nan,
                total_length_m=total_length_m,
            )

    simp_cfg_defaults = dict(
        max_angle_deviation=3.0,
        min_edge_length=0.2 * mm,
        remove_degree2_nodes=False,
        merge_at_degree2=True,
        merge_vertex_tolerance=1e-6,
        max_merged_edge_length=5 * mm,
        preserve_tips=True,
        preserve_junctions=True,
    )
    if isinstance(params.simplification_config, dict):
        simp_cfg_defaults.update(params.simplification_config)
    simp_cfg = SimplificationConfig(**simp_cfg_defaults)

    def solve_only(network,step_dir,save_inner_outputs=True):
        print(f"[solve] start: {step_dir.name} (Nv={len(network.vertices)}, Ne={len(network.edges)})")
        calc=DCENetworkStaticV4(material,network,applied)
        sol=calc.solve(**solver_kwargs)
        print(f"[solve] done : {step_dir.name}")
        res=DCEResultsNetworkV4(calc,sol)

        if save_inner_outputs:
            pl=DCEPlotterV4(res,out_dir=step_dir)
            _call(pl,"plot_network_graph")

            pl_def=DCEPlotterDeformedV4(res,out_dir=step_dir)
            _call(
                pl_def,
                "plot_deformed_network",
                scale=float(getattr(params, "deformed_network_scale", 50.0)),
                trim_core_junction_faces=True,
            )

        return res

    global_step = 0
    last_outer_cycle = 0
    terminate_run = False
    terminate_tag = ""
    res = None

    def _tip_outside_counts(network_state) -> Tuple[int, int]:
        tol = float(max(params.tip_outside_tolerance_m, 0.0))
        deg = netops.degree_map(network_state)
        polylines = netops.extract_open_polylines(network_state)
        tips = netops.extract_deg1_tips(network_state, polylines, deg) or []
        n_tips = int(len(tips))
        n_outside = 0
        for tip in tips:
            x_tip, y_tip = np.asarray(getattr(tip, "x_tip"), float).reshape(2,)
            if float(np.hypot(x_tip, y_tip)) > (R_disk + tol):
                n_outside += 1
        return n_tips, int(n_outside)

    def _tips_outside_disk_stop_condition(network_state) -> Tuple[bool, int, int, str]:
        n_tips, n_outside = _tip_outside_counts(network_state)
        mode_raw = str(getattr(params, "tip_outside_stop_mode", "any")).strip().lower()
        mode = "all" if mode_raw == "all" else "any"
        if n_tips <= 0:
            return False, n_tips, n_outside, mode
        if mode == "all":
            return bool(n_outside == n_tips), n_tips, n_outside, mode
        return bool(n_outside > 0), n_tips, n_outside, mode

    def _all_keff_below_kc(reports) -> Tuple[bool, int, int, float, float]:
        n_total = 0
        keff_vals = []
        for rep in (reports or []):
            n_total += 1
            KI = float(getattr(rep, "KI", np.nan))
            KII = float(getattr(rep, "KII", np.nan))
            if np.isfinite(KI) and np.isfinite(KII):
                keff_vals.append(float(np.sqrt(KI * KI + KII * KII)))
        n_finite = int(len(keff_vals))
        if n_total <= 0 or n_finite != n_total:
            return False, int(n_total), int(n_finite), np.nan, np.nan
        keff_arr = np.asarray(keff_vals, float)
        keff_min = float(np.min(keff_arr))
        keff_max = float(np.max(keff_arr))
        return bool(np.all(keff_arr <= float(Kc_eff))), int(n_total), int(n_finite), keff_min, keff_max

    def _alpha_relative_change_from_last_two_rows() -> float:
        if len(inner_cycle_rows) < 2:
            return np.nan
        alpha_prev = float(inner_cycle_rows[-2].get("alpha", np.nan))
        alpha_curr = float(inner_cycle_rows[-1].get("alpha", np.nan))
        if not (np.isfinite(alpha_prev) and np.isfinite(alpha_curr)):
            return np.nan
        denom = max(abs(alpha_prev), float(max(params.alpha_rel_change_ref_floor, np.finfo(float).eps)))
        return float(abs(alpha_curr - alpha_prev) / denom)

    try:
        # INITIAL
        step_dir=out_dir/"cycle_00_initial"
        step_dir.mkdir(parents=True,exist_ok=True)
        res=solve_only(net,step_dir)
        if plot_hook: plot_hook("initial",res,step_dir)
        _record_tip_snapshot(net, res, phase="initial", outer_cycle=0, inner_step=0, global_step=global_step)

        for cyc in range(1,params.max_cycles+1):
            last_outer_cycle = int(cyc)
            inner_step = 0

            # Growth loop
            while True:
                net_before_growth = net
                result=prop.grow_one_increment(net_before_growth)
                inner_step += 1
                _record_inner_step_reports(
                    net_before_growth,
                    result.reports,
                    outer_cycle=cyc,
                    inner_step=inner_step,
                    global_step=global_step + 1,
                )
                net=result.network_new
                grew=sum(1 for r in result.reports if bool(getattr(r,"grew",False)))
                global_step+=1
                # Enforce intersections + prune small segments after each inner step
                net=update_network_with_intersections(
                    net,
                    detect_mode=params.intersection_detect_mode,
                    verbose=params.intersection_verbose,
                )
                simplifier=CrackNetworkSimplifier(config=simp_cfg)
                V=np.array([[int(v.id),v.x,v.y] for v in net.vertices],float)
                C=np.array([[int(e.v0),int(e.v1)] for e in net.edges],int)
                simplifier.load_from_arrays(V,C)
                simplifier.simplify(verbose=False)
                V2,C2=simplifier.to_arrays()

                net=CrackNetworkV4.from_vertices_connectivity(
                    vertices=V2,connectivity=C2,Nv_max=4,validate=True)

                need_inner_metrics = bool(params.enable_inner_cycle_metrics_save) or bool(params.stop_on_alpha_stabilization)
                if need_inner_metrics:
                    _record_inner_cycle_metrics(
                        net,
                        outer_cycle=cyc,
                        inner_step=inner_step,
                        global_step=global_step,
                        reports=result.reports,
                    )
                    _persist_inner_cycle_metrics()

                if bool(params.check_tip_outside_disk):
                    tips_stop, n_tips, n_outside, outside_mode = _tips_outside_disk_stop_condition(net)
                    if tips_stop:
                        print(
                            "[stop] tip-boundary criterion met "
                            f"(mode={outside_mode}, outside={n_outside}/{n_tips}, "
                            f"R={R_disk:.6e} m, tol={float(params.tip_outside_tolerance_m):.6e} m) "
                            f"at outer={cyc}, inner={inner_step}, global_step={global_step}."
                        )
                        terminate_run = True
                        terminate_tag = f"{outside_mode}_tips_outside"
                        break

                if bool(params.stop_on_all_keff_below_kc):
                    keff_stop, n_tips_total, n_tips_finite, keff_min_pa, keff_max_pa = _all_keff_below_kc(result.reports)
                    if keff_stop:
                        print(
                            "[stop] all-tip K_eff criterion met "
                            f"(tips={n_tips_finite}/{n_tips_total}, "
                            f"K_eff_min={keff_min_pa * 1e-6:.6e} MPa*sqrt(m), "
                            f"K_eff_max={keff_max_pa * 1e-6:.6e} MPa*sqrt(m), "
                            f"Kc={float(Kc_eff) * 1e-6:.6e} MPa*sqrt(m)) "
                            f"at outer={cyc}, inner={inner_step}, global_step={global_step}."
                        )
                        terminate_run = True
                        terminate_tag = "all_keff_below_kc"
                        break

                if bool(params.stop_on_alpha_stabilization):
                    alpha_rel_change = _alpha_relative_change_from_last_two_rows()
                    alpha_thresh = float(max(params.alpha_rel_change_threshold, 0.0))
                    if np.isfinite(alpha_rel_change) and (alpha_rel_change <= alpha_thresh):
                        alpha_prev = float(inner_cycle_rows[-2].get("alpha", np.nan))
                        alpha_curr = float(inner_cycle_rows[-1].get("alpha", np.nan))
                        print(
                            "[stop] alpha stabilization criterion met "
                            f"(alpha_prev={alpha_prev:.6e}, alpha_curr={alpha_curr:.6e}, "
                            f"rel_change={alpha_rel_change:.6e}, threshold={alpha_thresh:.6e}) "
                            f"at outer={cyc}, inner={inner_step}, global_step={global_step}."
                        )
                        terminate_run = True
                        terminate_tag = "alpha_stable"
                        break

                reached_inner_cap = (
                    params.max_inner_cycles_per_outer is not None
                    and int(params.max_inner_cycles_per_outer) > 0
                    and int(inner_step) >= int(params.max_inner_cycles_per_outer)
                )
                if reached_inner_cap:
                    print(
                        "[inner-stop] reached max_inner_cycles_per_outer="
                        f"{int(params.max_inner_cycles_per_outer)} at outer={cyc}."
                    )
                    break

                if grew==0 or len(net.vertices)>=params.vertex_high:
                    break

            if terminate_run:
                # Solve current network once for a clean terminal state, then stop outer cycles.
                tag = str(terminate_tag).strip() if str(terminate_tag).strip() else "criterion"
                step_dir=out_dir/f"cycle_{cyc:02d}_terminated_{tag}"
                step_dir.mkdir(parents=True,exist_ok=True)
                res=solve_only(net,step_dir,save_inner_outputs=bool(params.enable_inner_cycle_plot_save))
                if plot_hook: plot_hook(f"cycle_{cyc:02d}_terminated_{tag}",res,step_dir)
                _record_tip_snapshot(net, res, phase=f"terminated_{tag}", outer_cycle=cyc, inner_step=inner_step, global_step=global_step)
                break

            # Pre-simplify (or post-step) visualization
            step_dir=out_dir/f"cycle_{cyc:02d}_pre_simplify"
            step_dir.mkdir(parents=True,exist_ok=True)
            res=solve_only(net,step_dir,save_inner_outputs=bool(params.enable_inner_cycle_plot_save))
            if plot_hook: plot_hook(f"cycle_{cyc:02d}_pre_simplify",res,step_dir)
            _record_tip_snapshot(net, res, phase="outer_pre_simplify", outer_cycle=cyc, inner_step=inner_step, global_step=global_step)

            if not params.simplify_each_step:
                # Intersection BEFORE simplify
                net=update_network_with_intersections(
                    net,
                    detect_mode=params.intersection_detect_mode,
                    verbose=params.intersection_verbose,
                )

                # Simplify
                simplifier=CrackNetworkSimplifier(config=simp_cfg)
                V=np.array([[int(v.id),v.x,v.y] for v in net.vertices],float)
                C=np.array([[int(e.v0),int(e.v1)] for e in net.edges],int)
                simplifier.load_from_arrays(V,C)
                simplifier.simplify(verbose=False)
                V2,C2=simplifier.to_arrays()

                net=CrackNetworkV4.from_vertices_connectivity(
                    vertices=V2,connectivity=C2,Nv_max=4,validate=True)

            # Post-simplify (or per-step-simplified) solve
            step_dir=out_dir/f"cycle_{cyc:02d}_post_simplify"
            step_dir.mkdir(parents=True,exist_ok=True)
            res=solve_only(net,step_dir,save_inner_outputs=bool(params.enable_inner_cycle_plot_save))
            if plot_hook: plot_hook(f"cycle_{cyc:02d}_post_simplify",res,step_dir)
            _record_tip_snapshot(net, res, phase="outer_post_simplify", outer_cycle=cyc, inner_step=inner_step, global_step=global_step)

        if plot_hook: plot_hook("final",res,step_dir)
        _record_tip_snapshot(net, res, phase="final", outer_cycle=last_outer_cycle, inner_step=0, global_step=global_step)
    except KeyboardInterrupt:
        print("[interrupt] KeyboardInterrupt received; persisting partial outputs.")
        raise
    finally:
        _persist_tip_history()
        metrics = _persist_inner_cycle_metrics()
        if (metrics is not None) and (res is not None):
            setattr(res, "inner_cycle_metrics", metrics)
        if res is not None:
            setattr(res, "tip_history", tip_history)

    if res is None:
        raise RuntimeError("No solver result was produced during network growth run.")
    return res
