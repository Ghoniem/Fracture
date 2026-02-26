
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
from typing import Callable, Dict, Optional
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
    vertex_high: int = 18
    L_limit_mm: float = 20.0
    enable_inner_cycle_plot_save: bool = True
    deformed_network_scale: float = 50.0

    simplification_config: Optional[Dict] = None
    intersection_detect_mode: str = "single_pass"
    intersection_verbose: bool = False
    simplify_each_step: bool = True


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

    tough = ConstantToughness(params.Kc_demo)
    dir_law = MaximumHoopStressLaw()

    evaluator = CandidateEvaluator(material=material,applied=applied,
                                   solver_kwargs=solver_kwargs,
                                   direction_law=dir_law,
                                   enable_ne_half_escalation=False)

    prop = CrackPropagator(cfg=cfg,evaluator=evaluator,
                           toughness=tough,direction_law=dir_law)
    netops = get_netops()
    tip_history = []

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
        })

    def _record_inner_step_reports(network_before_growth, reports, *, outer_cycle, inner_step, global_step):
        if not reports:
            return
        vid_to_xy = {
            int(getattr(v, "id")): (float(getattr(v, "x")), float(getattr(v, "y")))
            for v in getattr(network_before_growth, "vertices", [])
        }
        for rep in reports:
            vid = int(getattr(rep, "tip_vid", -1))
            xy = vid_to_xy.get(vid, (np.nan, np.nan))
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
                grew=bool(getattr(rep, "grew", False)),
                reason=str(getattr(rep, "reason", "")),
            )

    def _record_tip_snapshot(network_state, res_state, *, phase, outer_cycle, inner_step, global_step):
        deg = netops.degree_map(network_state)
        polylines = netops.extract_open_polylines(network_state)
        tips = netops.extract_deg1_tips(network_state, polylines, deg) or []
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

    global_step=0
    last_outer_cycle = 0

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
            if grew==0 or len(net.vertices)>=params.vertex_high:
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

<<<<<<< Updated upstream
        # Post-simplify
        step_dir=out_dir/f"cycle_{cyc:02d}_post_simplify"
        step_dir.mkdir(parents=True,exist_ok=True)
        res=solve_only(net,step_dir,save_inner_outputs=bool(params.enable_inner_cycle_plot_save))
        if plot_hook: plot_hook(f"cycle_{cyc:02d}_post_simplify",res,step_dir)
        _record_tip_snapshot(net, res, phase="outer_post_simplify", outer_cycle=cyc, inner_step=inner_step, global_step=global_step)
=======
            # Post-simplify
            step_dir=out_dir/f"cycle_{cyc:02d}_post_simplify"
            step_dir.mkdir(parents=True,exist_ok=True)
            res=solve_only(net,step_dir)
            if plot_hook: plot_hook(f"cycle_{cyc:02d}_post_simplify",res,step_dir)
        else:
            # Still emit the post_simplify stage using the per-step-simplified net
            step_dir=out_dir/f"cycle_{cyc:02d}_post_simplify"
            step_dir.mkdir(parents=True,exist_ok=True)
            res=solve_only(net,step_dir)
            if plot_hook: plot_hook(f"cycle_{cyc:02d}_post_simplify",res,step_dir)
>>>>>>> Stashed changes

    if plot_hook: plot_hook("final",res,step_dir)
    _record_tip_snapshot(net, res, phase="final", outer_cycle=last_outer_cycle, inner_step=0, global_step=global_step)

    # Persist per-run tip history arrays for downstream aggregation.
    if tip_history:
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
            phase=np.asarray([r["phase"] for r in tip_history], dtype=str),
            outer_cycle=np.asarray([r["outer_cycle"] for r in tip_history], int),
            inner_step=np.asarray([r["inner_step"] for r in tip_history], int),
            global_step=np.asarray([r["global_step"] for r in tip_history], int),
            tip_vid=np.asarray([r["tip_vid"] for r in tip_history], int),
        )

    setattr(res, "tip_history", tip_history)
    return res
