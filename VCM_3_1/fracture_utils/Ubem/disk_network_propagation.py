
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
    deformed_plot_scale: float = 5e3

    simplification_config: Optional[Dict] = None


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
    def _hook_requests_stop(ret) -> bool:
        if isinstance(ret, bool):
            return bool(ret)
        if isinstance(ret, dict):
            return bool(ret.get("stop", False))
        return False

    from preamble import (
        Material, AppliedStress, CrackNetworkV4,
        DCENetworkStaticV4, DCEResultsNetworkV4,
        DCEPlotterV4, DCEPlotterDeformedV4,
        PropagationConfig, ConstantToughness, MaximumHoopStressLaw,
        CandidateEvaluator, CrackPropagator,
        _call,
    )
    from fracture_utils.Ugenerator.crack_network_simplifier import CrackNetworkSimplifier, SimplificationConfig

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

    simp_cfg = SimplificationConfig(max_angle_deviation=3.0,
                                    min_edge_length=0.2*mm,
                                    remove_degree2_nodes=False,
                                    merge_at_degree2=True,
                                    merge_vertex_tolerance=1e-6,
                                    max_merged_edge_length=5*mm,
                                    preserve_tips=True,
                                    preserve_junctions=True)

    def solve_only(network,step_dir):
        print(f"[solve] start: {step_dir.name} (Nv={len(network.vertices)}, Ne={len(network.edges)})")
        calc=DCENetworkStaticV4(material,network,applied)
        sol=calc.solve(**solver_kwargs)
        print(f"[solve] done : {step_dir.name}")
        res=DCEResultsNetworkV4(calc,sol)

        pl=DCEPlotterV4(res,out_dir=step_dir)
        _call(pl,"plot_network_graph")

        pl_def=DCEPlotterDeformedV4(res,out_dir=step_dir)
        _call(
            pl_def,
            "plot_deformed_network",
            scale=float(getattr(params, "deformed_plot_scale", 5e3)),
            trim_core_junction_faces=True,
        )

        return res

    # INITIAL
    step_dir=out_dir/"cycle_00_initial"
    step_dir.mkdir(parents=True,exist_ok=True)
    res=solve_only(net,step_dir)
    stop_requested = False
    if plot_hook:
        stop_requested = _hook_requests_stop(plot_hook("initial",res,step_dir))

    global_step=0

    for cyc in range(1,params.max_cycles+1):
        if stop_requested:
            break

        # Growth loop
        while True:
            result=prop.grow_one_increment(net)
            net=result.network_new
            grew=sum(1 for r in result.reports if bool(getattr(r,"grew",False)))
            global_step+=1
            if grew==0 or len(net.vertices)>=params.vertex_high:
                break

        # Pre-simplify
        step_dir=out_dir/f"cycle_{cyc:02d}_pre_simplify"
        step_dir.mkdir(parents=True,exist_ok=True)
        res=solve_only(net,step_dir)
        if plot_hook:
            stop_requested = _hook_requests_stop(plot_hook(f"cycle_{cyc:02d}_pre_simplify",res,step_dir))
            if stop_requested:
                break

        # Intersection BEFORE simplify
        net=update_network_with_intersections(net,verbose=False)

        # Simplify
        simplifier=CrackNetworkSimplifier(config=simp_cfg)
        V=np.array([[int(v.id),v.x,v.y] for v in net.vertices],float)
        C=np.array([[int(e.v0),int(e.v1)] for e in net.edges],int)
        simplifier.load_from_arrays(V,C)
        simplifier.simplify(verbose=False)
        V2,C2=simplifier.to_arrays()

        net=CrackNetworkV4.from_vertices_connectivity(
            vertices=V2,connectivity=C2,Nv_max=4,validate=True)

        # Post-simplify
        step_dir=out_dir/f"cycle_{cyc:02d}_post_simplify"
        step_dir.mkdir(parents=True,exist_ok=True)
        res=solve_only(net,step_dir)
        if plot_hook:
            stop_requested = _hook_requests_stop(plot_hook(f"cycle_{cyc:02d}_post_simplify",res,step_dir))
            if stop_requested:
                break

    if plot_hook: plot_hook("final",res,step_dir)
    return res
