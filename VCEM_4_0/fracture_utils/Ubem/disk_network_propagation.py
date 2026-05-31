
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
from typing import Callable, Dict, Iterable, Optional, Any, Tuple
import json
import subprocess
import sys
import time
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

    # Disk-radius step size: Δa_ref = f_disk_radius * disk_radius_m.
    # Constant in absolute terms regardless of network length -- this is the
    # preferred knob for the Brazilian-disk pipeline. When disk_radius_m
    # is set (the Excel loader injects cfg.disk.R automatically), f0 /
    # f_fixed / step_mode below are bypassed by NetworkGrowthRunner.
    f_disk_radius: float = 0.05

    # Legacy L_total-fraction step knobs -- kept for non-disk pipelines.
    # Ignored when disk_radius_m > 0.
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
    # When True, also save brazilian_disk_{sxx,syy,sxy}_contour.png in each
    # cycle directory (total stress field = BEM baseline + crack contribution).
    # Renders the per-cycle frames used by tools/make_run_videos.py to build
    # the contour evolution videos.
    enable_per_cycle_total_contour_save: bool = True
    deformed_network_scale: float = 50.0

    # Soft segment-to-segment kink clamp. theta from MTS is measured
    # relative to the previous segment, so |theta| <= max_kink_deg clamps
    # the segment-to-segment direction change. The clamp is bypassed on
    # each tip's first emission (the very first new edge added at that
    # tip) so the initial crack can find its preferred direction; every
    # subsequent emission is clamped. <= 0 disables.
    max_kink_deg: float = 20.0

    # Deprecated alongside simplify_each_step. Values are read but ignored.
    simplification_config: Optional[Dict] = None
    intersection_detect_mode: str = "single_pass"
    intersection_verbose: bool = False
    simp_max_angle_deg: float = 3.0
    simp_min_edge_mm: float = 0.2
    simp_max_merged_edge_mm: float = 5.0
    simp_merge_tolerance_m: float = 1e-6
    # Deprecated: per-polyline segment-merge simplifier was removed when
    # cubic-spline parametrization was rolled back. Field kept for workbook
    # back-compat; value is ignored by the runner.
    simplify_each_step: bool = False

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
    # Simplification pass at the very start of the run and at the start of
    # each outer cycle. (1) merges near-aligned segments at deg-2 nodes
    # (controlled by simp_max_angle_deg / simp_max_merged_edge_mm /
    # simp_merge_tolerance_m); (2) re-detects segment-segment intersections
    # and reclassifies vertex degrees via update_network_with_intersections.
    # Inner-step intersection enforcement is unchanged. Set False to keep
    # the initial / outer-cycle network exactly as grown.
    simplify_at_outer_cycle_start: bool = True

    # Kept for backward compatibility; tip-outside stop is no longer used.
    check_tip_outside_disk: bool = False
    tip_outside_tolerance_m: float = 0.0
    tip_outside_stop_mode: str = "any"  # no-op
    # Kept for backward compatibility; alpha-stabilization stop is no longer used.
    stop_on_alpha_stabilization: bool = False
    alpha_rel_change_threshold: float = 0.10
    alpha_rel_change_ref_floor: float = 1e-12
    stop_on_all_keff_below_kc: bool = False

    # Graceful stop when every deg-1 tip in the propagation result has
    # growth_force == 0 (either snapped at the disk boundary, below the
    # toughness threshold, or eliminated below the delta_a tolerance). The
    # runner sets res.stopped_reason = "all_growth_force_zero" so the
    # outer-coupling driver can break its own loop.
    stop_on_all_growth_force_zero: bool = True

    # Wall-clock runtime limit (seconds). When set, the STEP loop breaks
    # before starting any step that would push the cumulative run time past
    # this threshold; the finalize path then writes step_history.json,
    # plot_step_metrics outputs, and (if enabled) the metric videos before
    # returning. None disables the check.
    runtime_limit_s: Optional[float] = None

    # Auto-encode per-cycle PNGs into MP4s on finalize via
    # tools/make_run_videos.py (subprocessed with the running interpreter).
    # Disable to skip the video step (e.g. on machines without ffmpeg /
    # imageio).
    enable_video_encode: bool = True
    video_fps: int = 4


# ============================================================
# Intersection update (safe + local imports)
# ============================================================
def update_network_with_intersections(net, detect_mode="single_pass", verbose=False, snap_tol_m: float = 0.0):
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
    # Vertex-to-segment snap: generalisation of the segment-segment
    # intersection split. A vertex within snap_tol_m of any non-incident
    # segment is projected onto the closest point of that segment and the
    # segment is split there, so the vertex becomes a junction. When the
    # projection foot lies at (or within snap_tol_m of) a segment endpoint
    # this degenerates to a vertex-vertex merge.
    if float(snap_tol_m) > 0.0:
        gen.detect_and_snap_vertices_to_segments(
            tol=float(snap_tol_m), verbose=verbose,
        )
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
# Outer-cycle simplification (merge aligned + intersect)
# ============================================================
def simplify_and_intersect(
    net,
    *,
    max_angle_deg: float,
    max_merged_edge_m: float,
    merge_vertex_tol_m: float,
    snap_tol_m: float = 0.0,
    detect_mode: str = "single_pass",
    verbose: bool = False,
):
    """Apply outer-cycle simplification: merge near-aligned segments at deg-2
    nodes, then run the snap-and-intersect pass so any vertex within
    ``snap_tol_m`` of a non-incident segment is projected onto that
    segment (forming a new junction), and any segment-segment crossings
    are split as usual.

    Vertex IDs are preserved across the merge step (no renumbering), so
    callers tracking initial-vertex ids do not lose track of which tips
    pre-existed. Snap/intersection splits may still add new vertex ids for
    new interior nodes -- those are degree>=2 by construction.
    """
    import numpy as _np
    from preamble import CrackNetworkV4 as _CrackNetworkV4
    from fracture_utils.Ugenerator.crack_network_simplifier import (
        CrackNetworkSimplifier, SimplificationConfig,
    )

    if len(getattr(net, "edges", [])) == 0:
        return net

    V = _np.array([[int(v.id), float(v.x), float(v.y)] for v in net.vertices], float)
    C = _np.array([[int(e.v0), int(e.v1)] for e in net.edges], int)

    # Only the colinear-merge step is needed here. Vertex-vertex merge is
    # left at the safety-floor tol (the vertex-to-segment snap performed
    # downstream by update_network_with_intersections subsumes it at the
    # user-facing simp_min_edge_mm scale). Small-edge removal is off
    # (min_edge_length=0 makes the `length < min_len` test always false)
    # and the blanket deg-2 collapse is off -- a deg-2 node is collapsed
    # only when its two incident segments are *nearly aligned*.
    simp_cfg = SimplificationConfig(
        max_angle_deviation=float(max_angle_deg),
        min_edge_length=0.0,
        merge_vertex_tolerance=float(merge_vertex_tol_m),
        max_merged_edge_length=float(max_merged_edge_m),
        merge_at_degree2=True,
        remove_degree2_nodes=False,
        preserve_tips=True,
        preserve_junctions=True,
    )
    simp = CrackNetworkSimplifier(simp_cfg)
    simp.load_from_arrays(V, C)
    simp.simplify(verbose=bool(verbose))

    # Convert back PRESERVING ids (do NOT call simp.to_arrays(), which
    # renumbers all nodes sequentially).
    nodes = list(simp.G.nodes())
    V2 = _np.array(
        [[int(nid), float(simp.G.nodes[nid]["pos"][0]), float(simp.G.nodes[nid]["pos"][1])]
         for nid in nodes],
        float,
    )
    if simp.G.number_of_edges() > 0:
        C2 = _np.array([[int(u), int(v)] for u, v in simp.G.edges()], int)
    else:
        C2 = _np.zeros((0, 2), int)

    net_merged = _CrackNetworkV4.from_vertices_connectivity(
        vertices=V2, connectivity=C2, Nv_max=4, validate=True,
    )
    return update_network_with_intersections(
        net_merged,
        detect_mode=detect_mode,
        verbose=verbose,
        snap_tol_m=float(snap_tol_m),
    )


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
    original_initial_vertex_ids: Optional[Iterable[int]] = None,
    snapped_vertex_ids: Optional[Iterable[int]] = None,
    plot_initial: bool = True,
    plot_params: Optional[Any] = None,
):

    from preamble import (
        Material, AppliedStress, CrackNetworkV4,
        DCENetworkStaticV4, DCEResultsNetworkV4,
        DCEPlotterV4, DCEPlotterDeformedV4,
        PropagationConfig, ConstantToughness, MaximumHoopStressLaw,
        CandidateEvaluator, CrackPropagator,
        _call,
    )
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

    # Initial simplification + intersection reclassification, applied once
    # before any growth so the initial solve and tip-history reflect the
    # cleaned network. Per-outer-cycle pass below repeats this between
    # cycles. _init_vids is captured AFTER this so the propagator's
    # first-emission set matches the simplified network.
    if bool(params.simplify_at_outer_cycle_start):
        print("[simplify] initial pass: merge aligned + snap-to-segment + intersect")
        net = simplify_and_intersect(
            net,
            max_angle_deg=float(params.simp_max_angle_deg),
            max_merged_edge_m=float(params.simp_max_merged_edge_mm) * mm,
            merge_vertex_tol_m=float(params.simp_merge_tolerance_m),
            snap_tol_m=float(params.simp_min_edge_mm) * mm,
            detect_mode=params.intersection_detect_mode,
            verbose=bool(params.intersection_verbose),
        )

    # Spatial applied stress from BEM grid
    xs = np.load(Path(bem_dir)/"xs.npy")
    ys = np.load(Path(bem_dir)/"ys.npy")
    Sxx = np.nan_to_num(np.load(Path(bem_dir)/"Sxx.npy"),nan=0.0)
    Syy = np.nan_to_num(np.load(Path(bem_dir)/"Syy.npy"),nan=0.0)
    Sxy = np.nan_to_num(np.load(Path(bem_dir)/"Sxy.npy"),nan=0.0)

    # Defensive shape alignment. RegularGridInterpolator((ys,xs), Z) requires
    # Z.shape == (ys.size, xs.size). Stale arrays in bem_dir (e.g. from a
    # previous run with a different n_grid, or saved in the opposite axis
    # order) trip this check otherwise.
    def _align_stress(S, Ny, Nx, name):
        S = np.asarray(S, float)
        if S.shape == (Ny, Nx):
            return S
        if S.shape == (Nx, Ny):
            return S.T
        raise ValueError(
            f"BEM stress array {name}.npy has shape {S.shape}, which is "
            f"inconsistent with the saved grid (ys.size={Ny}, xs.size={Nx}). "
            f"This usually means stale arrays in bem_dir={bem_dir!s} from a "
            f"prior run with a different n_grid. Set skip_bem_solve=False "
            f"in the workbook (or delete xs.npy/ys.npy/Sxx.npy/Syy.npy/Sxy.npy "
            f"in that dir) to refresh."
        )
    Ny, Nx = int(ys.size), int(xs.size)
    Sxx = _align_stress(Sxx, Ny, Nx, "Sxx")
    Syy = _align_stress(Syy, Ny, Nx, "Syy")
    Sxy = _align_stress(Sxy, Ny, Nx, "Sxy")

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
        n_crack_elements=100,
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

    cfg = PropagationConfig(
        f0=params.f0,
        f_fixed=params.f_fixed,
        step_mode=params.step_mode,
        simultaneous_tip_growth=params.simultaneous_tip_growth,
        f_disk_radius=float(params.f_disk_radius),
        disk_radius_m=float(params.disk_radius_m) if params.disk_radius_m is not None else 0.0,
    )

    Kc_eff = float(params.Kc_demo if Kc_from_kwargs is None else Kc_from_kwargs)
    tough = ConstantToughness(Kc_eff)
    dir_law = MaximumHoopStressLaw()

    evaluator = CandidateEvaluator(material=material,applied=applied,
                                   solver_kwargs=solver_kwargs,
                                   direction_law=dir_law,
                                   enable_n_crack_elements_escalation=False)

    # First-emission set: vertex ids that pre-existed this growth run.
    # Tips on these ids bypass the max-kink clamp (first emission); any
    # tip on a fresh id (added by extend_tip during this run, or any
    # earlier run if the caller threaded its original ids through) is
    # clamped to +/- params.max_kink_deg.
    if original_initial_vertex_ids is None:
        _init_vids = set(int(v.id) for v in net.vertices)
    else:
        _init_vids = set(int(v) for v in original_initial_vertex_ids)

    prop = CrackPropagator(cfg=cfg,evaluator=evaluator,
                           toughness=tough,direction_law=dir_law,
                           max_kink_deg=float(params.max_kink_deg),
                           initial_vertex_ids=_init_vids,
                           snapped_vertex_ids=snapped_vertex_ids)
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
                disk_radius=(float(params.disk_radius_m)
                             if params.disk_radius_m is not None else None),
            )

            # Per-cycle total stress contours (BEM baseline + crack contribution).
            # Renamed from plot_total_field's natural output so make_run_videos.py
            # picks them up under the same name in every cycle directory.
            if bool(getattr(params, "enable_per_cycle_total_contour_save", False)) and plot_params is not None:
                try:
                    from fracture_utils.Ubem.disk_crack_plotting import plot_total_field
                    tag = step_dir.name
                    plot_total_field(bem_dir, res, out_dir=step_dir, tag=tag, params=plot_params, show=False)
                    for comp in ("sxx", "syy", "sxy"):
                        src = step_dir / f"total_{comp}_{tag}.png"
                        dst = step_dir / f"brazilian_disk_{comp}_contour.png"
                        if src.exists():
                            if dst.exists():
                                dst.unlink()
                            src.rename(dst)
                except Exception as e:
                    print(f"[contour] {step_dir.name}: skipped ({e})")

        return res

    global_step = 0
    last_outer_cycle = 0
    terminate_run = False
    terminate_tag = ""
    res = None

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

    try:
        # INITIAL
        # The initial state of every outer-coupling cycle after the first
        # is identical to the final state of the previous cycle, so the
        # caller can pass plot_initial=False to suppress the redundant
        # per-step plots (DCEPlotterV4 outputs + plot_hook). The solve
        # still runs so we can record the tip-history snapshot.
        step_dir=out_dir/"cycle_00_initial"
        step_dir.mkdir(parents=True,exist_ok=True)
        res=solve_only(net,step_dir,save_inner_outputs=bool(plot_initial))
        if plot_initial and plot_hook: plot_hook("initial",res,step_dir)
        _record_tip_snapshot(net, res, phase="initial", outer_cycle=0, inner_step=0, global_step=global_step)

        for cyc in range(1,params.max_cycles+1):
            last_outer_cycle = int(cyc)
            inner_step = 0

            # Outer-cycle simplification: merge near-aligned segments at
            # deg-2 nodes, then re-detect intersections so any new
            # crossings split edges and vertex degrees are reclassified
            # before the next growth round begins.
            if bool(params.simplify_at_outer_cycle_start):
                print(f"[simplify] outer cycle {cyc}: merge aligned + snap-to-segment + intersect")
                net = simplify_and_intersect(
                    net,
                    max_angle_deg=float(params.simp_max_angle_deg),
                    max_merged_edge_m=float(params.simp_max_merged_edge_mm) * mm,
                    merge_vertex_tol_m=float(params.simp_merge_tolerance_m),
                    snap_tol_m=float(params.simp_min_edge_mm) * mm,
                    detect_mode=params.intersection_detect_mode,
                    verbose=bool(params.intersection_verbose),
                )

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
                # Enforce intersections after each inner step. Segment-merge
                # simplification is disabled here — raw grown segments are kept
                # so we can study the as-grown crack path. The snap-to-segment
                # pass also runs (vertex within simp_min_edge_mm of a non-
                # incident segment becomes a junction on that segment) since
                # it is just a generalisation of intersection detection.
                net=update_network_with_intersections(
                    net,
                    detect_mode=params.intersection_detect_mode,
                    verbose=params.intersection_verbose,
                    snap_tol_m=float(params.simp_min_edge_mm) * mm,
                )

                need_inner_metrics = bool(params.enable_inner_cycle_metrics_save)
                if need_inner_metrics:
                    _record_inner_cycle_metrics(
                        net,
                        outer_cycle=cyc,
                        inner_step=inner_step,
                        global_step=global_step,
                        reports=result.reports,
                    )
                    _persist_inner_cycle_metrics()

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

                # All-tip growth_force == 0 stop: nobody is driving growth
                # this step (every tip is snapped at the boundary, below
                # Kc, or eliminated below the delta_a tolerance). Trigger
                # the same graceful-stop path so the outer driver can
                # break its loop too.
                if bool(params.stop_on_all_growth_force_zero) and grew == 0 and len(result.reports) > 0:
                    n_snapped_now = sum(
                        1 for r in result.reports
                        if str(getattr(r, "reason", "")) == "snapped_at_boundary"
                    )
                    print(
                        "[stop] all-tip growth_force=0 "
                        f"(tips={len(result.reports)}, snapped={n_snapped_now}) "
                        f"at outer={cyc}, inner={inner_step}, global_step={global_step}."
                    )
                    terminate_run = True
                    terminate_tag = "all_growth_force_zero"
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

            step_dir=out_dir/f"cycle_{cyc:02d}"
            step_dir.mkdir(parents=True,exist_ok=True)
            res=solve_only(net,step_dir,save_inner_outputs=bool(params.enable_inner_cycle_plot_save))
            if plot_hook: plot_hook(f"cycle_{cyc:02d}",res,step_dir)
            _record_tip_snapshot(net, res, phase="outer", outer_cycle=cyc, inner_step=inner_step, global_step=global_step)

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
            setattr(res, "snapped_vertex_ids", set(prop.snapped_vertex_ids))
            setattr(res, "stopped_reason",
                    str(terminate_tag) if terminate_run else "")

    if res is None:
        raise RuntimeError("No solver result was produced during network growth run.")
    return res


# ============================================================
# Flat step-based driver (new pipeline)
# ============================================================
def run_growth_steps(
    *,
    bem_dir: Path,
    out_dir: Path,
    vertices: np.ndarray,
    connectivity: np.ndarray,
    params: CrackGrowthParams,
    max_steps: int,
    bem_correction_frequency: int = 0,
    bem_correction_hook: Optional[Callable[[Any, int], None]] = None,
    plot_hook: Optional[Callable] = None,
    plot_params: Optional[Any] = None,
    original_initial_vertex_ids: Optional[Iterable[int]] = None,
    snapped_vertex_ids: Optional[Iterable[int]] = None,
    plot_initial: bool = True,
):
    """Flat STEP-based growth driver.

    Writes ``STEP_00`` (initial pre-grow solve) and
    ``STEP_01 .. STEP_<max_steps>`` (each = one grow_one_increment + solve)
    directly under ``out_dir``. Replaces the legacy three-level
    (outer-cycle -> cycle_NN -> inner while-loop) structure.

    Parameters
    ----------
    max_steps
        Number of growth STEPS to run after STEP_00. The total folder
        count is max_steps + 1 (STEP_00 .. STEP_<max_steps>).
    bem_correction_frequency
        Fire the BEM correction every K steps. 0 disables (equivalent
        to coupling_method='no_coupling').
    bem_correction_hook : callable(res, step) -> None
        Required when bem_correction_frequency > 0. The hook is expected
        to overwrite ``bem_dir/{Sxx,Syy,Sxy}.npy`` in place (typically by
        calling ``solve_bem_with_extra_boundary_tractions(...)`` for the
        iterative path, or running a direct/full-KKT pass). The
        propagator and applied-stress interpolators are rebuilt from the
        refreshed BEM arrays after the hook returns, so the next step's
        grow uses the corrected field.
    plot_hook : callable(tag, res, step_dir) -> None
        Tags emitted are ``"initial"`` for STEP_00 and ``"step_NN"`` for
        STEP_NN.

    Termination
    -----------
    Early-stops on ``params.stop_on_all_growth_force_zero`` and
    ``params.stop_on_all_keff_below_kc`` (same semantics as the legacy
    runner). Sets ``res.stopped_reason`` accordingly.
    """
    from preamble import (
        Material, AppliedStress, CrackNetworkV4,
        DCENetworkStaticV4, DCEResultsNetworkV4,
        DCEPlotterV4, DCEPlotterDeformedV4,
        PropagationConfig, ConstantToughness, MaximumHoopStressLaw,
        CandidateEvaluator, CrackPropagator,
        _call,
    )
    from fracture_utils.Upropagation.network_ops import get_netops
    from fracture_utils.Ubem.crack_energetics import topology_metrics_for_network

    mm = 1e-3
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bem_dir = Path(bem_dir)
    max_steps = int(max_steps)
    bem_freq = int(bem_correction_frequency)
    if bem_freq > 0 and bem_correction_hook is None:
        raise ValueError(
            "bem_correction_frequency > 0 requires bem_correction_hook "
            "to overwrite the BEM arrays in bem_dir."
        )

    # Per-STEP metrics: alpha/Q/P_infty/L_total_m + per-tip KI/KII.
    step_history: list = []

    # Wall-clock start for runtime-limit checks; finalize bookkeeping.
    _t_start = time.monotonic()
    _runtime_limit = (float(params.runtime_limit_s)
                      if getattr(params, "runtime_limit_s", None) is not None
                      else None)

    def _persist_step_history_json():
        """Write step_history to out_dir/step_history.json (idempotent).

        Called after every _record_step_metrics append so an interrupted run
        leaves the metric data on disk; plot_step_metrics can then be re-run
        from disk without the live res object.
        """
        try:
            payload = {
                "step_history": list(step_history),
                "n_steps": len(step_history),
                "runtime_s": float(time.monotonic() - _t_start),
            }
            with (out_dir / "step_history.json").open("w") as fh:
                json.dump(payload, fh, indent=2, default=float)
        except Exception as _e:
            print(f"[step-history] persist failed: {_e}")

    def _encode_videos_subprocess():
        """Invoke tools/make_run_videos.py on out_dir via subprocess.

        Uses the running interpreter (sys.executable) so the conda env's
        imageio + PIL are picked up. Failures are logged and swallowed so
        finalize never aborts on the video step.
        """
        here = Path(__file__).resolve()
        script = None
        for cand in (here.parents[3] / "tools" / "make_run_videos.py",
                     here.parents[2] / "tools" / "make_run_videos.py"):
            if cand.exists():
                script = cand
                break
        if script is None:
            print("[finalize] tools/make_run_videos.py not found; skipping videos")
            return
        try:
            print(f"[finalize] encoding videos via {script.name} (fps={params.video_fps})")
            subprocess.run(
                [sys.executable, str(script), str(out_dir),
                 "--fps", str(int(params.video_fps))],
                check=False,
            )
        except Exception as _e:
            print(f"[finalize] video encode failed: {_e}")

    def _finalize_run(reason: str):
        """Persist step_history, generate metrics plots, encode videos.

        Always runs from the outer try/finally so it executes on success,
        on graceful KeyboardInterrupt, and on runtime-limit timeout. The
        nonlocal terminate_tag is set only if no earlier criterion fired.
        """
        nonlocal terminate_tag
        if not terminate_tag and reason:
            terminate_tag = reason
        _persist_step_history_json()
        if step_history:
            try:
                from fracture_utils.Ubem.disk_crack_plotting import plot_step_metrics
                plot_step_metrics(step_history, out_dir)
            except Exception as _e:
                print(f"[finalize] plot_step_metrics failed: {_e}")
        else:
            print("[finalize] step_history empty; skipping plot_step_metrics")
        if bool(getattr(params, "enable_video_encode", True)):
            _encode_videos_subprocess()

    # ----- 1) Initial network + optional simplification ----------------
    net = CrackNetworkV4.from_vertices_connectivity(
        vertices=np.asarray(vertices, float),
        connectivity=np.asarray(connectivity, int),
        Nv_max=4,
        validate=True,
    )
    if bool(params.simplify_at_outer_cycle_start):
        print("[simplify] initial pass: merge aligned + snap-to-segment + intersect")
        net = simplify_and_intersect(
            net,
            max_angle_deg=float(params.simp_max_angle_deg),
            max_merged_edge_m=float(params.simp_max_merged_edge_mm) * mm,
            merge_vertex_tol_m=float(params.simp_merge_tolerance_m),
            snap_tol_m=float(params.simp_min_edge_mm) * mm,
            detect_mode=params.intersection_detect_mode,
            verbose=bool(params.intersection_verbose),
        )

    if original_initial_vertex_ids is None:
        _init_vids = set(int(v.id) for v in net.vertices)
    else:
        _init_vids = set(int(v) for v in original_initial_vertex_ids)
    _snapped_in = set(int(v) for v in snapped_vertex_ids) if snapped_vertex_ids else set()

    # ----- 2) Per-state factories (rebuilt after each BEM correction) ---
    # The propagator captures `applied` (which captures sigma_func via the
    # RegularGridInterpolator). After a BEM correction overwrites the BEM
    # arrays, we rebuild applied + evaluator + prop so the next grow uses
    # the refreshed field.
    material = Material(
        E=params.material_E, nu=params.material_nu,
        plane_stress=params.plane_stress,
    )
    base_solver_kwargs = dict(
        n_crack_elements=100,
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
        base_solver_kwargs.update(params.solver_kwargs)
    Kc_from_kwargs = base_solver_kwargs.pop("Kc_demo", None)
    _ = base_solver_kwargs.pop("K_demo", None)
    Kc_eff = float(params.Kc_demo if Kc_from_kwargs is None else Kc_from_kwargs)
    tough = ConstantToughness(Kc_eff)
    dir_law = MaximumHoopStressLaw()
    prop_cfg = PropagationConfig(
        f0=params.f0,
        f_fixed=params.f_fixed,
        step_mode=params.step_mode,
        simultaneous_tip_growth=params.simultaneous_tip_growth,
        f_disk_radius=float(params.f_disk_radius),
        disk_radius_m=float(params.disk_radius_m) if params.disk_radius_m is not None else 0.0,
    )

    def _build_applied_from_bem():
        from scipy.interpolate import RegularGridInterpolator
        xs = np.load(bem_dir / "xs.npy")
        ys = np.load(bem_dir / "ys.npy")
        Sxx = np.nan_to_num(np.load(bem_dir / "Sxx.npy"), nan=0.0)
        Syy = np.nan_to_num(np.load(bem_dir / "Syy.npy"), nan=0.0)
        Sxy = np.nan_to_num(np.load(bem_dir / "Sxy.npy"), nan=0.0)

        def _align(S, name):
            S = np.asarray(S, float)
            if S.shape == (ys.size, xs.size):
                return S
            if S.shape == (xs.size, ys.size):
                return S.T
            raise ValueError(
                f"BEM stress array {name}.npy has shape {S.shape}; expected "
                f"(ys.size={ys.size}, xs.size={xs.size}). Stale arrays in "
                f"bem_dir={bem_dir!s}? Re-run with skip_bem_solve=False or "
                f"delete the *_npy files."
            )
        Sxx = _align(Sxx, "Sxx"); Syy = _align(Syy, "Syy"); Sxy = _align(Sxy, "Sxy")

        Ixx = RegularGridInterpolator((ys, xs), Sxx, bounds_error=False, fill_value=0.0)
        Iyy = RegularGridInterpolator((ys, xs), Syy, bounds_error=False, fill_value=0.0)
        Ixy = RegularGridInterpolator((ys, xs), Sxy, bounds_error=False, fill_value=0.0)

        def sigma_func(X):
            X = np.asarray(X, float)
            pts = np.column_stack([X[:, 1], X[:, 0]])
            S = np.zeros((len(X), 2, 2))
            S[:, 0, 0] = Ixx(pts)
            S[:, 1, 1] = Iyy(pts)
            S[:, 0, 1] = Ixy(pts)
            S[:, 1, 0] = Ixy(pts)
            return S

        aug_active = isinstance(base_solver_kwargs.get("augmented_coupling", None), dict)
        keep_bem_applied = bool(base_solver_kwargs.get("augmented_keep_bem_applied", True))
        if aug_active and (not keep_bem_applied):
            return AppliedStress(sigma_xx=0.0, sigma_yy=0.0, sigma_xy=0.0)
        return AppliedStress(sigma_func=sigma_func)

    def _build_prop(applied, snapped_vids):
        evaluator = CandidateEvaluator(
            material=material, applied=applied,
            solver_kwargs=base_solver_kwargs,
            direction_law=dir_law,
            enable_n_crack_elements_escalation=False,
        )
        prop = CrackPropagator(
            cfg=prop_cfg, evaluator=evaluator,
            toughness=tough, direction_law=dir_law,
            max_kink_deg=float(params.max_kink_deg),
            initial_vertex_ids=_init_vids,
            snapped_vertex_ids=snapped_vids,
        )
        return prop, evaluator

    applied = _build_applied_from_bem()
    prop, evaluator = _build_prop(applied, _snapped_in)
    netops = get_netops()

    # ----- 2b) topology-metrics helpers (alpha, Q, ...) ----------------
    # Build the top/bottom-arc regions used by the alignment-factor Q. We
    # only have these knobs when disk_radius_m is set; otherwise we infer
    # the disk radius from the BEM grid extent.
    _xs_grid = np.load(bem_dir / "xs.npy")
    _ys_grid = np.load(bem_dir / "ys.npy")
    R_disk = float(params.disk_radius_m) if params.disk_radius_m is not None else float(
        max(np.max(np.abs(_xs_grid)), np.max(np.abs(_ys_grid)))
    )
    A_domain = float(params.domain_area_m2) if params.domain_area_m2 is not None else float(
        np.pi * R_disk * R_disk
    )
    _band = float(max(params.boundary_band_frac, 0.0)) * R_disk
    _arc = float(params.arc_half_angle_deg)

    def _region_from_theta(theta_center_deg: float):
        lo = float(theta_center_deg - _arc)
        hi = float(theta_center_deg + _arc)
        def _f(node_id, node_data):
            pos = node_data.get("pos", None)
            if pos is None:
                return False
            x, y = float(pos[0]), float(pos[1])
            r = float(np.hypot(x, y))
            if r < (R_disk - _band):
                return False
            th = float(np.degrees(np.arctan2(y, x)))
            th = ((th + 180.0) % 360.0) - 180.0
            return (th >= lo) and (th <= hi)
        return _f

    _top_region = _region_from_theta(90.0)
    _bot_region = _region_from_theta(-90.0)

    def _record_step_metrics(step_idx, network_state, res_state):
        """Append per-step topology metrics + per-tip SIFs to step_history."""
        try:
            topo = topology_metrics_for_network(
                network_state,
                domain_area=A_domain,
                region_1=_top_region,
                region_2=_bot_region,
                loading_axis=tuple(params.loading_axis),
            )
        except Exception as _e:
            print(f"[step-metrics] STEP_{step_idx:02d}: topology failed ({_e})")
            topo = {"alpha": float("nan"), "Q": float("nan"),
                    "P_infty": float("nan"), "L_total_m": float("nan")}

        # Per-tip SIFs at this step.
        tips_out = []
        try:
            deg = netops.degree_map(network_state)
            polylines = netops.extract_open_polylines(network_state)
            tips = netops.extract_deg1_tips(network_state, polylines, deg) or []
            for tip in tips:
                vid = int(getattr(tip, "v_tip", -1))
                try:
                    tev = evaluator.eval_tip(res_state, tip)
                    KI = float(getattr(tev, "KI", float("nan")))
                    KII = float(getattr(tev, "KII", float("nan")))
                except Exception:
                    KI = float("nan"); KII = float("nan")
                tips_out.append({"tip_vid": vid, "KI": KI, "KII": KII})
        except Exception as _e:
            print(f"[step-metrics] STEP_{step_idx:02d}: tip enumeration failed ({_e})")

        step_history.append({
            "step": int(step_idx),
            "alpha": float(topo.get("alpha", float("nan"))),
            "Q": float(topo.get("Q", float("nan"))),
            "P_infty": float(topo.get("P_infty", float("nan"))),
            "L_total_m": float(topo.get("L_total_m", float("nan"))),
            "tips": tips_out,
        })
        _persist_step_history_json()

    # ----- 3) Per-step solve + save helper ------------------------------
    def _solve_step(network, step_dir, *, save_outputs: bool):
        print(f"[solve] {step_dir.name} (Nv={len(network.vertices)}, Ne={len(network.edges)})")
        calc = DCENetworkStaticV4(material, network, applied)
        sol = calc.solve(**base_solver_kwargs)
        res = DCEResultsNetworkV4(calc, sol)
        if save_outputs:
            pl = DCEPlotterV4(res, out_dir=step_dir)
            _call(pl, "plot_network_graph")
            pl_def = DCEPlotterDeformedV4(res, out_dir=step_dir)
            _call(
                pl_def, "plot_deformed_network",
                scale=float(getattr(params, "deformed_network_scale", 50.0)),
                trim_core_junction_faces=True,
                disk_radius=(float(params.disk_radius_m)
                             if params.disk_radius_m is not None else None),
            )
            if bool(getattr(params, "enable_per_cycle_total_contour_save", False)) and plot_params is not None:
                try:
                    from fracture_utils.Ubem.disk_crack_plotting import plot_total_field
                    tag = step_dir.name
                    plot_total_field(bem_dir, res, out_dir=step_dir, tag=tag, params=plot_params, show=False)
                    for comp in ("sxx", "syy", "sxy"):
                        src = step_dir / f"total_{comp}_{tag}.png"
                        dst = step_dir / f"brazilian_disk_{comp}_contour.png"
                        if src.exists():
                            if dst.exists():
                                dst.unlink()
                            src.rename(dst)
                except Exception as e:
                    print(f"[contour] {step_dir.name}: skipped ({e})")
        return res

    # ----- 4) Termination helpers --------------------------------------
    def _all_keff_below(reports):
        n = 0; finite = []
        for rep in (reports or []):
            n += 1
            KI = float(getattr(rep, "KI", np.nan))
            KII = float(getattr(rep, "KII", np.nan))
            if np.isfinite(KI) and np.isfinite(KII):
                finite.append(float(np.sqrt(KI * KI + KII * KII)))
        if n <= 0 or len(finite) != n:
            return False, n, 0, np.nan, np.nan
        arr = np.asarray(finite, float)
        return bool(np.all(arr <= Kc_eff)), n, len(finite), float(arr.min()), float(arr.max())

    # ----- 5) STEP_00 = initial solve -----------------------------------
    step_dir = out_dir / "STEP_00"
    step_dir.mkdir(parents=True, exist_ok=True)
    res = _solve_step(net, step_dir, save_outputs=bool(plot_initial))
    if plot_initial and plot_hook:
        plot_hook("initial", res, step_dir)
    _record_step_metrics(0, net, res)
    terminate_tag = ""

    # ----- 6) STEP_01..STEP_<max_steps> ---------------------------------
    try:
        for step in range(1, max_steps + 1):
            if _runtime_limit is not None:
                elapsed = float(time.monotonic() - _t_start)
                if elapsed >= _runtime_limit:
                    print(
                        f"[stop] runtime limit {_runtime_limit:.1f}s reached at "
                        f"STEP_{step:02d} (elapsed={elapsed:.1f}s); finalizing."
                    )
                    terminate_tag = "runtime_limit"
                    break
            print(f"\n[STEP] {step}/{max_steps}", flush=True)
            net_before = net
            result = prop.grow_one_increment(net_before)
            net = result.network_new
            net = update_network_with_intersections(
                net,
                detect_mode=params.intersection_detect_mode,
                verbose=params.intersection_verbose,
                snap_tol_m=float(params.simp_min_edge_mm) * mm,
            )

            step_dir = out_dir / f"STEP_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            res = _solve_step(net, step_dir, save_outputs=True)
            if plot_hook:
                plot_hook(f"step_{step:02d}", res, step_dir)
            _record_step_metrics(step, net, res)

            grew = sum(1 for r in result.reports if bool(getattr(r, "grew", False)))

            # Early-stop checks (mirror legacy semantics).
            if bool(params.stop_on_all_keff_below_kc):
                hit, n_total, n_finite, kmin, kmax = _all_keff_below(result.reports)
                if hit:
                    print(
                        f"[stop] all-tip K_eff below Kc at STEP_{step:02d} "
                        f"(tips={n_finite}/{n_total}, "
                        f"K_min={kmin*1e-6:.3e} MPa*sqrt(m), "
                        f"K_max={kmax*1e-6:.3e} MPa*sqrt(m), "
                        f"Kc={Kc_eff*1e-6:.3e} MPa*sqrt(m))."
                    )
                    terminate_tag = "all_keff_below_kc"
                    break
            if bool(params.stop_on_all_growth_force_zero) and grew == 0 and len(result.reports) > 0:
                n_snapped = sum(
                    1 for r in result.reports
                    if str(getattr(r, "reason", "")) == "snapped_at_boundary"
                )
                print(
                    f"[stop] all-tip growth_force=0 at STEP_{step:02d} "
                    f"(tips={len(result.reports)}, snapped={n_snapped})."
                )
                terminate_tag = "all_growth_force_zero"
                break

            # ----- BEM correction every K steps -----
            if bem_freq > 0 and (step % bem_freq == 0) and step < max_steps:
                print(f"[BEM-correction] STEP_{step:02d}: firing hook", flush=True)
                bem_correction_hook(res, step)
                # Rebuild prop with the refreshed BEM field. Carry over the
                # propagator's mutated snapped_vertex_ids so the next grow
                # keeps the same boundary-snapping state.
                applied = _build_applied_from_bem()
                prop, evaluator = _build_prop(applied, set(prop.snapped_vertex_ids))
    except KeyboardInterrupt:
        # Graceful interrupt: do NOT re-raise. The finally block below
        # runs the full finalize path (step_history.json + metrics plots
        # + videos) so a Ctrl+C mid-run leaves a complete recoverable
        # artifact bundle on disk.
        print("[interrupt] KeyboardInterrupt received; finalizing gracefully.")
        terminate_tag = "keyboard_interrupt"
    finally:
        _finalize_run(reason=terminate_tag or "completed")
        if res is not None:
            setattr(res, "snapped_vertex_ids", set(prop.snapped_vertex_ids))
            setattr(res, "stopped_reason", str(terminate_tag))
            setattr(res, "step_history", list(step_history))

    return res
