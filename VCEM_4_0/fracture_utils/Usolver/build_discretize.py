"""
Discretization + operator assembly for polyline-based crack representations.

This module is shared between HALF and FULL crack modes:
- discretize_polylines(): builds panel geometry, collocation points, quadrature sources
- allocate_unknowns(): density and (optional) junction jump DOFs
- assemble_operator(): builds linear operator K and rhs

Geometry evaluation:
- straight polylines -> polyline_point_and_frame (from .polyline)
- arcs -> arc_point_and_frame (from .build_curved)
- csplines -> cspline_point_and_frame (from .build_curved)
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Dict
import math
import numpy as np

from .network import CrackNetworkV4
from .material import Material, AppliedStress
from .solve_kernels import stress_edge_dislocation, edge_dislocation_u
from .polyline import polyline_point_and_frame

from .build_curved import arc_point_and_frame, cspline_point_and_frame
from .build_geometry import vertex_degrees
from fracture_utils.Ubem.bem_solver import (
    Segment,
    gauss_legendre,
    map_to_segment,
    kelvin_dU_dfield,
    kelvin_dT_dfield,
    shear_modulus,
    lame_lambda,
)

from .build_mesh import (
    identify_refinement_segments,
    allocate_panels_per_segment,
    build_s_nodes_for_polyline,
)


def _merge_short_panels(s_nodes: np.ndarray, min_length: float) -> Tuple[np.ndarray, int]:
    """Greedy in-place merge of consecutive s_nodes whose gap < min_length.

    Repeatedly removes the shorter of the two neighbours of the smallest panel
    (preserving the polyline endpoints s_nodes[0] and s_nodes[-1]) until every
    surviving panel has length >= min_length. Returns the updated s_nodes and
    the number of merges performed.

    The system matrix in the KKT crack solve becomes severely ill-conditioned
    when panel lengths span many decades -- tip-clustering or kink-refinement
    can produce panels orders of magnitude smaller than their peers, and the
    K^T K normal-equation matrix in solve_kkt_lsq_eq inherits the *square* of
    that conditioning. The threshold is typically set as a small fraction of
    the polyline's total length L (e.g. min_length = 1e-3 * L).
    """
    if s_nodes.size < 3 or min_length <= 0.0:
        return s_nodes, 0
    s = list(map(float, s_nodes))
    n_merged = 0
    while True:
        ds = np.diff(np.asarray(s))
        if ds.size == 0:
            break
        k_short = int(np.argmin(ds))
        if float(ds[k_short]) >= min_length:
            break
        # Pick which boundary node of the offending panel to remove. Endpoints
        # s[0] and s[-1] are immovable (they pin the polyline to its vertices).
        if k_short == 0:
            del s[1]                       # absorb first panel into the next
        elif k_short == len(ds) - 1:
            del s[-2]                      # absorb last panel into the previous
        else:
            # Merge with the smaller neighbour (smallest local mesh change).
            if ds[k_short - 1] <= ds[k_short + 1]:
                del s[k_short]             # merge into left neighbour
            else:
                del s[k_short + 1]         # merge into right neighbour
        n_merged += 1
        if len(s) < 3:                     # only one panel left, can't shrink further
            break
    return np.asarray(s, float), n_merged


def discretize_polylines(
    network: CrackNetworkV4,
    polylines: List[dict],
    *,
    n_crack_elements: int,
    node_distribution: str,
    collocation_mode: str,
    representation: str,
    nq_stress: int,
    tip_min_nodes: int = 0,
    endpoint_min_nodes: int = 8,
    kink_side_nodes: int = 4,
    refine_junction_endpoints: bool = True,
    refine_kinks: bool = True,
    tip_cluster: str = "power",
    tip_cluster_power: float = 2.0,
    other_min_panels: int = 1,
    min_panel_length_ratio: float = 1.0e-3,
) -> List[dict]:
    """
    Discretize polyline records into panel midpoints, collocation points, and quadrature sources.

    v2.0 refinement knobs:
    - endpoint_min_nodes: minimum nodes clustered near deg=1 tips and deg>=3 junction endpoints
    - kink_side_nodes: additional nodes on either side of deg=2 kink vertices (within a polyline path)
    - refine_junction_endpoints/refine_kinks: toggles
    """
    use_tip_singular = (str(representation).lower().strip() == "singular")

    deg = vertex_degrees(network)

    tip_min_nodes = max(0, int(tip_min_nodes))
    tip_min_panels = max(0, tip_min_nodes - 1)

    # --- v2.0 refinement knobs (branched crack validation)
    endpoint_min_nodes = int(max(0, endpoint_min_nodes))
    kink_side_nodes = int(max(0, kink_side_nodes))
    refine_junction_endpoints = bool(refine_junction_endpoints)
    refine_kinks = bool(refine_kinks)

    # Backward compatibility: if user still passes tip_min_nodes, honor it as a lower bound
    endpoint_min_nodes = max(endpoint_min_nodes, tip_min_nodes if int(tip_min_nodes) > 0 else 0)

    endpoint_min_panels = max(0, endpoint_min_nodes - 1)  # nodes -> panels on endpoint-adjacent segment
    kink_min_panels = max(0, kink_side_nodes)             # panels on each side adjacent to a kink (deg=2)

    other_min_panels = max(0, int(other_min_panels))
    tip_cluster = str(tip_cluster).lower().strip()
    tip_cluster_power = float(tip_cluster_power)
    if not np.isfinite(tip_cluster_power) or tip_cluster_power <= 1.0:
        tip_cluster_power = 2.0

    poly_panels: List[dict] = []

    # Length-aware panel budget: rather than give every polyline the same
    # 2*n_crack_elements panels (which over-resolves short branches and
    # bloats the KKT system), scale by L / L_max so the *target* per-panel
    # length is roughly the same across polylines. Long polylines still get
    # the full budget, short ones drop to MIN_PANELS_PER_POLYLINE.
    MIN_PANELS_PER_POLYLINE = 8
    NP_MAX = max(MIN_PANELS_PER_POLYLINE, int(2 * n_crack_elements))
    _all_L = [float(p.get("total_length", 0.0)) for p in polylines]
    _L_max = max((L for L in _all_L if L > 0.0), default=0.0)

    for pid, p in enumerate(polylines):
        kind = str(p.get("kind", "polyline")).lower().strip()
        vids_path = p.get("path_vertex_ids", [])
        segL = np.array(p.get("segment_lengths", []), float)
        L = float(p.get("total_length", 0.0))

        # Total panels for this polyline -- length-aware.
        # Np_i = NP_MAX * L_i / L_max, clipped to [MIN_PANELS_PER_POLYLINE, NP_MAX].
        if _L_max > 0.0:
            Np = max(MIN_PANELS_PER_POLYLINE,
                     min(NP_MAX, int(NP_MAX * L / _L_max + 0.5)))
        else:
            Np = MIN_PANELS_PER_POLYLINE

        # Single-segment curved edges (arc/cspline) are treated as one segment of length L.
        nseg = int(len(segL)) if (len(segL) > 0) else 1
        if nseg <= 0 or L <= 0.0:
            continue

        v_start = int(p.get("v_start", vids_path[0] if len(vids_path) else -1))
        v_end = int(p.get("v_end", vids_path[-1] if len(vids_path) else -1))
        # ----------------------------
        # Mesh / node distribution (refinement budget)
        # ----------------------------
        # segment lengths for curved single-segment cases
        if len(segL) == 0:
            segL_eff = np.array([L], float)
        else:
            segL_eff = segL

        # Identify refined endpoint segments + kink-adjacent segments
        refined_endpoint_segs, kink_adj_segs, cluster_target = identify_refinement_segments(
            vids_path=list(vids_path),
            deg=deg,
            nseg=nseg,
            refine_junction_endpoints=refine_junction_endpoints,
            refine_kinks=bool(refine_kinks and (kind == "polyline")),
        )

        # Backward compatibility: honor explicit tip_min_panels by raising endpoint minimum
        endpoint_min_panels_eff = int(max(endpoint_min_panels, tip_min_panels))

        # Allocate panels per segment (sum to Np)
        Nseg = allocate_panels_per_segment(
            Np=int(Np),
            nseg=int(nseg),
            refined_endpoint_segs=refined_endpoint_segs,
            kink_adj_segs=kink_adj_segs,
            endpoint_min_panels=int(endpoint_min_panels_eff),
            kink_min_panels=int(kink_min_panels),
            other_min_panels=int(other_min_panels),
        )

        # Build s_nodes by concatenating per-segment node sets.
        # NOTE: supports two-sided clustering when nseg==1 and both ends are refined (symmetry fix).
        s_nodes = build_s_nodes_for_polyline(
            segL=np.asarray(segL_eff, float),
            Nseg=list(Nseg),
            refined_endpoint_segs=refined_endpoint_segs,
            kink_adj_segs=kink_adj_segs,
            cluster_target=cluster_target,
            tip_cluster=tip_cluster,
            tip_cluster_power=tip_cluster_power,
        )

        # Ill-conditioning guard: merge any panels shorter than
        # min_panel_length_ratio * L. KKT solve_kkt_lsq_eq forms K^T K so a
        # large length-ratio between panels squares into the condition number;
        # tip-clustering can easily produce panels 4-6 decades smaller than
        # their peers.
        if min_panel_length_ratio > 0.0 and L > 0.0:
            min_length = float(min_panel_length_ratio) * float(L)
            s_nodes, n_merged = _merge_short_panels(s_nodes, min_length)
            if n_merged > 0:
                Np_new = int(s_nodes.size - 1)
                warnings.warn(
                    f"Polyline pid={pid}: merged {n_merged} sub-threshold panels "
                    f"(min_length = {min_panel_length_ratio:.0e} * L = {min_length:.3e}); "
                    f"Np {Np} -> {Np_new}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                Np = Np_new

        if collocation_mode == "nodes" and Np >= 2:
            s_col = s_nodes[1:-1].copy()
        else:
            s_col = 0.5 * (s_nodes[:-1] + s_nodes[1:])

        s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
        ds = (s_nodes[1:] - s_nodes[:-1])

        # Point and frame evaluation
        def eval_point_frame(ss: float):
            if kind == "arc":
                return arc_point_and_frame(p, ss)
            if kind == "cspline":
                return cspline_point_and_frame(p, ss)
            return polyline_point_and_frame(network, vids_path, segL_eff, ss)

        x_mid = []
        t_mid = []
        n_mid = []
        for sm in s_mid:
            x, t, n = eval_point_frame(float(sm))
            x_mid.append(x)
            t_mid.append(t)
            n_mid.append(n)
        x_mid = np.array(x_mid, float)
        t_mid = np.array(t_mid, float)
        n_mid = np.array(n_mid, float)

        x_col = []
        t_col = []
        n_col = []
        for sc in s_col:
            x, t, n = eval_point_frame(float(sc))
            x_col.append(x)
            t_col.append(t)
            n_col.append(n)
        x_col = np.array(x_col, float)
        t_col = np.array(t_col, float)
        n_col = np.array(n_col, float)

        nq = max(2, int(nq_stress))
        xg, wg = np.polynomial.legendre.leggauss(nq)

        src_pts = []
        src_t = []
        src_n = []
        src_w = []
        panel_src = []
        panel_wsing = []
        eps_s = 1e-14 * L if L > 0 else 1e-14

        idx0 = 0
        for k in range(Np):
            sLk = float(s_nodes[k])
            sRk = float(s_nodes[k + 1])
            Jk = 0.5 * (sRk - sLk)
            smk = 0.5 * (sRk + sLk)
            sq = smk + Jk * xg
            wq = Jk * wg

            wsing_k = 0.0
            for sqq, wqq in zip(sq, wq):
                x, t, n = eval_point_frame(float(sqq))
                src_pts.append(x)
                src_t.append(t)
                src_n.append(n)

                if use_tip_singular:
                    if str(p.get("crack_mode", "full")).lower().strip() == "full":
                        ss0 = max(float(sqq), eps_s)
                        tt0 = max(float(L - float(sqq)), eps_s)
                        sing = 1.0 / math.sqrt(ss0 * tt0)
                    else:
                        tt0 = max(float(L - float(sqq)), eps_s)
                        sing = 1.0 / math.sqrt(tt0)
                else:
                    sing = 1.0

                w_eff = float(wqq) * sing
                src_w.append(w_eff)
                wsing_k += w_eff

            panel_wsing.append(wsing_k)
            panel_src.append((idx0, idx0 + nq))
            idx0 += nq

        # Warn when the singular tip representation is asked for but the crack
        # is so short or coarsely panelled that the quadrature weights span a
        # huge dynamic range -- a known recipe for ill-conditioning. We use the
        # max/median weight ratio across panels; thresholds chosen so smooth
        # singular runs stay quiet and clearly degenerate ones surface.
        wsing_arr = np.array(panel_wsing, float)
        if use_tip_singular and wsing_arr.size >= 2:
            med = float(np.median(wsing_arr))
            mx = float(np.max(wsing_arr))
            ratio = (mx / med) if med > 0.0 else float("inf")
            if ratio > 1.0e3:
                warnings.warn(
                    "Singular tip representation on polyline pid="
                    f"{pid} produces extreme quadrature-weight contrast "
                    f"(max/median = {ratio:.2e}, L = {L:.3e}, Np = {Np}). "
                    "This usually means the crack is too short or too coarsely "
                    "panelled for the singular representation; consider "
                    "representation='regular' or larger n_crack_elements.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        poly_panels.append(dict(
            pid=int(pid),
            Np=int(Np),
            kind=kind,
            s_nodes=np.array(s_nodes, float),
            s_mid=np.array(s_mid, float),
            ds=np.array(ds, float),
            x_mid=x_mid,
            t_mid=t_mid,
            n_mid=n_mid,
            x_col=x_col,
            t_col=t_col,
            n_col=n_col,
            src_pts=np.array(src_pts, float),
            src_t=np.array(src_t, float),
            src_n=np.array(src_n, float),
            src_w=np.array(src_w, float),
            panel_src=panel_src,
            panel_wsing=wsing_arr,
            v_start=int(v_start),
            v_end=int(v_end),
            L=float(L),
        ))

    return poly_panels


def allocate_unknowns(
    poly_panels: List[dict],
    deg: dict[int, int],
    *,
    crack_mode: str,
    junction_model: str = "strict",
) -> Tuple[List[int], Dict[int, int], Dict[Tuple[int, str], int], int]:
    """
    Unknown blocks:
      1) Panel densities (always): 2*Np per polyline, contiguous per pid
      2) Jump DOFs at junctions (half mode only):
         - junction_model='strict': shared per vertex with degree>=2 (legacy Option A)
         - junction_model in {'core','soft'}: per-branch-end DOFs at endpoints with degree>=2
    """
    offsets: List[int] = []
    nunk = 0

    for pp in poly_panels:
        offsets.append(int(nunk))
        nunk += 2 * int(pp["Np"])

    junction_model = str(junction_model).lower().strip()
    if junction_model not in ("strict", "core", "soft"):
        junction_model = "strict"

    junction_dof: Dict[int, int] = {}
    branch_end_dof: Dict[Tuple[int, str], int] = {}

    if str(crack_mode).lower().strip() == "half":
        if junction_model == "strict":
            endpoint_verts = set()
            for pp in poly_panels:
                if int(deg.get(int(pp["v_start"]), 0)) >= 2:
                    endpoint_verts.add(int(pp["v_start"]))
                if int(deg.get(int(pp["v_end"]), 0)) >= 2:
                    endpoint_verts.add(int(pp["v_end"]))
            endpoint_verts = sorted(endpoint_verts)
            for i, vj in enumerate(endpoint_verts):
                junction_dof[int(vj)] = int(nunk + 2 * i)
            nunk += 2 * len(endpoint_verts)
        else:
            keys = []
            for pid, pp in enumerate(poly_panels):
                vs = int(pp["v_start"])
                ve = int(pp["v_end"])
                if int(deg.get(vs, 0)) >= 2:
                    keys.append((int(pid), "start"))
                if int(deg.get(ve, 0)) >= 2:
                    keys.append((int(pid), "end"))
            keys = sorted(set(keys))
            for i, k in enumerate(keys):
                branch_end_dof[k] = int(nunk + 2 * i)
            nunk += 2 * len(keys)

    return offsets, junction_dof, branch_end_dof, int(nunk)


def assemble_operator(
    *,
    material: Material,
    applied: AppliedStress,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    nq_col: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ncol_tot = sum(int(len(pp["x_col"])) for pp in poly_panels)
    K = np.zeros((2 * ncol_tot, nunk), float)
    rhs = np.zeros((2 * ncol_tot,), float)

    E = float(material.E)
    nu = float(material.nu)
    ps = bool(getattr(material, "plane_stress", False))
    mu = E / (2.0 * (1.0 + nu))
    # Applied stress can be constant or spatially varying.
    # Use applied.tensor_at(X) so uncoupled BEM fields are supported.
    row0 = 0
    for pid_i, pp_i in enumerate(poly_panels):
        Nci = int(len(pp_i["x_col"]))
        for ic in range(Nci):
            x_i = pp_i["x_col"][ic]
            t_i = pp_i["t_col"][ic]
            n_i = pp_i["n_col"][ic]
            sig_i = applied.tensor_at(np.asarray(x_i, float).reshape(1, 2))[0]
            tr = sig_i @ n_i.reshape(2,)
            tn0 = float(np.dot(n_i, tr))
            ts0 = float(np.dot(t_i, tr))
            rhs[row0] = -tn0
            rhs[ncol_tot + row0] = -ts0

            for pid_j, pp_j in enumerate(poly_panels):
                offj = offsets[pid_j]
                Npj = int(pp_j["Np"])
                for k in range(Npj):
                    s0, s1 = pp_j["panel_src"][k]
                    src_pts = pp_j["src_pts"][s0:s1]
                    src_t = pp_j["src_t"][s0:s1]
                    src_n = pp_j["src_n"][s0:s1]
                    wq = pp_j["src_w"][s0:s1]

                    tn_sum_I = 0.0
                    ts_sum_I = 0.0
                    tn_sum_II = 0.0
                    ts_sum_II = 0.0

                    for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                        dx = float(x_i[0] - xs[0])
                        dy = float(x_i[1] - xs[1])

                        dB = ns * float(ww)
                        sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu, plane_stress=ps)
                        tx = float(sxx * n_i[0] + sxy * n_i[1])
                        ty = float(sxy * n_i[0] + syy * n_i[1])
                        tn_sum_I += float(n_i[0] * tx + n_i[1] * ty)
                        ts_sum_I += float(t_i[0] * tx + t_i[1] * ty)

                        dB2 = ts * float(ww)
                        sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu, plane_stress=ps)
                        tx = float(sxx * n_i[0] + sxy * n_i[1])
                        ty = float(sxy * n_i[0] + syy * n_i[1])
                        tn_sum_II += float(n_i[0] * tx + n_i[1] * ty)
                        ts_sum_II += float(t_i[0] * tx + t_i[1] * ty)

                    col_I = offj + 2 * k + 0
                    col_II = offj + 2 * k + 1
                    K[row0, col_I] = tn_sum_I
                    K[ncol_tot + row0, col_I] = ts_sum_I
                    K[row0, col_II] = tn_sum_II
                    K[ncol_tot + row0, col_II] = ts_sum_II

            row0 += 1

    return K, rhs


def assemble_boundary_traction_operator(
    *,
    material: Material,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    boundary_xy: np.ndarray,
    boundary_n: np.ndarray,
) -> np.ndarray:
    """
    Assemble M such that crack-induced boundary traction vector satisfies:

        t_cr = M @ q

    where q is the solver unknown vector, and t_cr packs [tx0, ty0, tx1, ty1, ...].
    Junction jump DOFs (if present in q) have zero columns in M.
    """
    Xb = np.asarray(boundary_xy, float)
    Nb = np.asarray(boundary_n, float)
    if Xb.ndim != 2 or Xb.shape[1] != 2:
        raise ValueError("boundary_xy must have shape (Nb, 2).")
    if Nb.shape != Xb.shape:
        raise ValueError("boundary_n must have shape (Nb, 2), matching boundary_xy.")

    nb = int(Xb.shape[0])
    M = np.zeros((2 * nb, int(nunk)), float)

    E = float(material.E)
    nu = float(material.nu)
    ps = bool(getattr(material, "plane_stress", False))
    mu = E / (2.0 * (1.0 + nu))

    for ib in range(nb):
        x = float(Xb[ib, 0])
        y = float(Xb[ib, 1])
        nx = float(Nb[ib, 0])
        ny = float(Nb[ib, 1])

        row_tx = 2 * ib + 0
        row_ty = 2 * ib + 1

        for pid_j, pp_j in enumerate(poly_panels):
            offj = int(offsets[pid_j])
            Npj = int(pp_j["Np"])

            for k in range(Npj):
                s0, s1 = pp_j["panel_src"][k]
                src_pts = pp_j["src_pts"][s0:s1]
                src_t = pp_j["src_t"][s0:s1]
                src_n = pp_j["src_n"][s0:s1]
                wq = pp_j["src_w"][s0:s1]

                tx_I = 0.0
                ty_I = 0.0
                tx_II = 0.0
                ty_II = 0.0

                for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                    dx = x - float(xs[0])
                    dy = y - float(xs[1])
                    w = float(ww)

                    # Mode I basis contribution (B along local normal)
                    dB = ns * w
                    sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu, plane_stress=ps)
                    tx_I += float(sxx * nx + sxy * ny)
                    ty_I += float(sxy * nx + syy * ny)

                    # Mode II basis contribution (B along local tangent)
                    dB2 = ts * w
                    sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu, plane_stress=ps)
                    tx_II += float(sxx * nx + sxy * ny)
                    ty_II += float(sxy * nx + syy * ny)

                col_I = offj + 2 * k + 0
                col_II = offj + 2 * k + 1
                M[row_tx, col_I] = tx_I
                M[row_ty, col_I] = ty_I
                M[row_tx, col_II] = tx_II
                M[row_ty, col_II] = ty_II

    return M


def assemble_boundary_displacement_operator(
    *,
    material: Material,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    boundary_xy: np.ndarray,
) -> np.ndarray:
    """
    Assemble Mu such that crack-induced boundary displacement vector satisfies:

        u_cr = Mu @ q

    where u_cr packs [ux0, uy0, ux1, uy1, ...].
    Junction jump DOFs (if present in q) have zero columns in Mu.
    """
    Xb = np.asarray(boundary_xy, float)
    if Xb.ndim != 2 or Xb.shape[1] != 2:
        raise ValueError("boundary_xy must have shape (Nb, 2).")

    nb = int(Xb.shape[0])
    Mu = np.zeros((2 * nb, int(nunk)), float)
    nu = float(material.nu)
    ps = bool(getattr(material, "plane_stress", False))

    for ib in range(nb):
        x = float(Xb[ib, 0])
        y = float(Xb[ib, 1])
        row_ux = 2 * ib + 0
        row_uy = 2 * ib + 1

        for pid_j, pp_j in enumerate(poly_panels):
            offj = int(offsets[pid_j])
            Npj = int(pp_j["Np"])

            for k in range(Npj):
                s0, s1 = pp_j["panel_src"][k]
                src_pts = pp_j["src_pts"][s0:s1]
                src_t = pp_j["src_t"][s0:s1]
                src_n = pp_j["src_n"][s0:s1]
                wq = pp_j["src_w"][s0:s1]

                ux_I = 0.0
                uy_I = 0.0
                ux_II = 0.0
                uy_II = 0.0

                for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                    dx = x - float(xs[0])
                    dy = y - float(xs[1])
                    w = float(ww)

                    # Mode I basis contribution (B along local normal)
                    dB = ns * w
                    ux, uy = edge_dislocation_u(dx, dy, float(dB[0]), float(dB[1]), nu, plane_stress=ps)
                    ux_I += float(ux)
                    uy_I += float(uy)

                    # Mode II basis contribution (B along local tangent)
                    dB2 = ts * w
                    ux, uy = edge_dislocation_u(dx, dy, float(dB2[0]), float(dB2[1]), nu, plane_stress=ps)
                    ux_II += float(ux)
                    uy_II += float(uy)

                col_I = offj + 2 * k + 0
                col_II = offj + 2 * k + 1
                Mu[row_ux, col_I] = ux_I
                Mu[row_uy, col_I] = uy_I
                Mu[row_ux, col_II] = ux_II
                Mu[row_uy, col_II] = uy_II

    return Mu


def assemble_bem_boundary_to_crack_traction_operator(
    *,
    material: Material,
    poly_panels: List[dict],
    boundary_x1: np.ndarray,
    boundary_y1: np.ndarray,
    boundary_x2: np.ndarray,
    boundary_y2: np.ndarray,
    gauss_n: int = 4,
) -> np.ndarray:
    """
    Assemble N such that boundary unknown vector y maps to traction on crack
    collocation rows used by assemble_operator:

        t_col = N @ y

    with y = [u_bc(2Nb), t_bc(2Nb)].
    Output shape: (2*ncol_tot, 4*Nb), matching rows of K/rhs.
    """
    x1 = np.asarray(boundary_x1, float).reshape(-1)
    y1 = np.asarray(boundary_y1, float).reshape(-1)
    x2 = np.asarray(boundary_x2, float).reshape(-1)
    y2 = np.asarray(boundary_y2, float).reshape(-1)
    if not (x1.size == y1.size == x2.size == y2.size):
        raise ValueError("boundary endpoint arrays must have equal lengths.")
    nb = int(x1.size)
    if nb <= 0:
        return np.zeros((0, 0), float)

    segs: List[Segment] = []
    for i in range(nb):
        segs.append(
            Segment(
                x1=float(x1[i]),
                y1=float(y1[i]),
                x2=float(x2[i]),
                y2=float(y2[i]),
                is_traction=True,
                bc_x=0.0,
                bc_y=0.0,
            )
        )

    ncol_tot = int(sum(int(len(pp["x_col"])) for pp in poly_panels))
    Nop = np.zeros((2 * ncol_tot, 4 * nb), float)

    E = float(material.E)
    nu = float(material.nu)
    plane_strain = not bool(getattr(material, "plane_stress", False))
    mu = shear_modulus(E, nu)
    lam = lame_lambda(E, nu, plane_strain=plane_strain)
    xg, wg = gauss_legendre(int(max(2, gauss_n)))

    row0 = 0
    for pp in poly_panels:
        Nc = int(len(pp["x_col"]))
        for ic in range(Nc):
            xi = np.asarray(pp["x_col"][ic], float).reshape(2)
            ti = np.asarray(pp["t_col"][ic], float).reshape(2)
            ni = np.asarray(pp["n_col"][ic], float).reshape(2)

            row_tn = row0
            row_ts = ncol_tot + row0

            for j, sj in enumerate(segs):
                c_ux = 0.0
                c_uy = 0.0
                c_tx = 0.0
                c_ty = 0.0
                c2_ux = 0.0
                c2_uy = 0.0
                c2_tx = 0.0
                c2_ty = 0.0

                rf = 1e-16 * max(float(sj.length), 1e-12)

                for s, w in zip(xg, wg):
                    xq, yq, jac = map_to_segment(sj, float(s))
                    ww = float(w) * float(jac)

                    dUdx, dUdy = kelvin_dU_dfield(
                        field=(float(xi[0]), float(xi[1])),
                        source=(float(xq), float(yq)),
                        E=E,
                        nu=nu,
                        plane_strain=plane_strain,
                        r_floor=rf,
                    )
                    dTdx, dTdy = kelvin_dT_dfield(
                        field=(float(xi[0]), float(xi[1])),
                        source=(float(xq), float(yq)),
                        n_source=(float(sj.nx), float(sj.ny)),
                        E=E,
                        nu=nu,
                        plane_strain=plane_strain,
                        r_floor=rf,
                    )

                    # Coefficients for boundary displacement dofs (ux, uy):
                    # dudx = -dTdx @ u, dudy = -dTdy @ u
                    for k in range(2):
                        dux_dx = -float(dTdx[0, k]) * ww
                        duy_dy = -float(dTdy[1, k]) * ww
                        dux_dy = -float(dTdy[0, k]) * ww
                        duy_dx = -float(dTdx[1, k]) * ww

                        exx = dux_dx
                        eyy = duy_dy
                        exy = 0.5 * (dux_dy + duy_dx)
                        tr = exx + eyy
                        sxx = lam * tr + 2.0 * mu * exx
                        syy = lam * tr + 2.0 * mu * eyy
                        sxy = 2.0 * mu * exy

                        txv = sxx * float(ni[0]) + sxy * float(ni[1])
                        tyv = sxy * float(ni[0]) + syy * float(ni[1])
                        tn = float(ni[0]) * txv + float(ni[1]) * tyv
                        tsv = float(ti[0]) * txv + float(ti[1]) * tyv
                        if k == 0:
                            c_ux += tn
                            c2_ux += tsv
                        else:
                            c_uy += tn
                            c2_uy += tsv

                    # Coefficients for boundary traction dofs (tx, ty):
                    # dudx = dUdx @ t, dudy = dUdy @ t
                    for k in range(2):
                        dux_dx = float(dUdx[0, k]) * ww
                        duy_dy = float(dUdy[1, k]) * ww
                        dux_dy = float(dUdy[0, k]) * ww
                        duy_dx = float(dUdx[1, k]) * ww

                        exx = dux_dx
                        eyy = duy_dy
                        exy = 0.5 * (dux_dy + duy_dx)
                        tr = exx + eyy
                        sxx = lam * tr + 2.0 * mu * exx
                        syy = lam * tr + 2.0 * mu * eyy
                        sxy = 2.0 * mu * exy

                        txv = sxx * float(ni[0]) + sxy * float(ni[1])
                        tyv = sxy * float(ni[0]) + syy * float(ni[1])
                        tn = float(ni[0]) * txv + float(ni[1]) * tyv
                        tsv = float(ti[0]) * txv + float(ti[1]) * tyv
                        if k == 0:
                            c_tx += tn
                            c2_tx += tsv
                        else:
                            c_ty += tn
                            c2_ty += tsv

                cux = 2 * j + 0
                cuy = 2 * j + 1
                ctx = 2 * nb + 2 * j + 0
                cty = 2 * nb + 2 * j + 1

                Nop[row_tn, cux] = c_ux
                Nop[row_tn, cuy] = c_uy
                Nop[row_tn, ctx] = c_tx
                Nop[row_tn, cty] = c_ty

                Nop[row_ts, cux] = c2_ux
                Nop[row_ts, cuy] = c2_uy
                Nop[row_ts, ctx] = c2_tx
                Nop[row_ts, cty] = c2_ty

            row0 += 1

    return Nop
