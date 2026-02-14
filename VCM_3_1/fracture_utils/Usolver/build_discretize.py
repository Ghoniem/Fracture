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

from typing import List, Tuple, Dict
import math
import numpy as np

from .network import CrackNetworkV4
from .material import Material, AppliedStress
from .solve_kernels import stress_edge_dislocation
from .polyline import polyline_point_and_frame

from .build_curved import arc_point_and_frame, cspline_point_and_frame
from .build_geometry import vertex_degrees

from .build_mesh import (
    identify_refinement_segments,
    allocate_panels_per_segment,
    build_s_nodes_for_polyline,
)


def discretize_polylines(
    network: CrackNetworkV4,
    polylines: List[dict],
    *,
    ne_half: int,
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

    for pid, p in enumerate(polylines):
        kind = str(p.get("kind", "polyline")).lower().strip()
        vids_path = p.get("path_vertex_ids", [])
        segL = np.array(p.get("segment_lengths", []), float)
        L = float(p.get("total_length", 0.0))

        # Total panels for this polyline (legacy scaling)
        Np = max(8, int(2 * ne_half))

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
            panel_wsing=np.array(panel_wsing, float),
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
                        sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
                        tx = float(sxx * n_i[0] + sxy * n_i[1])
                        ty = float(sxy * n_i[0] + syy * n_i[1])
                        tn_sum_I += float(n_i[0] * tx + n_i[1] * ty)
                        ts_sum_I += float(t_i[0] * tx + t_i[1] * ty)

                        dB2 = ts * float(ww)
                        sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu)
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
