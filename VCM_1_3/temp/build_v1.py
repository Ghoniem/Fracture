"""
Helper module: polyline construction, discretization, operator assembly for solver_parametrized_half.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import math

from .network import CrackNetworkV4
from .material import Material, AppliedStress
from .solve_kernels import stress_edge_dislocation
from .polyline import *

def _edge_kind(e) -> str:
    return str(getattr(e, 'kind', 'line')).lower().strip()

def _edge_ctrl_vids(e):
    return tuple(getattr(e, 'ctrl_vids', ()))


def vertex_degrees(network: CrackNetworkV4) -> dict[int, int]:
    deg = {int(v.id): 0 for v in network.vertices}
    for e in network.edges:
        deg[int(e.v0)] = deg.get(int(e.v0), 0) + 1
        deg[int(e.v1)] = deg.get(int(e.v1), 0) + 1
    return deg


def build_polylines_full(network: CrackNetworkV4) -> Tuple[List[dict], dict[int, int]]:

    comps = find_polyline_components(network)
    if not comps:
        raise ValueError("No edges in network.")

    # Backward compatible: if every edge is a straight 'line', use the legacy polyline-component logic.
    if not all(_edge_kind(e) == "line" for e in network.edges):
        # Mixed or curved edges: build one polyline per edge (no multi-edge chaining yet).
        polylines: List[dict] = []
        edge_to_polyline: dict[int, int] = {}
        for pid, e in enumerate(network.edges):
            k = _edge_kind(e)
            v0 = int(e.v0); v1 = int(e.v1)
            p0 = network.vertex_coords(v0)
            p1 = network.vertex_coords(v1)
            eid = int(getattr(e, "id", pid))
            if k == "arc":
                ctrls = _edge_ctrl_vids(e)
                if len(ctrls) != 1:
                    raise ValueError(f"Arc edge {eid} must have exactly one ctrl_vid (center).")
                c = network.vertex_coords(int(ctrls[0]))
                polylines.append(make_arc_polyline(p0, p1, c, crack_mode="full", v_start=v0, v_end=v1))
            else:
                seg = p1 - p0
                segL = float(np.sqrt(np.sum(seg * seg)))
                polylines.append(dict(
                    kind="polyline",
                    path_vertex_ids=[v0, v1],
                    path_edge_indices=[eid],
                    segment_lengths=[segL],
                    total_length=segL,
                    crack_mode="full",
                    v_start=v0,
                    v_end=v1,
                ))
            edge_to_polyline[eid] = int(pid)
        return polylines, edge_to_polyline

    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}

    for pid, comp in enumerate(comps):
        vids_path, eidx_path = path_order_for_component(network, comp["edges"])
        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))

        for eidx in eidx_path:
            edge_to_polyline[int(eidx)] = int(pid)

        polylines.append(dict(
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(i) for i in eidx_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="full",
            v_start=int(vids_path[0]),
            v_end=int(vids_path[-1]),
        ))
    return polylines, edge_to_polyline


def build_polylines_half_branches(
    network: CrackNetworkV4,
    deg: dict[int, int],
) -> Tuple[List[dict], dict[int, int]]:
    """Build maximal half-crack 'branches' as polylines that may span multiple edges.

    A polyline (branch) is a maximal path whose interior vertices have degree 2, and whose
    endpoints have degree != 2 (tips deg=1, junctions deg>=3). Closed loops (all deg=2)
    are handled in a second pass.

    Returns
    -------
    polylines : list[dict]
        Each dict contains:
          - path_vertex_ids: ordered vertex ids along the branch
          - path_edge_indices: ordered edge indices along the branch
          - segment_lengths: list of segment lengths along the branch
          - total_length: total arc length
          - v_start, v_end: endpoint vertex ids (same for loops)
          - deg_start, deg_end: degrees of endpoints
    edge_to_polyline : dict[int,int]
        Map from network edge index to polyline id (pid).
    """
    # Backward compatible: if every edge is a straight 'line', use the legacy half-branch chaining logic.
    if not all(_edge_kind(e) == "line" for e in network.edges):
        polylines: List[dict] = []
        edge_to_polyline: dict[int, int] = {}
        pid = 0
        for e in network.edges:
            k = _edge_kind(e)
            v0 = int(e.v0); v1 = int(e.v1)
            p0 = network.vertex_coords(v0)
            p1 = network.vertex_coords(v1)
            eid = int(getattr(e, "id", pid))
            if k == "arc":
                ctrls = _edge_ctrl_vids(e)
                if len(ctrls) != 1:
                    raise ValueError(f"Arc edge {eid} must have exactly one ctrl_vid (center).")
                c = network.vertex_coords(int(ctrls[0]))
                polylines.append(make_arc_polyline(p0, p1, c, crack_mode="half", v_start=v0, v_end=v1))
            else:
                seg = p1 - p0
                segL = float(np.sqrt(np.sum(seg * seg)))
                polylines.append(dict(
                    kind="polyline",
                    path_vertex_ids=[v0, v1],
                    path_edge_indices=[eid],
                    segment_lengths=[segL],
                    total_length=segL,
                    crack_mode="half",
                    v_start=v0,
                    v_end=v1,
                    deg_start=int(deg.get(v0, 0)),
                    deg_end=int(deg.get(v1, 0)),
                ))
            edge_to_polyline[eid] = int(pid)
            pid += 1
        if not polylines:
            raise ValueError("No valid half polylines were built.")
        return polylines, edge_to_polyline
    v2e: dict[int, List[int]] = {}
    for eidx, e in enumerate(network.edges):
        a = int(e.v0); b = int(e.v1)
        v2e.setdefault(a, []).append(int(eidx))
        v2e.setdefault(b, []).append(int(eidx))

    def other_vertex(eidx: int, v: int) -> int:
        e = network.edges[int(eidx)]
        a = int(e.v0); b = int(e.v1)
        return b if a == v else a

    used_edges: set[int] = set()
    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}
    pid = 0

    # Terminal vertices start branches: deg != 2 (tips, junctions, isolated)
    start_verts = sorted([int(v) for v, d in deg.items() if int(d) != 2])

    for v0 in start_verts:
        inc_edges = v2e.get(int(v0), [])
        for e0 in inc_edges:
            e0 = int(e0)
            if e0 in used_edges:
                continue

            vids_path: List[int] = [int(v0)]
            eidx_path: List[int] = []
            cur_v = int(v0)
            cur_e = int(e0)

            while True:
                if cur_e in used_edges:
                    break
                used_edges.add(cur_e)
                eidx_path.append(int(cur_e))

                nxt_v = other_vertex(cur_e, cur_v)
                vids_path.append(int(nxt_v))

                dnext = int(deg.get(int(nxt_v), 0))
                if dnext != 2:
                    break

                # continue along the unique remaining unused incident edge at a deg-2 vertex
                inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
                if not inc:
                    break

                # In a clean deg-2 chain, there should be exactly one unused incident edge.
                # If there are more (deg mismatch), pick the first deterministically.
                cur_v = int(nxt_v)
                cur_e = int(inc[0])

            pts = np.array([network.vertex_coords(v) for v in vids_path], float)
            seg = pts[1:] - pts[:-1]
            segL = np.sqrt(np.sum(seg * seg, axis=1))
            L = float(np.sum(segL))
            if L <= 0:
                continue

            v_start = int(vids_path[0])
            v_end = int(vids_path[-1])

            # Orientation policy: keep tips at the start when possible
            if int(deg.get(v_start, 0)) == 1 and int(deg.get(v_end, 0)) > 1:
                vids_path = list(reversed(vids_path))
                eidx_path = list(reversed(eidx_path))
                segL = segL[::-1].copy()
                v_start = int(vids_path[0])
                v_end = int(vids_path[-1])

            for eidx in eidx_path:
                edge_to_polyline[int(eidx)] = int(pid)

            polylines.append(dict(
                path_vertex_ids=[int(v) for v in vids_path],
                path_edge_indices=[int(i) for i in eidx_path],
                segment_lengths=[float(x) for x in segL.tolist()],
                total_length=float(L),
                crack_mode="half",
                v_start=int(v_start),
                v_end=int(v_end),
                deg_start=int(deg.get(v_start, 0)),
                deg_end=int(deg.get(v_end, 0)),
            ))
            pid += 1

    # Second pass: handle closed loops (all vertices deg=2)
    for eidx0, e0 in enumerate(network.edges):
        eidx0 = int(eidx0)
        if eidx0 in used_edges:
            continue

        v0 = int(e0.v0)
        vids_path: List[int] = [v0]
        eidx_path: List[int] = []

        cur_v = v0
        cur_e = eidx0

        while True:
            if cur_e in used_edges:
                break
            used_edges.add(cur_e)
            eidx_path.append(int(cur_e))

            nxt_v = other_vertex(cur_e, cur_v)
            vids_path.append(int(nxt_v))

            if nxt_v == v0:
                break

            inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
            if not inc:
                break

            cur_v = int(nxt_v)
            cur_e = int(inc[0])

        # Require a closed loop
        if vids_path[-1] != v0:
            continue

        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))
        if L <= 0:
            continue

        for eidx in eidx_path:
            edge_to_polyline[int(eidx)] = int(pid)

        polylines.append(dict(
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(i) for i in eidx_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="half",
            v_start=int(v0),
            v_end=int(v0),
            deg_start=2,
            deg_end=2,
        ))
        pid += 1

    if not polylines:
        raise ValueError("No valid half polylines were built.")
    return polylines, edge_to_polyline




def discretize_polylines(
    network: CrackNetworkV4,
    polylines: List[dict],
    *,
    ne_half: int,
    node_distribution: str,
    collocation_mode: str,
    representation: str,
    nq_stress: int,
) -> List[dict]:
    use_tip_singular = (str(representation).lower().strip() == "singular")

    poly_panels: List[dict] = []
    for pid, p in enumerate(polylines):
        vids_path = p["path_vertex_ids"]
        segL = np.array(p["segment_lengths"], float)
        L = float(p["total_length"])

        Np = max(8, int(2 * ne_half))

        if str(node_distribution).lower().strip() == "uniform":
            s_nodes = np.linspace(0.0, L, Np + 1)
        else:
            theta = np.linspace(0.0, math.pi, Np + 1)
            s_nodes = 0.5 * L * (1.0 - np.cos(theta))

        if collocation_mode == "nodes" and Np >= 2:
            s_col = s_nodes[1:-1].copy()
        else:
            s_col = 0.5 * (s_nodes[:-1] + s_nodes[1:])

        s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
        ds = (s_nodes[1:] - s_nodes[:-1])

        x_mid = []; t_mid = []; n_mid = []
        for sm in s_mid:
            x, t, n = polyline_point_and_frame(network, vids_path, segL, float(sm))
            x_mid.append(x); t_mid.append(t); n_mid.append(n)
        x_mid = np.array(x_mid, float)
        t_mid = np.array(t_mid, float)
        n_mid = np.array(n_mid, float)

        x_col = []; t_col = []; n_col = []
        for sc in s_col:
            x, t, n = polyline_point_and_frame(network, vids_path, segL, float(sc))
            x_col.append(x); t_col.append(t); n_col.append(n)
        x_col = np.array(x_col, float)
        t_col = np.array(t_col, float)
        n_col = np.array(n_col, float)

        nq = max(2, int(nq_stress))
        xg, wg = np.polynomial.legendre.leggauss(nq)

        src_pts = []; src_t = []; src_n = []; src_w = []
        panel_src = []
        panel_wsing = []
        eps_s = 1e-14 * L if L > 0 else 1e-14

        idx0 = 0
        for k in range(Np):
            sLk = float(s_nodes[k]); sRk = float(s_nodes[k + 1])
            Jk = 0.5 * (sRk - sLk)
            smk = 0.5 * (sRk + sLk)
            sq = smk + Jk * xg
            wq = Jk * wg

            wsing_k = 0.0
            for sqq, wqq in zip(sq, wq):
                x, t, n = polyline_point_and_frame(network, vids_path, segL, float(sqq))
                src_pts.append(x); src_t.append(t); src_n.append(n)

                if use_tip_singular:
                    if p.get("crack_mode", "full") == "full":
                        ss = max(float(sqq), eps_s)
                        tt = max(float(L - float(sqq)), eps_s)
                        sing = 1.0 / math.sqrt(ss * tt)
                    else:
                        tt = max(float(L - float(sqq)), eps_s)
                        sing = 1.0 / math.sqrt(tt)
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
            s_nodes=np.array(s_nodes, float),
            s_mid=np.array(s_mid, float),
            ds=np.array(ds, float),
            x_mid=x_mid, t_mid=t_mid, n_mid=n_mid,
            x_col=x_col, t_col=t_col, n_col=n_col,
            src_pts=np.array(src_pts, float),
            src_t=np.array(src_t, float),
            src_n=np.array(src_n, float),
            src_w=np.array(src_w, float),
            panel_src=panel_src,
            panel_wsing=np.array(panel_wsing, float),
            v_start=int(p.get("v_start", vids_path[0])),
            v_end=int(p.get("v_end", vids_path[-1])),
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

    Returns
    -------
    offsets : list[int]
        Start index of density unknowns for each pid.
    junction_dof : dict[int,int]
        For strict: vertex -> DOF offset (Jx,Jy). Empty for core/soft.
    branch_end_dof : dict[(pid,endflag), int]
        For core/soft: (pid,'start'/'end') -> DOF offset (Jx,Jy). Empty for strict.
    nunk : int
        Total number of unknowns.
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
            # Per-branch-end DOFs at endpoints with degree>=2 (open-junction models)
            keys = []
            for pid, pp in enumerate(poly_panels):
                vs = int(pp["v_start"]); ve = int(pp["v_end"])
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

    sig = np.array([
        [float(applied.sigma_xx), float(applied.sigma_xy)],
        [float(applied.sigma_xy), float(applied.sigma_yy)],
    ], float)

    row0 = 0
    for pid_i, pp_i in enumerate(poly_panels):
        Nci = int(len(pp_i["x_col"]))
        for ic in range(Nci):
            x_i = pp_i["x_col"][ic]
            t_i = pp_i["t_col"][ic]
            n_i = pp_i["n_col"][ic]

            tr = sig @ n_i.reshape(2,)
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

                    tn_sum_I = 0.0; ts_sum_I = 0.0
                    tn_sum_II = 0.0; ts_sum_II = 0.0

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
