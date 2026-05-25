"""
Helper module: polyline construction, discretization, operator assembly for solver_parametrized_half.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import math

from .solver_network import CrackNetworkV4
from .solver_material import Material, AppliedStress

from .solver_kernels_half import stress_edge_dislocation
from .solver_polyline_half import polyline_point_and_frame


def vertex_degrees(network: CrackNetworkV4) -> dict[int, int]:
    deg = {int(v.id): 0 for v in network.vertices}
    for e in network.edges:
        deg[int(e.v0)] = deg.get(int(e.v0), 0) + 1
        deg[int(e.v1)] = deg.get(int(e.v1), 0) + 1
    return deg


def build_polylines_full(network: CrackNetworkV4) -> Tuple[List[dict], dict[int, int]]:
    from utils.solver_polyline import find_polyline_components, path_order_for_component

    comps = find_polyline_components(network)
    if not comps:
        raise ValueError("No edges in network.")

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
    v2e: dict[int, List[int]] = {}
    for eidx, e in enumerate(network.edges):
        a = int(e.v0); b = int(e.v1)
        v2e.setdefault(a, []).append(int(eidx))
        v2e.setdefault(b, []).append(int(eidx))

    def other_vertex(eidx: int, v: int) -> int:
        e = network.edges[int(eidx)]
        a = int(e.v0); b = int(e.v1)
        return b if a == v else a

    used_edges = set()
    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}
    pid = 0

    start_verts = sorted([int(v) for v, d in deg.items() if int(d) >= 2])

    for v0 in start_verts:
        for e0 in v2e.get(v0, []):
            if int(e0) in used_edges:
                continue

            vids_path = [int(v0)]
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

                inc = [ei for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
                if not inc:
                    break
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

    for eidx, e in enumerate(network.edges):
        if int(eidx) in used_edges:
            continue
        a = int(e.v0); b = int(e.v1)
        pa = np.array(network.vertex_coords(a), float)
        pb = np.array(network.vertex_coords(b), float)
        L = float(np.linalg.norm(pb - pa))
        if L <= 0:
            continue
        vids_path = [a, b]
        if int(deg.get(a, 0)) == 1 and int(deg.get(b, 0)) > 1:
            vids_path = [b, a]
        v_start = int(vids_path[0])
        v_end = int(vids_path[-1])

        edge_to_polyline[int(eidx)] = int(pid)
        polylines.append(dict(
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(eidx)],
            segment_lengths=[float(L)],
            total_length=float(L),
            crack_mode="half",
            v_start=int(v_start),
            v_end=int(v_end),
            deg_start=int(deg.get(v_start, 0)),
            deg_end=int(deg.get(v_end, 0)),
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
) -> Tuple[List[int], Dict[int, int], int]:
    offsets: List[int] = []
    nunk = 0

    for pp in poly_panels:
        offsets.append(int(nunk))
        nunk += 2 * int(pp["Np"])

    junction_dof: Dict[int, int] = {}
    if str(crack_mode).lower().strip() == "half":
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

    return offsets, junction_dof, int(nunk)


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
