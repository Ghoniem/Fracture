"""
Polyline construction for HALF (polar) crack mode.

- Chains LINE edges into maximal branches whose interior vertices have degree 2
- Handles closed loops in the LINE subgraph
- Adds ARC edges as standalone branches (center-defined)

Curved edge evaluation is handled by build_curved; graph utilities by build_geometry.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .network import CrackNetworkV4
from .build_geometry import line_edge_positions, arc_edge_positions, edge_ctrl_vids
from .build_curved import make_arc_polyline


def build_polylines_half_branches(
    network: CrackNetworkV4,
    deg: dict[int, int],
) -> Tuple[List[dict], dict[int, int]]:
    """Half crack mode: chain LINE edges into branches; append ARC edges as standalone branches."""
    if not network.edges:
        raise ValueError("No edges in network.")

    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}
    pid = 0

    # --- LINE-edge-only chaining
    line_epos = set(line_edge_positions(network))

    v2e: dict[int, List[int]] = {}
    for ep in sorted(line_epos):
        e = network.edges[int(ep)]
        a = int(e.v0)
        b = int(e.v1)
        v2e.setdefault(a, []).append(int(ep))
        v2e.setdefault(b, []).append(int(ep))

    def other_vertex(ep: int, v: int) -> int:
        e = network.edges[int(ep)]
        a = int(e.v0)
        b = int(e.v1)
        return b if a == v else a

    used_edges: set[int] = set()

    start_verts = sorted([int(v) for v, d in deg.items() if int(d) != 2])
    for v0 in start_verts:
        inc_edges = v2e.get(int(v0), [])
        for e0 in inc_edges:
            e0 = int(e0)
            if e0 in used_edges:
                continue

            vids_path: List[int] = [int(v0)]
            epos_path: List[int] = []
            cur_v = int(v0)
            cur_e = int(e0)

            while True:
                if cur_e in used_edges:
                    break
                used_edges.add(cur_e)
                epos_path.append(int(cur_e))

                nxt_v = other_vertex(cur_e, cur_v)
                vids_path.append(int(nxt_v))

                dnext = int(deg.get(int(nxt_v), 0))
                if dnext != 2:
                    break

                inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
                if not inc:
                    break
                cur_v = int(nxt_v)
                cur_e = int(sorted(inc)[0])

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
                epos_path = list(reversed(epos_path))
                segL = segL[::-1].copy()
                v_start = int(vids_path[0])
                v_end = int(vids_path[-1])

            for ep in epos_path:
                edge_to_polyline[int(ep)] = int(pid)

            polylines.append(dict(
                kind="polyline",
                path_vertex_ids=[int(v) for v in vids_path],
                path_edge_indices=[int(ep) for ep in epos_path],
                segment_lengths=[float(x) for x in segL.tolist()],
                total_length=float(L),
                crack_mode="half",
                v_start=int(v_start),
                v_end=int(v_end),
                deg_start=int(deg.get(v_start, 0)),
                deg_end=int(deg.get(v_end, 0)),
            ))
            pid += 1

    # Second pass: closed loops in LINE subgraph
    for ep0 in sorted(line_epos):
        if int(ep0) in used_edges:
            continue
        e0 = network.edges[int(ep0)]
        v0 = int(e0.v0)

        vids_path: List[int] = [v0]
        epos_path: List[int] = []
        cur_v = v0
        cur_e = int(ep0)

        while True:
            if cur_e in used_edges:
                break
            used_edges.add(cur_e)
            epos_path.append(int(cur_e))

            nxt_v = other_vertex(cur_e, cur_v)
            vids_path.append(int(nxt_v))

            if nxt_v == v0:
                break

            inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
            if not inc:
                break
            cur_v = int(nxt_v)
            cur_e = int(sorted(inc)[0])

        if vids_path[-1] != v0:
            continue

        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))
        if L <= 0:
            continue

        for ep in epos_path:
            edge_to_polyline[int(ep)] = int(pid)

        polylines.append(dict(
            kind="polyline",
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(ep) for ep in epos_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="half",
            v_start=int(v0),
            v_end=int(v0),
            deg_start=2,
            deg_end=2,
        ))
        pid += 1

    # --- ARC edges: standalone branches
    arc_epos = arc_edge_positions(network)
    for ep in arc_epos:
        e = network.edges[int(ep)]
        ctrls = edge_ctrl_vids(e)
        if len(ctrls) != 1:
            raise ValueError(f"Arc edge {int(getattr(e,'id',ep))} must have exactly one ctrl_vid (center).")
        v0 = int(e.v0)
        v1 = int(e.v1)
        c = network.vertex_coords(int(ctrls[0]))
        p0 = network.vertex_coords(v0)
        p1 = network.vertex_coords(v1)
        meta = make_arc_polyline(p0, p1, c, crack_mode="half", v_start=v0, v_end=v1)
        meta["deg_start"] = int(deg.get(v0, 0))
        meta["deg_end"] = int(deg.get(v1, 0))
        meta["path_edge_indices"] = [int(ep)]
        polylines.append(meta)
        edge_to_polyline[int(ep)] = int(pid)
        pid += 1

    if not polylines:
        raise ValueError("No valid half polylines were built.")
    return polylines, edge_to_polyline
