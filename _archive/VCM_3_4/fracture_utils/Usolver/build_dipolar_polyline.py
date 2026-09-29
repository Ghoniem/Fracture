"""
Polyline construction for FULL (dipolar) crack mode.

- Chains LINE edges into polylines by connected components + path ordering
- Adds ARC edges as standalone polylines (center-defined)

Curved edge evaluation is handled by build_curved; graph utilities by build_geometry.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .network import CrackNetworkV4
from .build_geometry import line_edge_positions, arc_edge_positions, components_from_edge_positions, order_path_from_component, edge_ctrl_vids
from .build_curved import make_arc_polyline


def build_polylines_full(network: CrackNetworkV4) -> Tuple[List[dict], dict[int, int]]:
    """Full crack mode: chain LINE edges; append ARC edges as standalone polylines."""
    if not network.edges:
        raise ValueError("No edges in network.")

    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}

    # --- LINE components
    line_epos = line_edge_positions(network)
    comps = components_from_edge_positions(network, line_epos)

    for pid, comp in enumerate(comps):
        vids_path, epos_path = order_path_from_component(network, comp)
        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))

        for ep in epos_path:
            edge_to_polyline[int(ep)] = int(pid)

        polylines.append(dict(
            kind="polyline",
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(ep) for ep in epos_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="full",
            v_start=int(vids_path[0]),
            v_end=int(vids_path[-1]),
        ))

    # --- ARC edges: standalone
    arc_epos = arc_edge_positions(network)
    pid0 = int(len(polylines))
    for j, ep in enumerate(arc_epos):
        e = network.edges[int(ep)]
        ctrls = edge_ctrl_vids(e)
        if len(ctrls) != 1:
            raise ValueError(f"Arc edge {int(getattr(e,'id',ep))} must have exactly one ctrl_vid (center).")
        v0 = int(e.v0)
        v1 = int(e.v1)
        c = network.vertex_coords(int(ctrls[0]))
        p0 = network.vertex_coords(v0)
        p1 = network.vertex_coords(v1)
        meta = make_arc_polyline(p0, p1, c, crack_mode="full", v_start=v0, v_end=v1)
        meta["path_edge_indices"] = [int(ep)]
        polylines.append(meta)
        edge_to_polyline[int(ep)] = pid0 + j

    if not polylines:
        raise ValueError("No valid polylines were built.")
    return polylines, edge_to_polyline
