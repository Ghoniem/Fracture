# fracture_utils/Upropagation/network_ops.py
# Network operations for propagation that work on general graphs (junctions allowed).
#
# We DO NOT rely on fracture_utils.Usolver.polyline.path_order_for_component because that helper
# assumes each connected component is a simple chain (deg sequence 1-2-...-2-1).
#
# Instead, we decompose the network into "open polylines" by splitting at all vertices with degree != 2
# (tips deg=1, junctions deg>=3, isolated deg=0). Each polyline is a maximal chain of deg-2 vertices
# between two "special" vertices (deg!=2). This supports:
#   - tip <-> tip
#   - tip <-> junction
#   - junction <-> junction
#
# Tip extraction then simply picks the deg-1 ends of these polylines.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class PolylinePath:
    pid: int
    vids: List[int]
    eids: List[int]


@dataclass(frozen=True)
class PolylineMetrics:
    """Lightweight geometric/topological metrics for a PolylinePath."""
    pid: int
    length: float
    n_vertices: int
    v_start: int
    v_end: int
    deg_start: int
    deg_end: int

    @property
    def is_tip_to_tip(self) -> bool:
        return (self.deg_start == 1) and (self.deg_end == 1)

    @property
    def is_tip_to_junction(self) -> bool:
        return (self.deg_start == 1 and self.deg_end >= 3) or (self.deg_end == 1 and self.deg_start >= 3)

    @property
    def is_junction_to_junction(self) -> bool:
        return (self.deg_start >= 3) and (self.deg_end >= 3)


def polyline_metrics(
    network: Any,
    polylines: Sequence[PolylinePath] | None = None,
    deg: Dict[int, int] | None = None,
) -> List[PolylineMetrics]:
    """Compute per-polyline metrics used by propagation policies.

    Notes
    -----
    * A "polyline" here is a maximal chain of deg-2 vertices between two special vertices (deg != 2),
      as returned by :func:`extract_open_polylines`.
    * The returned `length` is the *arc length* along the polyline geometry (sum of segment lengths),
      not a chord length.
    """
    if deg is None:
        deg = degree_map(network)
    if polylines is None:
        polylines = extract_open_polylines(network)

    out: List[PolylineMetrics] = []
    for pl in polylines:
        vids = pl.vids or []
        if len(vids) < 2:
            continue
        v0 = int(vids[0])
        v1 = int(vids[-1])
        L = _polyline_length(network, vids)
        out.append(
            PolylineMetrics(
                pid=int(pl.pid),
                length=float(L),
                n_vertices=int(len(vids)),
                v_start=v0,
                v_end=v1,
                deg_start=int(deg.get(v0, 0)),
                deg_end=int(deg.get(v1, 0)),
            )
        )
    return out


def longest_polyline(metrics: Sequence[PolylineMetrics]) -> PolylineMetrics | None:
    """Return the polyline with maximum arc length (or None if empty)."""
    if not metrics:
        return None
    return max(metrics, key=lambda m: float(m.length))


def polyline_id_to_metrics(metrics: Sequence[PolylineMetrics]) -> Dict[int, PolylineMetrics]:
    """Convenience map pid -> metrics."""
    return {int(m.pid): m for m in metrics}

def _iter_edges(network: Any):
    for e in getattr(network, "edges", []):
        eid = int(getattr(e, "id"))
        v0 = int(getattr(e, "v0"))
        v1 = int(getattr(e, "v1"))
        yield eid, v0, v1


def degree_map(network: Any) -> Dict[int, int]:
    deg: Dict[int, int] = {}
    for eid, v0, v1 in _iter_edges(network):
        deg[v0] = deg.get(v0, 0) + 1
        deg[v1] = deg.get(v1, 0) + 1
    for v in getattr(network, "vertices", []):
        vid = int(getattr(v, "id"))
        deg.setdefault(vid, 0)
    return deg


def _adjacency(network: Any) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, Tuple[int, int]]]:
    v2nbrs: Dict[int, List[Tuple[int, int]]] = {}
    eid2v: Dict[int, Tuple[int, int]] = {}
    for eid, v0, v1 in _iter_edges(network):
        eid2v[eid] = (v0, v1)
        v2nbrs.setdefault(v0, []).append((eid, v1))
        v2nbrs.setdefault(v1, []).append((eid, v0))
    return v2nbrs, eid2v


def _vertex_xy(network: Any, vid: int) -> np.ndarray:
    if hasattr(network, "vertex_coords"):
        return np.asarray(network.vertex_coords(int(vid)), float).reshape(2,)
    for v in getattr(network, "vertices", []):
        if int(getattr(v, "id")) == int(vid):
            return np.array([float(getattr(v, "x")), float(getattr(v, "y"))], float)
    raise KeyError(f"vertex {vid} not found")


def extract_open_polylines(network: Any) -> List[PolylinePath]:
    deg = degree_map(network)
    v2nbrs, _ = _adjacency(network)

    specials = {vid for vid, d in deg.items() if d != 2}
    visited_dir = set()  # (vid, eid) leaving vid along eid

    polylines: List[PolylinePath] = []
    pid = 0

    for v_start in sorted(specials):
        for eid, v_next in v2nbrs.get(v_start, []):
            if (v_start, eid) in visited_dir:
                continue

            vids = [int(v_start)]
            eids = [int(eid)]
            visited_dir.add((v_start, eid))

            v_prev = int(v_start)
            v_cur = int(v_next)

            while deg.get(v_cur, 0) == 2 and v_cur not in specials:
                vids.append(v_cur)
                nbrs = v2nbrs.get(v_cur, [])
                if len(nbrs) != 2:
                    break
                (eid0, n0), (eid1, n1) = nbrs
                if int(n0) == int(v_prev):
                    eid_next, v_next2 = int(eid1), int(n1)
                else:
                    eid_next, v_next2 = int(eid0), int(n0)

                eids.append(eid_next)
                visited_dir.add((v_cur, eid_next))
                v_prev, v_cur = v_cur, v_next2

            vids.append(int(v_cur))
            visited_dir.add((int(v_cur), int(eids[-1])))

            # canonical orientation
            a = vids[0]; b = vids[-1]
            if b < a:
                vids = list(reversed(vids))
                eids = list(reversed(eids))

            key = (tuple(vids), tuple(eids))
            if any((tuple(p.vids), tuple(p.eids)) == key for p in polylines):
                continue

            polylines.append(PolylinePath(pid=pid, vids=vids, eids=eids))
            pid += 1

    return polylines


def _segment_tangent_outward(network: Any, tip_vid: int, neighbor_vid: int) -> np.ndarray:
    x_tip = _vertex_xy(network, tip_vid)
    x_nb = _vertex_xy(network, neighbor_vid)
    t = (x_tip - x_nb)
    n = float(np.hypot(t[0], t[1]))
    if n <= 0:
        return np.array([1.0, 0.0], float)
    return t / n


def _polyline_length(network: Any, vids: Sequence[int]) -> float:
    L = 0.0
    for i in range(len(vids) - 1):
        p0 = _vertex_xy(network, int(vids[i]))
        p1 = _vertex_xy(network, int(vids[i + 1]))
        L += float(np.hypot(*(p1 - p0)))
    return float(L)


def extract_deg1_tips(network: Any, polylines: Sequence[PolylinePath], deg: Dict[int, int]):
    try:
        from .tip_state import TipState, TipID
    except Exception as e:
        raise ImportError(f"Could not import TipState/TipID from Upropagation.tip_state: {e}")

    tips = []
    for pl in polylines:
        vids = pl.vids
        if len(vids) < 2:
            continue
        Ltot = _polyline_length(network, vids)

        v0 = int(vids[0])
        v1 = int(vids[1])
        if deg.get(v0, 0) == 1:
            x = _vertex_xy(network, v0)
            t_hat = _segment_tangent_outward(network, v0, v1)
            tip_id = TipID(pid=int(pl.pid), which="start")
            tips.append(
                TipState(
                    tip_id=tip_id,
                    v_tip=int(v0),
                    x_tip=np.asarray(x, float),
                    t_hat=np.asarray(t_hat, float),
                    total_length=float(Ltot),
                )
            )

        vn = int(vids[-1])
        vn1 = int(vids[-2])
        if deg.get(vn, 0) == 1:
            x = _vertex_xy(network, vn)
            t_hat = _segment_tangent_outward(network, vn, vn1)
            tip_id = TipID(pid=int(pl.pid), which="end")
            tips.append(
                TipState(
                    tip_id=tip_id,
                    v_tip=int(vn),
                    x_tip=np.asarray(x, float),
                    t_hat=np.asarray(t_hat, float),
                    total_length=float(Ltot),
                )
            )

    return tips


class _NetOps:
    def degree_map(self, network: Any) -> Dict[int, int]:
        return degree_map(network)

    def extract_open_polylines(self, network: Any) -> List[PolylinePath]:
        return extract_open_polylines(network)

    def extract_deg1_tips(self, network: Any, polylines: Sequence[PolylinePath], deg: Dict[int, int]):
        return extract_deg1_tips(network, polylines, deg)

    def polyline_metrics(self, network: Any, polylines: Sequence[PolylinePath] | None = None, deg: Dict[int, int] | None = None):
        return polyline_metrics(network, polylines=polylines, deg=deg)

    def longest_polyline(self, metrics: Sequence[PolylineMetrics]):
        return longest_polyline(metrics)

    def polyline_id_to_metrics(self, metrics: Sequence[PolylineMetrics]) -> Dict[int, PolylineMetrics]:
        return polyline_id_to_metrics(metrics)



_NETOPS = _NetOps()


def get_netops():
    return _NETOPS
