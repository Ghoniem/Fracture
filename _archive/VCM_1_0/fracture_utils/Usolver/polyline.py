"""Polyline extraction and arc-length geometry helpers."""
from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
from .network import CrackNetworkV4

def _build_adjacency(network: CrackNetworkV4):
    V = [int(v.id) for v in network.vertices]
    E = network.edges
    v2e = {vid: [] for vid in V}
    e2v = []
    for ei, e in enumerate(E):
        a = int(e.v0); b = int(e.v1)
        e2v.append((a, b))
        v2e.setdefault(a, []).append(int(ei))
        v2e.setdefault(b, []).append(int(ei))
    return v2e, e2v


def find_polyline_components(network: CrackNetworkV4) -> list[dict]:
    v2e, e2v = _build_adjacency(network)
    v_adj = {int(v.id): set() for v in network.vertices}
    for (a, b) in e2v:
        v_adj[a].add(b)
        v_adj[b].add(a)

    unvisited = set(v_adj.keys())
    comps = []
    while unvisited:
        start = next(iter(unvisited))
        stack = [start]
        comp_verts = set()
        comp_edges = set()
        while stack:
            v = stack.pop()
            if v not in unvisited:
                continue
            unvisited.remove(v)
            comp_verts.add(int(v))
            for ei in v2e.get(v, []):
                comp_edges.add(int(ei))
                a, b = e2v[ei]
                w = b if a == v else a
                if w in unvisited:
                    stack.append(w)
        if comp_edges:
            comps.append({"verts": comp_verts, "edges": comp_edges})
    return comps


def path_order_for_component(network: CrackNetworkV4, comp_edges: set[int]) -> Tuple[List[int], List[int]]:
    v2e, e2v = _build_adjacency(network)
    comp_edges = set(int(ei) for ei in comp_edges)

    deg = {}
    comp_verts = set()
    for ei in comp_edges:
        a, b = e2v[ei]
        comp_verts.add(a); comp_verts.add(b)
    for v in comp_verts:
        inc = [ei for ei in v2e.get(v, []) if ei in comp_edges]
        deg[v] = len(inc)

    ends = [v for v, d in deg.items() if d == 1]
    if len(ends) != 2 or any(d not in (1, 2) for d in deg.values()):
        raise ValueError(
            "v4 polyline solver requires each connected component to be a simple chain "
            "(exactly two degree-1 vertices, all others degree-2)."
        )

    start = int(ends[0])
    vids_path = [start]
    eidx_path = []
    cur = start
    used_edges = set()
    for _ in range(len(comp_edges)):
        inc = [ei for ei in v2e.get(cur, []) if (ei in comp_edges and ei not in used_edges)]
        if not inc:
            break
        ei = int(inc[0])
        used_edges.add(ei)
        a, b = e2v[ei]
        nxt = b if a == cur else a
        eidx_path.append(ei)
        vids_path.append(int(nxt))
        cur = int(nxt)

    if len(eidx_path) != len(comp_edges):
        raise ValueError("Could not traverse component as a single path.")
    return vids_path, eidx_path


def polyline_point_and_frame(network: CrackNetworkV4, vids_path: list[int], seg_lengths: np.ndarray, s: float):
    """Map arc-length s in [0,L] to physical point x(s), and return local tangent and normal."""
    s = float(s)
    pts = np.array([network.vertex_coords(v) for v in vids_path], float)
    Ls = np.asarray(seg_lengths, float)
    Lcum = np.r_[0.0, np.cumsum(Ls)]
    L = float(Lcum[-1])
    s = min(max(s, 0.0), L)

    k = int(np.searchsorted(Lcum, s, side="right") - 1)
    k = max(0, min(k, len(Ls)-1))
    s0 = float(Lcum[k])
    le = float(Ls[k])
    lam = 0.0 if le <= 0 else (s - s0)/le
    p0 = pts[k]; p1 = pts[k+1]
    x = p0 + lam*(p1-p0)
    t = p1 - p0
    nrm = float(np.hypot(t[0], t[1]))
    if nrm <= 0:
        t = np.array([1.0, 0.0])
    else:
        t = t / nrm
    n = np.array([-t[1], t[0]])
    return x, t, n


# -------------------------
# Half-crack helpers
# -------------------------
def vertex_degrees(network: CrackNetworkV4) -> dict[int,int]:
    """Return vertex degree (number of incident edges) for each vertex id."""
    v2e, _ = _build_adjacency(network)
    return {int(v): int(len(eis)) for v, eis in v2e.items()}

def half_polylines_from_edges(network: CrackNetworkV4) -> list[dict]:
    """Build a polyline-like component list where each edge is treated as an independent 'half crack'.

    This is intentionally minimal: it enables junction modeling by letting each incident edge carry its
    own density, while junction closure is enforced via constraints in the solver.

    Returns a list of dicts with keys:
      - 'verts': {v0, v1}
      - 'edges': {ei}
    """
    comps = []
    for ei, e in enumerate(network.edges):
        comps.append({"verts": {int(e.v0), int(e.v1)}, "edges": {int(ei)}})
    return comps
