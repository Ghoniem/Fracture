"""
Topology / graph utilities for crack networks and edge subgraphs.

This module contains:
- edge-kind helpers (line/arc/etc.)
- vertex degrees
- LINE-subgraph connected components and path ordering

No quadrature, operator assembly, or discretization lives here.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .network import CrackNetworkV4

def edge_kind(e) -> str:
    return str(getattr(e, "kind", "line")).lower().strip()


def edge_ctrl_vids(e) -> tuple:
    return tuple(getattr(e, "ctrl_vids", ()))


def vertex_degrees(network: CrackNetworkV4) -> dict[int, int]:
    deg = {int(v.id): 0 for v in network.vertices}
    for e in network.edges:
        deg[int(e.v0)] = deg.get(int(e.v0), 0) + 1
        deg[int(e.v1)] = deg.get(int(e.v1), 0) + 1
    return deg

def line_edge_positions(network: CrackNetworkV4) -> list[int]:
    return [int(i) for i, e in enumerate(network.edges) if edge_kind(e) == "line"]

def arc_edge_positions(network: CrackNetworkV4) -> list[int]:
    return [int(i) for i, e in enumerate(network.edges) if edge_kind(e) == "arc"]

def components_from_edge_positions(network: CrackNetworkV4, edge_pos: list[int]) -> list[list[int]]:
    """Connected components of an edge-induced subgraph (edges referenced by their index in network.edges)."""
    if not edge_pos:
        return []
    v2e: dict[int, list[int]] = {}
    for ep in edge_pos:
        e = network.edges[int(ep)]
        a = int(e.v0)
        b = int(e.v1)
        v2e.setdefault(a, []).append(int(ep))
        v2e.setdefault(b, []).append(int(ep))

    unseen = set(int(ep) for ep in edge_pos)
    comps: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cur = stack.pop()
            e = network.edges[int(cur)]
            for v in (int(e.v0), int(e.v1)):
                for ej in v2e.get(v, []):
                    if ej in unseen:
                        unseen.remove(ej)
                        stack.append(ej)
                        comp.append(ej)
        comps.append(comp)
    return comps

def order_path_from_component(network: CrackNetworkV4, edge_pos: list[int]) -> tuple[list[int], list[int]]:
    """
    Order a connected component of LINE edges into (vertex path, edge_pos path).

    Works for both open chains and closed loops. For loops, start at smallest vertex id.
    """
    v2e: dict[int, list[int]] = {}
    for ep in edge_pos:
        e = network.edges[int(ep)]
        a = int(e.v0)
        b = int(e.v1)
        v2e.setdefault(a, []).append(int(ep))
        v2e.setdefault(b, []).append(int(ep))

    deg_sub = {v: len(es) for v, es in v2e.items()}
    ends = [v for v, d in deg_sub.items() if d == 1]

    def other_vertex(ep: int, v: int) -> int:
        e = network.edges[int(ep)]
        a = int(e.v0)
        b = int(e.v1)
        return b if a == v else a

    start = int(ends[0]) if len(ends) == 2 else int(min(deg_sub.keys()))
    vids_path = [start]
    epath: list[int] = []
    used: set[int] = set()
    cur_v = start
    prev_ep = None

    while True:
        cand = [ep for ep in v2e.get(cur_v, []) if ep not in used]
        if not cand:
            break
        if prev_ep is not None and len(cand) > 1:
            cand2 = [ep for ep in cand if ep != prev_ep]
            if cand2:
                cand = cand2
        cur_ep = int(sorted(cand)[0])
        used.add(cur_ep)
        epath.append(cur_ep)
        nxt = other_vertex(cur_ep, cur_v)
        vids_path.append(int(nxt))
        prev_ep = cur_ep
        cur_v = int(nxt)
        if cur_v == start and len(used) == len(edge_pos):
            break

    return vids_path, epath
