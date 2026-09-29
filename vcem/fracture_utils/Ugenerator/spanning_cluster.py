"""Spanning-cluster detection for the disk fracture problem.

A "spanning cluster" is a connected component of the crack network whose
vertex set contains at least two distinct boundary points of the disk.
When such a cluster exists, the loaded region is no longer simply
connected -- the disk has fractured into separate pieces -- and further
BEM solves give unphysical stress intensity factors.

Boundary vertices are detected two ways and unioned:
  - vertices flagged by the propagator as snapped-to-boundary (their ids
    accumulate in CrackPropagator.snapped_vertex_ids); and
  - vertices whose Cartesian radius lies within `boundary_tol_m` of the
    disk radius (catches vertices that are on the boundary by
    construction or by round-off).
"""
from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

import numpy as np


def boundary_vertex_ids(
    network,
    disk_R: float,
    *,
    snapped_vertex_ids: Optional[Iterable[int]] = None,
    boundary_tol_m: float = 1.0e-9,
) -> Set[int]:
    """Return the set of vertex ids considered to lie on the disk boundary."""
    out: Set[int] = set()
    if snapped_vertex_ids is not None:
        out.update(int(v) for v in snapped_vertex_ids)
    if disk_R is not None and float(disk_R) > 0.0:
        R = float(disk_R)
        tol = float(boundary_tol_m)
        for v in network.vertices:
            r = float(np.hypot(float(v.x), float(v.y)))
            if r >= R - tol:
                out.add(int(v.id))
    return out


def find_spanning_clusters(
    network,
    disk_R: float,
    *,
    snapped_vertex_ids: Optional[Iterable[int]] = None,
    boundary_tol_m: float = 1.0e-9,
) -> list[Tuple[Set[int], Set[int]]]:
    """Return all connected components that contain >=2 boundary vertices.

    Each entry is (component_vertex_ids, boundary_vertex_ids_in_component).
    Result is empty when no component spans the boundary.
    """
    bvids = boundary_vertex_ids(
        network, disk_R,
        snapped_vertex_ids=snapped_vertex_ids,
        boundary_tol_m=boundary_tol_m,
    )
    if len(bvids) < 2:
        return []

    # Build adjacency on vertex ids.
    adj: dict[int, list[int]] = {int(v.id): [] for v in network.vertices}
    for e in network.edges:
        a, b = int(e.v0), int(e.v1)
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)

    # BFS over each unvisited vertex; collect component, then test.
    visited: Set[int] = set()
    spanning: list[Tuple[Set[int], Set[int]]] = []
    for seed in adj:
        if seed in visited:
            continue
        comp: Set[int] = set()
        stack = [seed]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.add(v)
            stack.extend(adj[v])
        b_in_comp = comp & bvids
        if len(b_in_comp) >= 2:
            spanning.append((comp, b_in_comp))
    return spanning


def has_spanning_cluster(
    network,
    disk_R: float,
    *,
    snapped_vertex_ids: Optional[Iterable[int]] = None,
    boundary_tol_m: float = 1.0e-9,
) -> bool:
    """True iff at least one connected component contains >=2 boundary vertices."""
    return len(find_spanning_clusters(
        network, disk_R,
        snapped_vertex_ids=snapped_vertex_ids,
        boundary_tol_m=boundary_tol_m,
    )) > 0
