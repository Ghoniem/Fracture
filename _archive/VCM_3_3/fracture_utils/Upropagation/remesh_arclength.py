"""Arc-length based remeshing policy (Option A).

Goal
----
Provide (near-)symmetric node placement along a growing polyline crack, even after
multiple growth steps that introduce many segments.

Key ideas
---------
1) Work in a global arc-length coordinate s along each polyline crack.
2) Place nodes in s (not per-segment), with symmetric clustering near both tips.
3) Add "feature budgets" (tips/kinks/junctions) while enforcing a minimum bulk density
   so the crack is never starved in the interior as it grows.

Current scope (v1)
------------------
- Supports a single polyline crack (one connected chain with 2 degree-1 tips, no junctions).
- For general crack networks with junctions/branches, this file provides the API and
  helper routines, but the `remesh_network` method will raise NotImplementedError.

This module is designed to be used by Upropagation's propagator after each accepted
growth step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
except Exception:  # pragma: no cover
    CrackNetworkV4 = object  # type: ignore
    VertexV4 = object  # type: ignore
    EdgeV4 = object  # type: ignore


@dataclass
class ArcLengthBudget:
    """Controls how node budget is spent along a polyline."""
    # Minimum "bulk" node fraction (rest can be spent in clustered regions)
    bulk_frac_min: float = 0.55
    # Minimum number of bulk intervals (panels) regardless of crack length
    bulk_min_panels: int = 10

    # Tip clustering
    tip_cluster: str = "power"          # "power" or "cheb"
    tip_cluster_power: float = 2.0      # used when tip_cluster == "power"
    tip_frac: float = 0.18              # fraction of total nodes assigned to each tip region

    # Kink clustering (deg-2 interior vertices)
    kink_halfwidth_frac: float = 0.06   # half-window around kink in s/L
    kink_extra_per_kink: int = 6        # extra nodes per kink region


def _arc_length(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return cumulative arc-length array s (same length as pts) and total length."""
    d = np.diff(pts, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1]) if len(s) else 0.0
    return s, L


def _interp_polyline(pts: np.ndarray, s_nodes: np.ndarray, s_cum: np.ndarray) -> np.ndarray:
    """Map arc-length positions s_nodes to xy using piecewise linear interpolation."""
    # Guard
    if len(pts) < 2:
        return pts.copy()

    s_nodes = np.clip(np.asarray(s_nodes, float), float(s_cum[0]), float(s_cum[-1]))

    # segment index for each s_node
    idx = np.searchsorted(s_cum, s_nodes, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 2)

    s0 = s_cum[idx]
    s1 = s_cum[idx + 1]
    # avoid divide by zero
    t = (s_nodes - s0) / np.maximum(s1 - s0, 1e-30)

    p0 = pts[idx]
    p1 = pts[idx + 1]
    xy = (1.0 - t)[:, None] * p0 + t[:, None] * p1
    return xy


def _symmetric_tip_nodes(L: float, n: int, *, cluster: str, power: float) -> np.ndarray:
    """Generate symmetric nodes on [0,L] clustered near both ends."""
    if n < 2:
        return np.array([0.0], float)

    if cluster.lower().startswith("cheb"):
        # Chebyshev nodes mapped to [0,L] (symmetric by construction)
        j = np.arange(n, dtype=float)
        x = np.cos(np.pi * j / (n - 1))  # [-1,1]
        s = 0.5 * L * (1.0 - x)          # [0,L]
        return np.sort(s)

    # power-law symmetric clustering:
    # construct u in [0,1] with clustering near 0 and 1 via mirrored power transform
    u = np.linspace(0.0, 1.0, n)
    # map half [0,0.5] and mirror
    uh = np.linspace(0.0, 1.0, (n + 1)//2)
    w = uh ** power
    left = 0.5 * w
    right = 1.0 - left[::-1]
    if n % 2 == 0:
        u_sym = np.concatenate([left, right])
    else:
        u_sym = np.concatenate([left, right[1:]])
    return np.sort(L * u_sym)


def _merge_unique_sorted(s_lists: Iterable[np.ndarray], *, s_min: float, s_max: float, tol: float = 1e-12) -> np.ndarray:
    """Merge multiple s-arrays into one sorted unique array."""
    all_s = np.concatenate([np.asarray(a, float).ravel() for a in s_lists]) if s_lists else np.array([], float)
    if all_s.size == 0:
        return np.array([s_min, s_max], float)
    all_s = np.clip(all_s, s_min, s_max)
    all_s.sort()
    # unique with tolerance
    keep = [0]
    for i in range(1, len(all_s)):
        if abs(all_s[i] - all_s[keep[-1]]) > tol:
            keep.append(i)
    return all_s[keep]


class ArcLengthRemeshPolicy:
    """Remesh a (single) polyline crack using global arc-length symmetric placement."""

    def __init__(self, budget: Optional[ArcLengthBudget] = None):
        self.budget = budget or ArcLengthBudget()

    # -----------------------------
    # Public API
    # -----------------------------
    def remesh_network(self, network: CrackNetworkV4, *, ne_half: int) -> CrackNetworkV4:
        """
        Return a new CrackNetworkV4 with re-sampled vertices along the main polyline.

        Parameters
        ----------
        network:
            Current crack network.
        ne_half:
            Half-element budget used by solver. For v1, we interpret a target node count
            N_target = 2*ne_half + 1 along the polyline.

        Notes
        -----
        - v1 supports only a single open chain (2 tips, no junctions).
        """
        chain = self._extract_single_chain(network)
        if chain is None:
            raise NotImplementedError("ArcLengthRemeshPolicy v1 supports only a single open chain (no junctions).")

        tip0, tip1, ordered_vids = chain
        pts = np.array([network.vertex_coords(int(vid)) for vid in ordered_vids], float)
        s_cum, L = _arc_length(pts)
        if L <= 0:
            return network

        N_target = int(2 * int(ne_half) + 1)
        N_target = max(N_target, 5)

        s_nodes = self._build_s_nodes(s_cum, L, pts, N_target)

        xy_new = _interp_polyline(pts, s_nodes, s_cum)

        # rebuild as a straight polyline with new vertices (keeps tip ids by replacing endpoints)
        return self._rebuild_chain_network(network, ordered_vids, xy_new)

    # -----------------------------
    # Internals
    # -----------------------------
    def _build_s_nodes(self, s_cum: np.ndarray, L: float, pts: np.ndarray, N_target: int) -> np.ndarray:
        """Construct node locations along arc-length with feature budgets + bulk guard."""
        b = self.budget
        s_min, s_max = 0.0, float(L)

        # Identify kinks: interior vertices where turn angle is nonzero
        kink_s = self._detect_kinks_s(pts, s_cum)

        # Base symmetric nodes (bulk skeleton)
        # Keep at least bulk_min_panels => bulk_min_nodes = bulk_min_panels + 1
        bulk_min_nodes = int(b.bulk_min_panels) + 1
        bulk_nodes = max(int(np.ceil(b.bulk_frac_min * N_target)), bulk_min_nodes)
        bulk_nodes = min(bulk_nodes, N_target)
        s_bulk = _symmetric_tip_nodes(L, bulk_nodes, cluster="cheb", power=2.0)  # cheb gives robust symmetry

        # Tip clusters: extra nodes near each end in a window
        tip_extra_each = int(np.floor(b.tip_frac * N_target))
        tip_extra_each = max(tip_extra_each, 0)

        # Build clustered windows near tips using the same symmetric construction but restricted
        def tip_window_nodes(side: str) -> np.ndarray:
            if tip_extra_each <= 0:
                return np.array([], float)
            wL = float(b.tip_frac) * L
            n = tip_extra_each
            ss = _symmetric_tip_nodes(wL, n + 2, cluster=b.tip_cluster, power=b.tip_cluster_power)  # includes endpoints
            ss = np.unique(ss)  # monotone
            if side == "left":
                return ss
            else:
                return (L - ss[::-1])

        s_tip_L = tip_window_nodes("left")
        s_tip_R = tip_window_nodes("right")

        # Kink clusters
        kink_lists: List[np.ndarray] = []
        for sk in kink_s:
            # local window around kink: [sk-d, sk+d]
            d = float(b.kink_halfwidth_frac) * L
            n_local = max(int(b.kink_extra_per_kink), 0)
            if n_local <= 0:
                continue
            ss = np.linspace(max(0.0, sk - d), min(L, sk + d), n_local)
            kink_lists.append(ss)

        s_all = _merge_unique_sorted([s_bulk, s_tip_L, s_tip_R] + kink_lists, s_min=s_min, s_max=s_max)

        # If too many nodes (can happen if many kinks), downsample while keeping symmetry:
        if s_all.size > N_target:
            # Keep endpoints; pick the rest by uniform sampling in index space
            keep = np.linspace(0, s_all.size - 1, N_target).round().astype(int)
            keep = np.unique(keep)
            if keep.size < N_target:
                # fill missing indices
                missing = N_target - keep.size
                cand = np.setdiff1d(np.arange(s_all.size), keep)
                keep = np.sort(np.concatenate([keep, cand[:missing]]))
            s_all = s_all[keep]

        # If too few nodes, densify uniformly (preserve endpoints)
        if s_all.size < N_target:
            need = N_target - s_all.size
            extra = np.linspace(0.0, L, need + 2)[1:-1]
            s_all = _merge_unique_sorted([s_all, extra], s_min=s_min, s_max=s_max)

        return np.asarray(s_all, float)

    @staticmethod
    def _detect_kinks_s(pts: np.ndarray, s_cum: np.ndarray, *, angle_tol_deg: float = 1.0) -> List[float]:
        """Detect interior kink locations by turn angle."""
        if len(pts) < 3:
            return []
        ang_tol = np.deg2rad(angle_tol_deg)
        out: List[float] = []
        for i in range(1, len(pts) - 1):
            v0 = pts[i] - pts[i - 1]
            v1 = pts[i + 1] - pts[i]
            n0 = np.hypot(v0[0], v0[1])
            n1 = np.hypot(v1[0], v1[1])
            if n0 < 1e-30 or n1 < 1e-30:
                continue
            c = float(np.dot(v0, v1) / (n0 * n1))
            c = max(-1.0, min(1.0, c))
            theta = float(np.arccos(c))
            if theta > ang_tol:
                out.append(float(s_cum[i]))
        return out

    @staticmethod
    def _extract_single_chain(network: CrackNetworkV4) -> Optional[Tuple[int, int, List[int]]]:
        """Return (tip0, tip1, ordered_vids) for a single open chain, else None."""
        # Compute degrees
        deg = {int(v.id): len(v.edges) for v in network.vertices}
        tips = [vid for vid, d in deg.items() if d == 1]
        if len(tips) != 2:
            return None
        # Reject junctions/branches
        if any(d > 2 for d in deg.values()):
            return None

        tip0, tip1 = tips[0], tips[1]

        # Walk from tip0 to tip1
        ordered = [tip0]
        prev = None
        cur = tip0
        while cur != tip1:
            v = network.V(int(cur))
            # Find neighbor not equal prev
            nbr = None
            for eid in v.edges:
                e = next(ee for ee in network.edges if int(ee.id) == int(eid))
                other = int(e.v1) if int(e.v0) == int(cur) else int(e.v0)
                if prev is None or other != prev:
                    nbr = other
                    break
            if nbr is None:
                return None
            prev, cur = cur, nbr
            ordered.append(cur)
            if len(ordered) > len(network.vertices) + 5:
                return None
        return tip0, tip1, ordered

    @staticmethod
    def _rebuild_chain_network(network: CrackNetworkV4, old_vids: List[int], xy_new: np.ndarray) -> CrackNetworkV4:
        """Rebuild the chain network with new interior vertices, preserving endpoints ids."""
        if xy_new.shape[0] < 2:
            return network

        tip0 = int(old_vids[0])
        tip1 = int(old_vids[-1])

        # Determine new ids for interior vertices
        max_vid = int(max(int(v.id) for v in network.vertices))
        new_vids = [tip0]
        next_id = max_vid + 1
        for _ in range(1, xy_new.shape[0] - 1):
            new_vids.append(next_id)
            next_id += 1
        new_vids.append(tip1)

        # Build vertices
        V: List[VertexV4] = []
        for vid, xy in zip(new_vids, xy_new):
            V.append(VertexV4(int(vid), float(xy[0]), float(xy[1]), [], role="node"))

        # Build edges (line segments)
        max_eid = int(max(int(e.id) for e in network.edges)) if network.edges else -1
        E: List[EdgeV4] = []
        eid = max_eid + 1
        for i in range(len(new_vids) - 1):
            E.append(EdgeV4(int(eid), int(new_vids[i]), int(new_vids[i+1]), kind="line"))
            eid += 1

        # Fill vertex incident edges
        for ee in E:
            v0 = next(vv for vv in V if int(vv.id) == int(ee.v0))
            v1 = next(vv for vv in V if int(vv.id) == int(ee.v1))
            v0.edges.append(int(ee.id))
            v1.edges.append(int(ee.id))

        return CrackNetworkV4(vertices=V, edges=E, Nv_max=network.Nv_max)
