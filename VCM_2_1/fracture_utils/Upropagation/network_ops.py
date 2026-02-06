
"""
Upropagation.network_ops

Goal: centralize all CrackNetworkV4 topology operations (degree map, polyline extraction,
tip identification) behind a single interface.

Design:
- Prefer using fracture_utils.Ugenerator (user's network generator package) if present.
- Fall back to a small internal implementation for the common case of a single open polyline crack.

This avoids re-implementing network logic across Upropagation modules and prevents API drift
(e.g., differing extract_tips_from_polylines signatures).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib

import numpy as np


@dataclass
class PolylinePath:
    """A polyline path represented as an ordered list of vertex ids."""
    vids: List[int]


class NetworkOpsBackend:
    """Abstract backend for network topology operations."""
    def degree_map(self, network) -> Dict[int, int]:
        raise NotImplementedError

    def extract_open_polylines(self, network) -> List[PolylinePath]:
        raise NotImplementedError

    def extract_deg1_tips(self, network, polylines: List[PolylinePath], deg: Dict[int, int]):
        """Return list of TipState objects compatible with Upropagation.tip_state."""
        raise NotImplementedError


class InternalNetworkOps(NetworkOpsBackend):
    """
    Minimal internal backend.

    Supports:
    - networks consisting of one or more disjoint open chains made of line edges
      with no junctions (deg<=2).
    """
    def degree_map(self, network) -> Dict[int, int]:
        deg: Dict[int, int] = {}
        for v in network.vertices:
            deg[int(v.id)] = len(getattr(v, "edges", []) or [])
        return deg

    def extract_open_polylines(self, network) -> List[PolylinePath]:
        # Build adjacency (vertex -> neighbor vertices) for line edges
        adj: Dict[int, List[int]] = {}
        for e in network.edges:
            v0 = int(e.v0); v1 = int(e.v1)
            adj.setdefault(v0, []).append(v1)
            adj.setdefault(v1, []).append(v0)

        deg = self.degree_map(network)

        # Find endpoints (deg==1) and walk chains
        endpoints = [vid for vid, d in deg.items() if d == 1]
        visited_edges = set()
        paths: List[PolylinePath] = []

        # Helper to mark an undirected edge
        def edge_key(a: int, b: int) -> Tuple[int, int]:
            return (a, b) if a < b else (b, a)

        for start in endpoints:
            # If start already in a built path, skip
            # We'll check by seeing whether its incident edge has been visited
            nbrs = adj.get(start, [])
            if not nbrs:
                continue
            if edge_key(start, nbrs[0]) in visited_edges:
                continue

            vids = [start]
            prev = None
            cur = start
            while True:
                nbrs = adj.get(cur, [])
                # choose next not equal prev
                nxt = None
                for nb in nbrs:
                    if prev is None or nb != prev:
                        # avoid traversing an already visited edge unless no choice
                        if prev is None or edge_key(cur, nb) not in visited_edges:
                            nxt = nb
                            break
                if nxt is None:
                    break
                visited_edges.add(edge_key(cur, nxt))
                vids.append(int(nxt))
                prev, cur = cur, int(nxt)
                if deg.get(cur, 0) == 1:
                    break
            if len(vids) >= 2:
                paths.append(PolylinePath(vids=vids))
        return paths

    def extract_deg1_tips(self, network, polylines: List[PolylinePath], deg: Dict[int, int]):
        from .tip_state import TipID, TipState  # local import

        tips: List[TipState] = []
        for pid, pl in enumerate(polylines):
            if len(pl.vids) < 2:
                continue
            v_start = int(pl.vids[0])
            v_end   = int(pl.vids[-1])

            # Start tip
            if deg.get(v_start, 0) == 1:
                x_tip = network.vertex_coords(v_start)
                x_nbr = network.vertex_coords(int(pl.vids[1]))
                t_out = np.asarray(x_tip - x_nbr, float)
                n = float(np.hypot(t_out[0], t_out[1]))
                t_out = t_out / (n if n > 0 else 1.0)
                tips.append(
                    TipState(
                        tip_id=TipID(pid=pid, which="start"),
                        vid=int(v_start),
                        x_tip=np.asarray(x_tip, float),
                        t_hat=np.asarray(t_out, float),
                    )
                )
            # End tip
            if deg.get(v_end, 0) == 1:
                x_tip = network.vertex_coords(v_end)
                x_nbr = network.vertex_coords(int(pl.vids[-2]))
                t_out = np.asarray(x_tip - x_nbr, float)
                n = float(np.hypot(t_out[0], t_out[1]))
                t_out = t_out / (n if n > 0 else 1.0)
                tips.append(
                    TipState(
                        tip_id=TipID(pid=pid, which="end"),
                        vid=int(v_end),
                        x_tip=np.asarray(x_tip, float),
                        t_hat=np.asarray(t_out, float),
                    )
                )
        return tips


class UgeneratorNetworkOps(NetworkOpsBackend):
    """
    Adapter around the user's Ugenerator package.

    Since Ugenerator module layout may vary, we try a small list of likely module paths.
    You can extend 'CANDIDATE_MODULES' if needed.
    """
    CANDIDATE_MODULES = [
        "fracture_utils.Ugenerator.topology",
        "fracture_utils.Ugenerator.network_ops",
        "Ugenerator.topology",
        "Ugenerator.network_ops",
    ]

    def __init__(self):
        self.mod = None
        last_err = None
        for m in self.CANDIDATE_MODULES:
            try:
                self.mod = importlib.import_module(m)
                break
            except Exception as e:
                last_err = e
        if self.mod is None:
            raise ImportError(f"Could not import Ugenerator topology backend. Last error: {last_err}")

    def degree_map(self, network) -> Dict[int, int]:
        if hasattr(self.mod, "degree_map"):
            return {int(k): int(v) for k, v in self.mod.degree_map(network).items()}
        # Fallback: infer from network
        deg: Dict[int, int] = {}
        for v in network.vertices:
            deg[int(v.id)] = len(getattr(v, "edges", []) or [])
        return deg

    def extract_open_polylines(self, network) -> List[PolylinePath]:
        # Prefer a function that returns vertex-id sequences
        if hasattr(self.mod, "extract_open_polylines"):
            pls = self.mod.extract_open_polylines(network)
            out: List[PolylinePath] = []
            for pl in pls:
                vids = list(getattr(pl, "vids", pl))
                out.append(PolylinePath(vids=[int(x) for x in vids]))
            return out
        if hasattr(self.mod, "polylines_from_network"):
            pls = self.mod.polylines_from_network(network)
            out: List[PolylinePath] = []
            for pl in pls:
                vids = list(getattr(pl, "vids", pl))
                out.append(PolylinePath(vids=[int(x) for x in vids]))
            return out
        # Last resort: use internal approach
        return InternalNetworkOps().extract_open_polylines(network)

    def extract_deg1_tips(self, network, polylines: List[PolylinePath], deg: Dict[int, int]):
        # Prefer a direct tip extractor that returns TipState or equivalent
        if hasattr(self.mod, "extract_deg1_tips"):
            return self.mod.extract_deg1_tips(network, polylines, deg)
        # Otherwise, reuse internal TipState builder (outward tangent)
        return InternalNetworkOps().extract_deg1_tips(network, polylines, deg)


def get_network_ops(prefer_ugenerator: bool = True) -> NetworkOpsBackend:
    if prefer_ugenerator:
        try:
            return UgeneratorNetworkOps()
        except Exception:
            return InternalNetworkOps()
    return InternalNetworkOps()
