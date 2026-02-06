# fracture_utils/Upropagation/network_ops.py
# Clean, import-safe network operations using Usolver.polyline

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import numpy as np


# -----------------------------------------------------------------------------
# Simple container
# -----------------------------------------------------------------------------

@dataclass
class PolylinePath:
    vids: List[int]


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------

def get_netops():
    try:
        return _UsolverPolylineNetOps()
    except Exception:
        return _FallbackNetOps()


# -----------------------------------------------------------------------------
# Usolver-backed implementation
# -----------------------------------------------------------------------------

class _UsolverPolylineNetOps:
    def __init__(self):
        from fracture_utils.Usolver import polyline
        self.pl = polyline

    # ---------------------------------
    # Degrees
    # ---------------------------------
    def degree_map(self, network) -> Dict[int, int]:
        if hasattr(self.pl, "vertex_degrees"):
            return {int(k): int(v) for k, v in self.pl.vertex_degrees(network).items()}
        return {int(v.id): len(v.edges) for v in network.vertices}

    # ---------------------------------
    # Open polylines
    # ---------------------------------
    def extract_open_polylines(self, network) -> List[PolylinePath]:
        comps = self.pl.find_polyline_components(network)
        out: List[PolylinePath] = []

        for comp in comps:
            # normalize component → edge ids
            if isinstance(comp, dict):
                edges = comp.get("edges")
            elif isinstance(comp, (tuple, list)):
                edges = comp[0]
            else:
                edges = comp

            if edges is None:
                continue

            vids, _ = self.pl.path_order_for_component(network, edges)
            if vids is None or len(vids) < 2:
                continue

            out.append(PolylinePath(list(map(int, vids))))

        return out

    # ---------------------------------
    # Degree-1 tips
    # ---------------------------------
    def extract_deg1_tips(
        self,
        network,
        polylines: Sequence[PolylinePath],
        deg: Dict[int, int],
    ):
        from .tip_state import TipState, TipID

        def xy(vid):
            return np.asarray(network.vertex_coords(int(vid)), float)

        tips = []

        for pid, p in enumerate(polylines):
            vids = p.vids
            if len(vids) < 2:
                continue

            # total polyline length
            L = sum(
                np.linalg.norm(xy(a) - xy(b))
                for a, b in zip(vids[:-1], vids[1:])
            )

            # start and end
            for which, v_tip, v_in in (
                ("start", vids[0], vids[1]),
                ("end", vids[-1], vids[-2]),
            ):
                if deg.get(int(v_tip), 0) != 1:
                    continue

                t = xy(v_tip) - xy(v_in)
                n = np.linalg.norm(t)
                if n == 0.0:
                    continue

                tips.append(
                    TipState(
                        tip_id=TipID(pid=pid, which=which),
                        v_tip=int(v_tip),
                        x_tip=xy(v_tip),
                        t_hat=t / n,
                        total_length=float(L),
                    )
                )

        return tips


# -----------------------------------------------------------------------------
# Conservative fallback (single straight crack safe)
# -----------------------------------------------------------------------------

class _FallbackNetOps:
    def degree_map(self, network):
        return {int(v.id): len(v.edges) for v in network.vertices}

    def extract_open_polylines(self, network):
        vids = [int(v.id) for v in network.vertices]
        return [PolylinePath(vids)] if len(vids) >= 2 else []

    def extract_deg1_tips(self, network, polylines, deg):
        from .tip_state import TipState, TipID

        def xy(vid):
            return np.asarray(network.vertex_coords(int(vid)), float)

        tips = []
        for p in polylines:
            vids = p.vids
            if len(vids) < 2:
                continue

            L = np.linalg.norm(xy(vids[0]) - xy(vids[-1]))

            for which, v_tip, v_in in (
                ("start", vids[0], vids[1]),
                ("end", vids[-1], vids[-2]),
            ):
                if deg.get(v_tip, 0) != 1:
                    continue
                t = xy(v_tip) - xy(v_in)
                n = np.linalg.norm(t)
                if n == 0:
                    continue
                tips.append(
                    TipState(
                        tip_id=TipID(pid=0, which=which),
                        v_tip=v_tip,
                        x_tip=xy(v_tip),
                        t_hat=t / n,
                        total_length=float(L),
                    )
                )
        return tips
