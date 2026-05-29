"""Tip identification and local geometry extraction.

This module should not call the solver. It should:
- identify degree-1 vertices (tips)
- compute local tangents at those tips
- provide a stable identifier for each propagating tip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fracture_utils.Usolver.network import CrackNetworkV4

TipEnd = Literal["start", "end"]

@dataclass(frozen=True)
class TipID:
    """Stable identifier for a tip within a polyline record."""
    pid: int
    which: TipEnd  # 'start' or 'end'

@dataclass(frozen=True)
class TipState:
    """Geometric state at a crack tip."""
    tip_id: TipID
    v_tip: int
    x_tip: np.ndarray          # shape (2,)
    t_hat: np.ndarray          # unit tangent, shape (2,)
    total_length: float        # total crack length (polyline total or network total)

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(2,)
    n = float(np.hypot(v[0], v[1]))
    if n <= 0.0:
        return np.array([1.0, 0.0], float)
    return v / n

def extract_tips_from_polylines(
    network: "CrackNetworkV4",
    polylines: list[dict],
    deg: dict[int, int],
    *,
    total_length: float,
    smooth_tangent: bool = True,
) -> list[TipState]:
    """Extract deg-1 tips from parametrized polylines.

    Parameters
    ----------
    network
        CrackNetworkV4 instance.
    polylines
        List of polyline metadata dicts (from build_polylines_*).
    deg
        Vertex degrees (from build_geometry.vertex_degrees).
    total_length
        Total crack length used to set Δa = f * total_length.
    smooth_tangent
        If True, compute tangent using up to the first/last two segments (reduces noise).

    Returns
    -------
    tips : list[TipState]
    """
    tips: list[TipState] = []
    for pid, p in enumerate(polylines):
        kind = str(p.get("kind", "polyline")).lower().strip()
        if kind not in ("polyline", "arc", "cspline"):
            continue

        v_start = int(p.get("v_start", -1))
        v_end   = int(p.get("v_end", -1))

        # Start tip
        if int(deg.get(v_start, 0)) == 1:
            x0 = np.asarray(network.vertex_coords(v_start), float).reshape(2,)
            # tangent: from first segment direction
            t = _polyline_endpoint_tangent(network, p, which="start", smooth=smooth_tangent)
            tips.append(TipState(TipID(int(pid), "start"), v_start, x0, unit(t), float(total_length)))

        # End tip
        if int(deg.get(v_end, 0)) == 1:
            x1 = np.asarray(network.vertex_coords(v_end), float).reshape(2,)
            t = _polyline_endpoint_tangent(network, p, which="end", smooth=smooth_tangent)
            tips.append(TipState(TipID(int(pid), "end"), v_end, x1, unit(t), float(total_length)))

    return tips

def _polyline_endpoint_tangent(network: "CrackNetworkV4", p: dict, *, which: TipEnd, smooth: bool) -> np.ndarray:
    """Compute an endpoint tangent for a (polyline/arc/cspline) record.

    For polyline: tangent from first/last segment (optionally averaged over 2 segments).
    For arc/cspline: tangent is not directly available here; we approximate using endpoint chord.
    """
    kind = str(p.get("kind", "polyline")).lower().strip()
    if kind == "polyline":
        vids = list(p.get("path_vertex_ids", []))
        if len(vids) < 2:
            return np.array([1.0, 0.0], float)

        if which == "start":
            p0 = np.asarray(network.vertex_coords(int(vids[0])), float).reshape(2,)
            p1 = np.asarray(network.vertex_coords(int(vids[1])), float).reshape(2,)
            t1 = p1 - p0
            if smooth and len(vids) >= 3:
                p2 = np.asarray(network.vertex_coords(int(vids[2])), float).reshape(2,)
                t2 = p2 - p1
                return unit(t1) + unit(t2)
            return t1

        # end
        pN = np.asarray(network.vertex_coords(int(vids[-1])), float).reshape(2,)
        pNm1 = np.asarray(network.vertex_coords(int(vids[-2])), float).reshape(2,)
        t1 = pN - pNm1
        if smooth and len(vids) >= 3:
            pNm2 = np.asarray(network.vertex_coords(int(vids[-3])), float).reshape(2,)
            t2 = pNm1 - pNm2
            return unit(t2) + unit(t1)
        return t1

    # arc/cspline fallback: chord direction
    v0 = int(p.get("v_start", -1))
    v1 = int(p.get("v_end", -1))
    if v0 >= 0 and v1 >= 0:
        x0 = np.asarray(network.vertex_coords(v0), float).reshape(2,)
        x1 = np.asarray(network.vertex_coords(v1), float).reshape(2,)
        chord = (x1 - x0) if which == "start" else (x0 - x1)
        return chord
    return np.array([1.0, 0.0], float)
