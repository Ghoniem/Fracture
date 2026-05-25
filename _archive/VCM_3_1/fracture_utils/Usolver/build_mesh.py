"""
Mesh / panel allocation utilities for polyline discretization.

This module is intentionally solver-agnostic: it only decides *how many panels*
go on each segment and *where nodes lie* along each segment, including:
- endpoint/junction refinement
- kink refinement (deg==2 internal vertices)
- one-sided or two-sided clustering (fixes asymmetry for single-segment cracks)

Terminology (FEM/BEM-consistent):
- "nodes" are the parametric points s_nodes along a polyline
- "panels" are the intervals between consecutive nodes
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Set
import math
import numpy as np


def identify_refinement_segments(
    vids_path: List[int],
    deg: Dict[int, int],
    nseg: int,
    *,
    refine_junction_endpoints: bool,
    refine_kinks: bool,
) -> Tuple[List[int], Set[int], Dict[int, str]]:
    """
    Returns
    -------
    refined_endpoint_segs : list[int]
        Segment indices that should have endpoint clustering (tips or junction endpoints).
        Typically includes 0 and/or nseg-1.
    kink_adj_segs : set[int]
        Segment indices adjacent to internal deg==2 vertices.
    cluster_target : dict[int,str]
        For kink-adjacent segments only: seg_index -> 'start' or 'end' indicating
        which side of the segment should be clustered toward the kink.
    """
    refined_endpoint_segs: List[int] = []
    kink_adj_segs: Set[int] = set()
    cluster_target: Dict[int, str] = {}

    if nseg <= 0:
        return refined_endpoint_segs, kink_adj_segs, cluster_target

    v_start = int(vids_path[0]) if vids_path else -1
    v_end = int(vids_path[-1]) if vids_path else -1

    if int(deg.get(v_start, 0)) == 1 or (refine_junction_endpoints and int(deg.get(v_start, 0)) >= 3):
        refined_endpoint_segs.append(0)
    if int(deg.get(v_end, 0)) == 1 or (refine_junction_endpoints and int(deg.get(v_end, 0)) >= 3):
        if (nseg - 1) not in refined_endpoint_segs:
            refined_endpoint_segs.append(nseg - 1)

    # Kinks: internal deg==2 vertices along the polyline path.
    # For a kink at vids_path[i], segments (i-1) and (i) are adjacent.
    if refine_kinks and isinstance(vids_path, list) and len(vids_path) >= 3:
        for i in range(1, len(vids_path) - 1):
            vk = int(vids_path[i])
            if int(deg.get(vk, 0)) == 2:
                sL = i - 1
                sR = i
                if 0 <= sL < nseg:
                    kink_adj_segs.add(sL)
                    cluster_target[sL] = "end"    # cluster toward kink
                if 0 <= sR < nseg:
                    kink_adj_segs.add(sR)
                    cluster_target[sR] = "start"  # cluster toward kink

    return refined_endpoint_segs, kink_adj_segs, cluster_target


def allocate_panels_per_segment(
    *,
    Np: int,
    nseg: int,
    refined_endpoint_segs: List[int],
    kink_adj_segs: Set[int],
    endpoint_min_panels: int,
    kink_min_panels: int,
    other_min_panels: int,
) -> List[int]:
    """
    Allocate panel counts Nseg[k] per segment k, such that sum(Nseg)=Np.

    Strategy:
    - Start with other_min_panels on each segment
    - Enforce endpoint_min_panels on refined endpoint segments
    - Enforce kink_min_panels on kink-adjacent segments
    - Distribute remaining panels uniformly across non-special segments, else across specials.
    """
    if nseg <= 0:
        return []

    endpoint_min_panels = max(0, int(endpoint_min_panels))
    kink_min_panels = max(0, int(kink_min_panels))
    other_min_panels = max(0, int(other_min_panels))

    Nseg = [int(other_min_panels) for _ in range(nseg)]

    for k in refined_endpoint_segs:
        if 0 <= k < nseg and endpoint_min_panels > 0:
            Nseg[k] = max(Nseg[k], int(endpoint_min_panels))

    for k in kink_adj_segs:
        if 0 <= k < nseg and kink_min_panels > 0:
            Nseg[k] = max(Nseg[k], int(kink_min_panels))

    N_used = int(sum(Nseg))
    if N_used > int(Np):
        raise ValueError(
            f"Not enough total panels Np={Np} for requested minimums: "
            f"sum(min panels)={N_used}, nseg={nseg}."
        )

    R = int(Np - N_used)
    special = set(refined_endpoint_segs) | set(kink_adj_segs)
    others = [k for k in range(nseg) if k not in special]

    if others:
        q, r = divmod(R, len(others))
        for k in others:
            Nseg[k] += int(q)
        for k in others[:r]:
            Nseg[k] += 1
    elif special:
        spec_list = list(sorted(special))
        q, r = divmod(R, len(spec_list))
        for k in spec_list:
            Nseg[k] += int(q)
        for k in spec_list[:r]:
            Nseg[k] += 1
    # else: single segment and no refinement; nothing to do

    return [int(max(1, nk)) for nk in Nseg]


def _xi_two_sided(u: np.ndarray, *, mode: str, power: float) -> np.ndarray:
    """
    Two-sided clustering on [0,1].
    - cheb: Chebyshev nodes (dense near both ends)
    - power: symmetric power mapping (dense near both ends for power>1)
    """
    mode = str(mode).lower().strip()
    if mode in ("cheb", "chebyshev"):
        th = np.linspace(0.0, math.pi, len(u))
        return 0.5 * (1.0 - np.cos(th))
    # symmetric power map: xi = u^p / (u^p + (1-u)^p)
    p = float(power)
    if not np.isfinite(p) or p <= 1.0:
        p = 2.0
    a = np.power(u, p)
    b = np.power(1.0 - u, p)
    denom = a + b
    # avoid division by 0 at endpoints
    denom = np.where(denom == 0.0, 1.0, denom)
    return a / denom


def segment_parametric_nodes(
    Nk: int,
    *,
    tip_cluster: str,
    tip_cluster_power: float,
    cluster_start: bool,
    cluster_end: bool,
) -> np.ndarray:
    """
    Return xi array of length Nk+1 in [0,1] for a segment.
    Supports:
    - no clustering (uniform)
    - one-sided clustering (start OR end)
    - two-sided clustering (start AND end)  <-- critical bugfix for nseg==1 cracks
    """
    Nk = int(max(1, Nk))
    u = np.linspace(0.0, 1.0, Nk + 1)

    tip_cluster = str(tip_cluster).lower().strip()
    if tip_cluster in ("", "uniform", "none"):
        return u

    # two-sided clustering
    if bool(cluster_start) and bool(cluster_end):
        return _xi_two_sided(u, mode=tip_cluster, power=float(tip_cluster_power))

    # one-sided clustering
    pwr = float(tip_cluster_power)
    if not np.isfinite(pwr) or pwr <= 1.0:
        pwr = 2.0

    if bool(cluster_start):
        if tip_cluster in ("cheb", "chebyshev"):
            th = np.linspace(0.0, math.pi, Nk + 1)
            xi = 0.5 * (1.0 - np.cos(th))
            return 1.0 - xi  # dense near 0
        return np.power(u, pwr)

    if bool(cluster_end):
        if tip_cluster in ("cheb", "chebyshev"):
            th = np.linspace(0.0, math.pi, Nk + 1)
            xi = 0.5 * (1.0 - np.cos(th))
            return xi  # dense near 1
        return 1.0 - np.power(1.0 - u, pwr)

    return u


def build_s_nodes_for_polyline(
    *,
    segL: np.ndarray,
    Nseg: List[int],
    refined_endpoint_segs: List[int],
    kink_adj_segs: Set[int],
    cluster_target: Dict[int, str],
    tip_cluster: str,
    tip_cluster_power: float,
) -> np.ndarray:
    """
    Build concatenated s_nodes for the entire polyline by concatenating each segment's node set.

    Endpoint clustering:
      - segment 0 clusters toward start (polyline start)
      - segment nseg-1 clusters toward end (polyline end)

    Kink clustering:
      - cluster toward kink as specified in cluster_target for those segments

    Two-sided clustering occurs naturally when nseg==1 and both ends are refined.
    """
    segL = np.asarray(segL, float)
    nseg = int(len(segL))
    if nseg <= 0:
        return np.array([0.0, 1.0], float)

    s_nodes_list: List[np.ndarray] = []
    s0 = 0.0

    for k in range(nseg):
        Nk = int(max(1, Nseg[k]))
        Le = float(segL[k])
        if Le <= 0.0:
            continue

        # Determine clustering needs for this segment
        cluster_start = False
        cluster_end = False

        # Kinks can force one-sided clustering toward kink
        if k in cluster_target:
            cluster_start = (cluster_target[k] == "start")
            cluster_end = (cluster_target[k] == "end")
        else:
            if k == 0 and (0 in refined_endpoint_segs):
                cluster_start = True
            if k == nseg - 1 and ((nseg - 1) in refined_endpoint_segs):
                cluster_end = True

        xi = segment_parametric_nodes(
            Nk,
            tip_cluster=tip_cluster,
            tip_cluster_power=tip_cluster_power,
            cluster_start=cluster_start,
            cluster_end=cluster_end,
        )

        s_local = s0 + Le * xi
        if not s_nodes_list:
            s_nodes_list.append(s_local)
        else:
            s_nodes_list.append(s_local[1:])  # avoid duplicating shared boundary node
        s0 += Le

    return np.concatenate(s_nodes_list) if s_nodes_list else np.linspace(0.0, float(np.sum(segL)), int(np.sum(Nseg)) + 1)
