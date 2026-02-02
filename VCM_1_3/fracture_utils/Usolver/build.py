"""
Helper module: polyline construction, discretization, operator assembly for solver_parametrized_half.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

# ============================================================
# Arc polyline support (first-class edge geometry)
# ============================================================
def make_arc_polyline(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    *,
    crack_mode: str = "full",
    v_start: int = -1,
    v_end: int = -1,
    ccw: bool | None = None,
) -> dict:
    """Create a polyline dict representing a circular arc.

    Parameters
    ----------
    p0, p1 : array-like (2,)
        Endpoints on the arc (start and end).
    center : array-like (2,)
        Circle center.
    ccw : bool | None
        If None, choose the signed shortest arc from p0 to p1.
        If True/False, force counterclockwise/clockwise direction.

    Returns
    -------
    dict
        Polyline record compatible with discretize_polylines().
    """
    p0 = np.asarray(p0, float).reshape(2)
    p1 = np.asarray(p1, float).reshape(2)
    c  = np.asarray(center, float).reshape(2)

    r0 = float(np.linalg.norm(p0 - c))
    r1 = float(np.linalg.norm(p1 - c))
    if r0 <= 0.0:
        raise ValueError("Arc radius must be positive.")
    if abs(r1 - r0) > 1e-8 * max(r0, 1.0):
        raise ValueError(f"Arc endpoints are not on the same circle: r0={r0}, r1={r1}.")

    th0 = float(np.arctan2(p0[1] - c[1], p0[0] - c[0]))
    th1 = float(np.arctan2(p1[1] - c[1], p1[0] - c[0]))

    # Wrap delta into (-pi, pi]
    dth = (th1 - th0 + np.pi) % (2 * np.pi) - np.pi
    if ccw is True and dth < 0:
        dth += 2 * np.pi
    elif ccw is False and dth > 0:
        dth -= 2 * np.pi

    L = abs(dth) * r0
    if L <= 0.0:
        raise ValueError("Degenerate arc length (p0 == p1 or invalid direction).")

    return dict(
        kind="arc",
        crack_mode=str(crack_mode),
        arc_center=c,
        arc_radius=r0,
        arc_theta0=th0,
        arc_dtheta=dth,
        arc_p0=p0,
        arc_p1=p1,
        total_length=float(L),
        segment_lengths=[float(L)],
        path_vertex_ids=[],   # unused for arc
        path_edge_indices=[], # may be filled by caller
        v_start=int(v_start),
        v_end=int(v_end),
    )


def _arc_point_and_frame(p: dict, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate arc position x(s), unit tangent t(s), unit normal n(s) for s in [0,L]."""
    c = np.asarray(p["arc_center"], float).reshape(2)
    r = float(p["arc_radius"])
    th0 = float(p["arc_theta0"])
    dth = float(p["arc_dtheta"])
    L = float(p["total_length"])

    ss = float(np.clip(s, 0.0, L))
    th = th0 + dth * (ss / L)

    x = c + r * np.array([np.cos(th), np.sin(th)], float)

    sgn = 1.0 if dth >= 0 else -1.0
    t = sgn * np.array([-np.sin(th), np.cos(th)], float)
    t /= max(np.linalg.norm(t), 1e-300)

    n = np.array([-t[1], t[0]], float)
    n /= max(np.linalg.norm(n), 1e-300)

    return x, t, n

# ============================================================
# Cubic spline (natural) support (interpolating through vertices)
# ============================================================
def _chord_length_param(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    if t[-1] <= 0:
        return t
    return t / t[-1]

def _cubic_spline_natural(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Natural cubic spline second derivatives M at nodes."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 3:
        return np.zeros(n, float)
    h = np.diff(t)
    A = np.zeros((n-2, n-2), float)
    rhs = np.zeros((n-2,), float)
    for i in range(1, n-1):
        hi1 = h[i-1]
        hi  = h[i]
        row = i-1
        if row-1 >= 0:
            A[row, row-1] = hi1
        A[row, row] = 2.0*(hi1+hi)
        if row+1 <= n-3:
            A[row, row+1] = hi
        rhs[row] = 6.0*((y[i+1]-y[i])/hi - (y[i]-y[i-1])/hi1)
    M_inner = np.linalg.solve(A, rhs) if (n-2) > 0 else np.array([], float)
    M = np.zeros(n, float)
    M[1:n-1] = M_inner
    return M

def _eval_cubic_spline(t: np.ndarray, y: np.ndarray, M: np.ndarray, tt: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float); y = np.asarray(y, float); M = np.asarray(M, float); tt = np.asarray(tt, float)
    idx = np.searchsorted(t, tt, side="right") - 1
    idx = np.clip(idx, 0, len(t)-2)
    h = t[idx+1] - t[idx]
    a = (t[idx+1] - tt) / h
    b = (tt - t[idx]) / h
    S = (a*y[idx] + b*y[idx+1] + ((a**3-a)*M[idx] + (b**3-b)*M[idx+1])*(h**2)/6.0)
    return S

def _eval_cubic_spline_deriv(t: np.ndarray, y: np.ndarray, M: np.ndarray, tt: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float); y = np.asarray(y, float); M = np.asarray(M, float); tt = np.asarray(tt, float)
    idx = np.searchsorted(t, tt, side="right") - 1
    idx = np.clip(idx, 0, len(t)-2)
    h = t[idx+1] - t[idx]
    a = (t[idx+1] - tt) / h
    b = (tt - t[idx]) / h
    dS = (y[idx+1] - y[idx]) / h + (h/6.0) * (-(3*a*a - 1.0)*M[idx] + (3*b*b - 1.0)*M[idx+1])
    return dS

def make_cspline_polyline_from_path(
    pts: np.ndarray,
    vids_path: list[int],
    eidx_path: list[int],
    *,
    crack_mode: str,
    v_start: int,
    v_end: int,
    n_len_samples: int = 2000,
) -> dict:
    """Build a 'cspline' polyline record that interpolates given points."""
    pts = np.asarray(pts, float)
    if pts.shape[0] < 3:
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))
        return dict(
            kind="polyline",
            path_vertex_ids=list(vids_path),
            path_edge_indices=list(eidx_path),
            segment_lengths=list(map(float, segL)),
            total_length=L,
            crack_mode=str(crack_mode),
            v_start=int(v_start),
            v_end=int(v_end),
        )

    t = _chord_length_param(pts)
    Mx = _cubic_spline_natural(t, pts[:, 0])
    My = _cubic_spline_natural(t, pts[:, 1])

    tt = np.linspace(t[0], t[-1], max(50, int(n_len_samples)))
    xx = _eval_cubic_spline(t, pts[:, 0], Mx, tt)
    yy = _eval_cubic_spline(t, pts[:, 1], My, tt)
    dP = np.diff(np.c_[xx, yy], axis=0)
    L = float(np.sum(np.hypot(dP[:, 0], dP[:, 1])))

    return dict(
        kind="cspline",
        crack_mode=str(crack_mode),
        path_vertex_ids=list(vids_path),
        path_edge_indices=list(eidx_path),
        segment_lengths=[],
        total_length=L,
        v_start=int(v_start),
        v_end=int(v_end),
        spline_t=t,
        spline_x=pts[:, 0].copy(),
        spline_y=pts[:, 1].copy(),
        spline_Mx=Mx,
        spline_My=My,
    )

def _cspline_point_and_frame(p: dict, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = float(p.get("total_length", 0.0))
    if L <= 0:
        return np.array([0.0, 0.0], float), np.array([1.0, 0.0], float), np.array([0.0, 1.0], float)

    ss = float(np.clip(s, 0.0, L))
    tt = ss / L  # approximate arc-length parameter in [0,1]

    t_nodes = np.asarray(p["spline_t"], float)
    x_nodes = np.asarray(p["spline_x"], float)
    y_nodes = np.asarray(p["spline_y"], float)
    Mx = np.asarray(p["spline_Mx"], float)
    My = np.asarray(p["spline_My"], float)

    x = float(_eval_cubic_spline(t_nodes, x_nodes, Mx, np.array([tt]))[0])
    y = float(_eval_cubic_spline(t_nodes, y_nodes, My, np.array([tt]))[0])
    dx = float(_eval_cubic_spline_deriv(t_nodes, x_nodes, Mx, np.array([tt]))[0])
    dy = float(_eval_cubic_spline_deriv(t_nodes, y_nodes, My, np.array([tt]))[0])

    tvec = np.array([dx, dy], float)
    nrm = float(np.linalg.norm(tvec))
    if nrm <= 0:
        tvec = np.array([1.0, 0.0], float); nrm = 1.0
    tvec /= nrm
    nvec = np.array([-tvec[1], tvec[0]], float)
    return np.array([x, y], float), tvec, nvec
import math

from .network import CrackNetworkV4
from .material import Material, AppliedStress
from .solve_kernels import stress_edge_dislocation
from .polyline import *

def _edge_kind(e) -> str:
    return str(getattr(e, 'kind', 'line')).lower().strip()

def _edge_ctrl_vids(e):
    return tuple(getattr(e, 'ctrl_vids', ()))


def vertex_degrees(network: CrackNetworkV4) -> dict[int, int]:
    deg = {int(v.id): 0 for v in network.vertices}
    for e in network.edges:
        deg[int(e.v0)] = deg.get(int(e.v0), 0) + 1
        deg[int(e.v1)] = deg.get(int(e.v1), 0) + 1
    return deg


def build_polylines_full(network: CrackNetworkV4) -> Tuple[List[dict], dict[int, int]]:

    comps = find_polyline_components(network)
    if not comps:
        raise ValueError("No edges in network.")

    # Backward compatible: if every edge is a straight 'line', use the legacy polyline-component logic.
    if not all(_edge_kind(e) == "line" for e in network.edges):
        # Mixed or curved edges: build one polyline per edge (no multi-edge chaining yet).
        polylines: List[dict] = []
        edge_to_polyline: dict[int, int] = {}
        for pid, e in enumerate(network.edges):
            k = _edge_kind(e)
            v0 = int(e.v0); v1 = int(e.v1)
            p0 = network.vertex_coords(v0)
            p1 = network.vertex_coords(v1)
            eid = int(getattr(e, "id", pid))
            if k == "arc":
                ctrls = _edge_ctrl_vids(e)
                if len(ctrls) != 1:
                    raise ValueError(f"Arc edge {eid} must have exactly one ctrl_vid (center).")
                c = network.vertex_coords(int(ctrls[0]))
                polylines.append(make_arc_polyline(p0, p1, c, crack_mode="full", v_start=v0, v_end=v1))
            else:
                seg = p1 - p0
                segL = float(np.sqrt(np.sum(seg * seg)))
                polylines.append(dict(
                    kind="polyline",
                    path_vertex_ids=[v0, v1],
                    path_edge_indices=[eid],
                    segment_lengths=[segL],
                    total_length=segL,
                    crack_mode="full",
                    v_start=v0,
                    v_end=v1,
                ))
            edge_to_polyline[eid] = int(pid)
        return polylines, edge_to_polyline

    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}

    for pid, comp in enumerate(comps):
        vids_path, eidx_path = path_order_for_component(network, comp["edges"])
        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))

        for eidx in eidx_path:
            edge_to_polyline[int(eidx)] = int(pid)

        polylines.append(dict(
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(i) for i in eidx_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="full",
            v_start=int(vids_path[0]),
            v_end=int(vids_path[-1]),
        ))
    return polylines, edge_to_polyline


def build_polylines_half_branches(
    network: CrackNetworkV4,
    deg: dict[int, int],
) -> Tuple[List[dict], dict[int, int]]:
    """Build maximal half-crack 'branches' as polylines that may span multiple edges.

    A polyline (branch) is a maximal path whose interior vertices have degree 2, and whose
    endpoints have degree != 2 (tips deg=1, junctions deg>=3). Closed loops (all deg=2)
    are handled in a second pass.

    Returns
    -------
    polylines : list[dict]
        Each dict contains:
          - path_vertex_ids: ordered vertex ids along the branch
          - path_edge_indices: ordered edge indices along the branch
          - segment_lengths: list of segment lengths along the branch
          - total_length: total arc length
          - v_start, v_end: endpoint vertex ids (same for loops)
          - deg_start, deg_end: degrees of endpoints
    edge_to_polyline : dict[int,int]
        Map from network edge index to polyline id (pid).
    """
    # Backward compatible: if every edge is a straight 'line', use the legacy half-branch chaining logic.
    if not all(_edge_kind(e) == "line" for e in network.edges):
        polylines: List[dict] = []
        edge_to_polyline: dict[int, int] = {}
        pid = 0
        for e in network.edges:
            k = _edge_kind(e)
            v0 = int(e.v0); v1 = int(e.v1)
            p0 = network.vertex_coords(v0)
            p1 = network.vertex_coords(v1)
            eid = int(getattr(e, "id", pid))
            if k == "arc":
                ctrls = _edge_ctrl_vids(e)
                if len(ctrls) != 1:
                    raise ValueError(f"Arc edge {eid} must have exactly one ctrl_vid (center).")
                c = network.vertex_coords(int(ctrls[0]))
                polylines.append(make_arc_polyline(p0, p1, c, crack_mode="half", v_start=v0, v_end=v1))
            else:
                seg = p1 - p0
                segL = float(np.sqrt(np.sum(seg * seg)))
                polylines.append(dict(
                    kind="polyline",
                    path_vertex_ids=[v0, v1],
                    path_edge_indices=[eid],
                    segment_lengths=[segL],
                    total_length=segL,
                    crack_mode="half",
                    v_start=v0,
                    v_end=v1,
                    deg_start=int(deg.get(v0, 0)),
                    deg_end=int(deg.get(v1, 0)),
                ))
            edge_to_polyline[eid] = int(pid)
            pid += 1
        if not polylines:
            raise ValueError("No valid half polylines were built.")
        return polylines, edge_to_polyline
    v2e: dict[int, List[int]] = {}
    for eidx, e in enumerate(network.edges):
        a = int(e.v0); b = int(e.v1)
        v2e.setdefault(a, []).append(int(eidx))
        v2e.setdefault(b, []).append(int(eidx))

    def other_vertex(eidx: int, v: int) -> int:
        e = network.edges[int(eidx)]
        a = int(e.v0); b = int(e.v1)
        return b if a == v else a

    used_edges: set[int] = set()
    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}
    pid = 0

    # Terminal vertices start branches: deg != 2 (tips, junctions, isolated)
    start_verts = sorted([int(v) for v, d in deg.items() if int(d) != 2])

    for v0 in start_verts:
        inc_edges = v2e.get(int(v0), [])
        for e0 in inc_edges:
            e0 = int(e0)
            if e0 in used_edges:
                continue

            vids_path: List[int] = [int(v0)]
            eidx_path: List[int] = []
            cur_v = int(v0)
            cur_e = int(e0)

            while True:
                if cur_e in used_edges:
                    break
                used_edges.add(cur_e)
                eidx_path.append(int(cur_e))

                nxt_v = other_vertex(cur_e, cur_v)
                vids_path.append(int(nxt_v))

                dnext = int(deg.get(int(nxt_v), 0))
                if dnext != 2:
                    break

                # continue along the unique remaining unused incident edge at a deg-2 vertex
                inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
                if not inc:
                    break

                # In a clean deg-2 chain, there should be exactly one unused incident edge.
                # If there are more (deg mismatch), pick the first deterministically.
                cur_v = int(nxt_v)
                cur_e = int(inc[0])

            pts = np.array([network.vertex_coords(v) for v in vids_path], float)
            seg = pts[1:] - pts[:-1]
            segL = np.sqrt(np.sum(seg * seg, axis=1))
            L = float(np.sum(segL))
            if L <= 0:
                continue

            v_start = int(vids_path[0])
            v_end = int(vids_path[-1])

            # Orientation policy: keep tips at the start when possible
            if int(deg.get(v_start, 0)) == 1 and int(deg.get(v_end, 0)) > 1:
                vids_path = list(reversed(vids_path))
                eidx_path = list(reversed(eidx_path))
                segL = segL[::-1].copy()
                v_start = int(vids_path[0])
                v_end = int(vids_path[-1])

            for eidx in eidx_path:
                edge_to_polyline[int(eidx)] = int(pid)

            polylines.append(dict(
                path_vertex_ids=[int(v) for v in vids_path],
                path_edge_indices=[int(i) for i in eidx_path],
                segment_lengths=[float(x) for x in segL.tolist()],
                total_length=float(L),
                crack_mode="half",
                v_start=int(v_start),
                v_end=int(v_end),
                deg_start=int(deg.get(v_start, 0)),
                deg_end=int(deg.get(v_end, 0)),
            ))
            pid += 1

    # Second pass: handle closed loops (all vertices deg=2)
    for eidx0, e0 in enumerate(network.edges):
        eidx0 = int(eidx0)
        if eidx0 in used_edges:
            continue

        v0 = int(e0.v0)
        vids_path: List[int] = [v0]
        eidx_path: List[int] = []

        cur_v = v0
        cur_e = eidx0

        while True:
            if cur_e in used_edges:
                break
            used_edges.add(cur_e)
            eidx_path.append(int(cur_e))

            nxt_v = other_vertex(cur_e, cur_v)
            vids_path.append(int(nxt_v))

            if nxt_v == v0:
                break

            inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
            if not inc:
                break

            cur_v = int(nxt_v)
            cur_e = int(inc[0])

        # Require a closed loop
        if vids_path[-1] != v0:
            continue

        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))
        if L <= 0:
            continue

        for eidx in eidx_path:
            edge_to_polyline[int(eidx)] = int(pid)

        polylines.append(dict(
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(i) for i in eidx_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="half",
            v_start=int(v0),
            v_end=int(v0),
            deg_start=2,
            deg_end=2,
        ))
        pid += 1

    if not polylines:
        raise ValueError("No valid half polylines were built.")
    return polylines, edge_to_polyline




def discretize_polylines(
    network: CrackNetworkV4,
    polylines: List[dict],
    *,
    ne_half: int,
    node_distribution: str,
    collocation_mode: str,
    representation: str,
    nq_stress: int,
    tip_min_nodes: int = 0,
    tip_cluster: str = "power",
    tip_cluster_power: float = 2.0,
    other_min_panels: int = 1,
) -> List[dict]:
    use_tip_singular = (str(representation).lower().strip() == "singular")

    deg = vertex_degrees(network)
    tip_min_nodes = max(0, int(tip_min_nodes))
    tip_min_panels = max(0, tip_min_nodes - 1)
    other_min_panels = max(0, int(other_min_panels))
    tip_cluster = str(tip_cluster).lower().strip()
    tip_cluster_power = float(tip_cluster_power)
    if not np.isfinite(tip_cluster_power) or tip_cluster_power <= 1.0:
        tip_cluster_power = 2.0

    poly_panels: List[dict] = []
    for pid, p in enumerate(polylines):
        vids_path = p["path_vertex_ids"]
        segL = np.array(p["segment_lengths"], float)
        L = float(p["total_length"])

        # Total panels for this polyline (keep legacy scaling: ~2*ne_half for 'half' and 'full')
        Np = max(8, int(2 * ne_half))

        # --- Option A: redistribute panels across segments, enforcing extra resolution on tip segments (deg-1 vertices)
        nseg = int(len(segL))
        if nseg <= 0:
            continue

        v_start = int(p.get('v_start', vids_path[0] if len(vids_path) else -1))
        v_end   = int(p.get('v_end',   vids_path[-1] if len(vids_path) else -1))

        tip_segs = []
        if int(deg.get(v_start, 0)) == 1:
            tip_segs.append(0)
        if int(deg.get(v_end, 0)) == 1 and (nseg - 1) not in tip_segs:
            tip_segs.append(nseg - 1)

        # Base allocation: at least other_min_panels per segment
        Nseg = [int(other_min_panels) for _ in range(nseg)]
        for k in tip_segs:
            if tip_min_panels > 0:
                Nseg[k] = max(Nseg[k], int(tip_min_panels))

        N_used = int(sum(Nseg))
        if N_used > Np:
            raise ValueError(
                f"ne_half={ne_half} too small for tip_min_nodes={tip_min_nodes} and nseg={nseg}. "
                f"Need at least total_panels >= {N_used}, but have {Np}."
            )

        R = int(Np - N_used)
        others = [k for k in range(nseg) if k not in tip_segs]
        if others:
            q, r = divmod(R, len(others))
            for k in others:
                Nseg[k] += int(q)
            for k in others[:r]:
                Nseg[k] += 1
        elif tip_segs:
            q, r = divmod(R, len(tip_segs))
            for k in tip_segs:
                Nseg[k] += int(q)
            for k in tip_segs[:r]:
                Nseg[k] += 1
        # else: single-seg polyline with no tips? nothing to distribute

        # Build s_nodes by concatenating per-segment node sets
        s_nodes_list = []
        s0 = 0.0
        for k in range(nseg):
            Nk = int(max(1, Nseg[k]))
            Le = float(segL[k])
            if Le <= 0.0:
                continue

            # local param in [0,1] with optional one-sided clustering at deg-1 tip
            if k in tip_segs and tip_cluster != 'uniform':
                # cluster near the tip vertex end of this segment
                if (k == 0) and (int(deg.get(v_start, 0)) == 1):
                    end = 'start'
                elif (k == nseg - 1) and (int(deg.get(v_end, 0)) == 1):
                    end = 'end'
                else:
                    end = 'end'

                u = np.linspace(0.0, 1.0, Nk + 1)
                if tip_cluster == 'cheb':
                    # symmetric clustering; then flip if needed
                    th = np.linspace(0.0, math.pi, Nk + 1)
                    xi = 0.5 * (1.0 - np.cos(th))
                    if end == 'start':
                        xi = 1.0 - xi
                else:
                    # one-sided power clustering (default)
                    pwr = float(tip_cluster_power)
                    if end == 'start':
                        xi = u**pwr
                    else:
                        xi = 1.0 - (1.0 - u)**pwr
            else:
                # default: uniform within segment
                xi = np.linspace(0.0, 1.0, Nk + 1)

            s_local = s0 + Le * xi
            if not s_nodes_list:
                s_nodes_list.append(s_local)
            else:
                # avoid duplicating the shared node at segment boundaries
                s_nodes_list.append(s_local[1:])
            s0 += Le

        s_nodes = np.concatenate(s_nodes_list) if s_nodes_list else np.linspace(0.0, float(L), Np + 1)
        if collocation_mode == "nodes" and Np >= 2:
            s_col = s_nodes[1:-1].copy()
        else:
            s_col = 0.5 * (s_nodes[:-1] + s_nodes[1:])

        s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
        ds = (s_nodes[1:] - s_nodes[:-1])

        x_mid = []; t_mid = []; n_mid = []
        for sm in s_mid:
            if p.get('kind','polyline') == 'arc':
                x, t, n = _arc_point_and_frame(p, float(sm))
            elif p.get('kind','polyline') == 'cspline':
                x, t, n = _cspline_point_and_frame(p, float(sm))
            else:
                x, t, n = polyline_point_and_frame(network, vids_path, segL, float(sm))
            x_mid.append(x); t_mid.append(t); n_mid.append(n)
        x_mid = np.array(x_mid, float)
        t_mid = np.array(t_mid, float)
        n_mid = np.array(n_mid, float)

        x_col = []; t_col = []; n_col = []
        for sc in s_col:
            if p.get('kind','polyline') == 'arc':
                x, t, n = _arc_point_and_frame(p, float(sc))
            elif p.get('kind','polyline') == 'cspline':
                x, t, n = _cspline_point_and_frame(p, float(sc))
            else:
                x, t, n = polyline_point_and_frame(network, vids_path, segL, float(sc))
            x_col.append(x); t_col.append(t); n_col.append(n)
        x_col = np.array(x_col, float)
        t_col = np.array(t_col, float)
        n_col = np.array(n_col, float)

        nq = max(2, int(nq_stress))
        xg, wg = np.polynomial.legendre.leggauss(nq)

        src_pts = []; src_t = []; src_n = []; src_w = []
        panel_src = []
        panel_wsing = []
        eps_s = 1e-14 * L if L > 0 else 1e-14

        idx0 = 0
        for k in range(Np):
            sLk = float(s_nodes[k]); sRk = float(s_nodes[k + 1])
            Jk = 0.5 * (sRk - sLk)
            smk = 0.5 * (sRk + sLk)
            sq = smk + Jk * xg
            wq = Jk * wg

            wsing_k = 0.0
            for sqq, wqq in zip(sq, wq):
                if p.get('kind','polyline') == 'arc':
                    x, t, n = _arc_point_and_frame(p, float(sqq))
                elif p.get('kind','polyline') == 'cspline':
                    x, t, n = _cspline_point_and_frame(p, float(sqq))
                else:
                    x, t, n = polyline_point_and_frame(network, vids_path, segL, float(sqq))
                src_pts.append(x); src_t.append(t); src_n.append(n)

                if use_tip_singular:
                    if p.get("crack_mode", "full") == "full":
                        ss = max(float(sqq), eps_s)
                        tt = max(float(L - float(sqq)), eps_s)
                        sing = 1.0 / math.sqrt(ss * tt)
                    else:
                        tt = max(float(L - float(sqq)), eps_s)
                        sing = 1.0 / math.sqrt(tt)
                else:
                    sing = 1.0

                w_eff = float(wqq) * sing
                src_w.append(w_eff)
                wsing_k += w_eff

            panel_wsing.append(wsing_k)
            panel_src.append((idx0, idx0 + nq))
            idx0 += nq

        poly_panels.append(dict(
            pid=int(pid),
            Np=int(Np),
            s_nodes=np.array(s_nodes, float),
            s_mid=np.array(s_mid, float),
            ds=np.array(ds, float),
            x_mid=x_mid, t_mid=t_mid, n_mid=n_mid,
            x_col=x_col, t_col=t_col, n_col=n_col,
            src_pts=np.array(src_pts, float),
            src_t=np.array(src_t, float),
            src_n=np.array(src_n, float),
            src_w=np.array(src_w, float),
            panel_src=panel_src,
            panel_wsing=np.array(panel_wsing, float),
            v_start=int(p.get("v_start", vids_path[0] if len(vids_path) else -1)),
            v_end=int(p.get("v_end", vids_path[-1] if len(vids_path) else -1)),
            L=float(L),
        ))
    return poly_panels


def allocate_unknowns(
    poly_panels: List[dict],
    deg: dict[int, int],
    *,
    crack_mode: str,
    junction_model: str = "strict",
) -> Tuple[List[int], Dict[int, int], Dict[Tuple[int, str], int], int]:
    """
    Unknown blocks:
      1) Panel densities (always): 2*Np per polyline, contiguous per pid
      2) Jump DOFs at junctions (half mode only):
         - junction_model='strict': shared per vertex with degree>=2 (legacy Option A)
         - junction_model in {'core','soft'}: per-branch-end DOFs at endpoints with degree>=2

    Returns
    -------
    offsets : list[int]
        Start index of density unknowns for each pid.
    junction_dof : dict[int,int]
        For strict: vertex -> DOF offset (Jx,Jy). Empty for core/soft.
    branch_end_dof : dict[(pid,endflag), int]
        For core/soft: (pid,'start'/'end') -> DOF offset (Jx,Jy). Empty for strict.
    nunk : int
        Total number of unknowns.
    """
    offsets: List[int] = []
    nunk = 0

    for pp in poly_panels:
        offsets.append(int(nunk))
        nunk += 2 * int(pp["Np"])

    junction_model = str(junction_model).lower().strip()
    if junction_model not in ("strict", "core", "soft"):
        junction_model = "strict"

    junction_dof: Dict[int, int] = {}
    branch_end_dof: Dict[Tuple[int, str], int] = {}

    if str(crack_mode).lower().strip() == "half":
        if junction_model == "strict":
            endpoint_verts = set()
            for pp in poly_panels:
                if int(deg.get(int(pp["v_start"]), 0)) >= 2:
                    endpoint_verts.add(int(pp["v_start"]))
                if int(deg.get(int(pp["v_end"]), 0)) >= 2:
                    endpoint_verts.add(int(pp["v_end"]))
            endpoint_verts = sorted(endpoint_verts)
            for i, vj in enumerate(endpoint_verts):
                junction_dof[int(vj)] = int(nunk + 2 * i)
            nunk += 2 * len(endpoint_verts)
        else:
            # Per-branch-end DOFs at endpoints with degree>=2 (open-junction models)
            keys = []
            for pid, pp in enumerate(poly_panels):
                vs = int(pp["v_start"]); ve = int(pp["v_end"])
                if int(deg.get(vs, 0)) >= 2:
                    keys.append((int(pid), "start"))
                if int(deg.get(ve, 0)) >= 2:
                    keys.append((int(pid), "end"))
            keys = sorted(set(keys))
            for i, k in enumerate(keys):
                branch_end_dof[k] = int(nunk + 2 * i)
            nunk += 2 * len(keys)

    return offsets, junction_dof, branch_end_dof, int(nunk)



def assemble_operator(
    *,
    material: Material,
    applied: AppliedStress,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    nq_col: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ncol_tot = sum(int(len(pp["x_col"])) for pp in poly_panels)
    K = np.zeros((2 * ncol_tot, nunk), float)
    rhs = np.zeros((2 * ncol_tot,), float)

    E = float(material.E)
    nu = float(material.nu)
    mu = E / (2.0 * (1.0 + nu))

    sig = np.array([
        [float(applied.sigma_xx), float(applied.sigma_xy)],
        [float(applied.sigma_xy), float(applied.sigma_yy)],
    ], float)

    row0 = 0
    for pid_i, pp_i in enumerate(poly_panels):
        Nci = int(len(pp_i["x_col"]))
        for ic in range(Nci):
            x_i = pp_i["x_col"][ic]
            t_i = pp_i["t_col"][ic]
            n_i = pp_i["n_col"][ic]

            tr = sig @ n_i.reshape(2,)
            tn0 = float(np.dot(n_i, tr))
            ts0 = float(np.dot(t_i, tr))
            rhs[row0] = -tn0
            rhs[ncol_tot + row0] = -ts0

            for pid_j, pp_j in enumerate(poly_panels):
                offj = offsets[pid_j]
                Npj = int(pp_j["Np"])
                for k in range(Npj):
                    s0, s1 = pp_j["panel_src"][k]
                    src_pts = pp_j["src_pts"][s0:s1]
                    src_t = pp_j["src_t"][s0:s1]
                    src_n = pp_j["src_n"][s0:s1]
                    wq = pp_j["src_w"][s0:s1]

                    tn_sum_I = 0.0; ts_sum_I = 0.0
                    tn_sum_II = 0.0; ts_sum_II = 0.0

                    for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                        dx = float(x_i[0] - xs[0])
                        dy = float(x_i[1] - xs[1])

                        dB = ns * float(ww)
                        sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
                        tx = float(sxx * n_i[0] + sxy * n_i[1])
                        ty = float(sxy * n_i[0] + syy * n_i[1])
                        tn_sum_I += float(n_i[0] * tx + n_i[1] * ty)
                        ts_sum_I += float(t_i[0] * tx + t_i[1] * ty)

                        dB2 = ts * float(ww)
                        sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu)
                        tx = float(sxx * n_i[0] + sxy * n_i[1])
                        ty = float(sxy * n_i[0] + syy * n_i[1])
                        tn_sum_II += float(n_i[0] * tx + n_i[1] * ty)
                        ts_sum_II += float(t_i[0] * tx + t_i[1] * ty)

                    col_I = offj + 2 * k + 0
                    col_II = offj + 2 * k + 1
                    K[row0, col_I] = tn_sum_I
                    K[ncol_tot + row0, col_I] = ts_sum_I
                    K[row0, col_II] = tn_sum_II
                    K[ncol_tot + row0, col_II] = ts_sum_II

            row0 += 1

    return K, rhs



# ============================================================
# v4 overrides: mixed edge kinds with preserved LINE chaining
# - LINE edges are chained into polylines/branches
# - ARC edges are always standalone polylines (allowed between any vertex degrees, including junction–junction)
# - cspline support is inherited from build_v3 (make_cspline_polyline_from_path + discretize handling)
# ============================================================

def _line_edge_positions(network: CrackNetworkV4) -> list[int]:
    return [int(i) for i, e in enumerate(network.edges) if _edge_kind(e) == "line"]

def _arc_edge_positions(network: CrackNetworkV4) -> list[int]:
    return [int(i) for i, e in enumerate(network.edges) if _edge_kind(e) == "arc"]

def _components_from_edge_positions(network: CrackNetworkV4, edge_pos: list[int]) -> list[list[int]]:
    if not edge_pos:
        return []
    v2e: dict[int, list[int]] = {}
    for ep in edge_pos:
        e = network.edges[int(ep)]
        a = int(e.v0); b = int(e.v1)
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

def _order_path_from_component(network: CrackNetworkV4, edge_pos: list[int]) -> tuple[list[int], list[int]]:
    """Order a connected component of LINE edges into (vertex path, edge_pos path)."""
    v2e: dict[int, list[int]] = {}
    for ep in edge_pos:
        e = network.edges[int(ep)]
        a = int(e.v0); b = int(e.v1)
        v2e.setdefault(a, []).append(int(ep))
        v2e.setdefault(b, []).append(int(ep))
    deg_sub = {v: len(es) for v, es in v2e.items()}
    ends = [v for v, d in deg_sub.items() if d == 1]

    def other_vertex(ep: int, v: int) -> int:
        e = network.edges[int(ep)]
        a = int(e.v0); b = int(e.v1)
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


def build_polylines_full(network: CrackNetworkV4) -> Tuple[List[dict], dict[int, int]]:
    """Full crack mode: chain LINE edges; append ARC edges as standalone polylines."""
    if not network.edges:
        raise ValueError("No edges in network.")

    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}

    # --- LINE components
    line_epos = _line_edge_positions(network)
    comps = _components_from_edge_positions(network, line_epos)

    for pid, comp in enumerate(comps):
        vids_path, epos_path = _order_path_from_component(network, comp)
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
    arc_epos = _arc_edge_positions(network)
    pid0 = int(len(polylines))
    for j, ep in enumerate(arc_epos):
        e = network.edges[int(ep)]
        ctrls = _edge_ctrl_vids(e)
        if len(ctrls) != 1:
            raise ValueError(f"Arc edge {int(getattr(e,'id',ep))} must have exactly one ctrl_vid (center).")
        v0 = int(e.v0); v1 = int(e.v1)
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


def build_polylines_half_branches(
    network: CrackNetworkV4,
    deg: dict[int, int],
) -> Tuple[List[dict], dict[int, int]]:
    """Half crack mode: chain LINE edges into branches; append ARC edges as standalone branches."""
    if not network.edges:
        raise ValueError("No edges in network.")

    polylines: List[dict] = []
    edge_to_polyline: dict[int, int] = {}
    pid = 0

    # --- LINE-edge-only chaining
    line_epos = set(_line_edge_positions(network))

    v2e: dict[int, List[int]] = {}
    for ep in sorted(line_epos):
        e = network.edges[int(ep)]
        a = int(e.v0); b = int(e.v1)
        v2e.setdefault(a, []).append(int(ep))
        v2e.setdefault(b, []).append(int(ep))

    def other_vertex(ep: int, v: int) -> int:
        e = network.edges[int(ep)]
        a = int(e.v0); b = int(e.v1)
        return b if a == v else a

    used_edges: set[int] = set()

    start_verts = sorted([int(v) for v, d in deg.items() if int(d) != 2])
    for v0 in start_verts:
        inc_edges = v2e.get(int(v0), [])
        for e0 in inc_edges:
            e0 = int(e0)
            if e0 in used_edges:
                continue

            vids_path: List[int] = [int(v0)]
            epos_path: List[int] = []
            cur_v = int(v0)
            cur_e = int(e0)

            while True:
                if cur_e in used_edges:
                    break
                used_edges.add(cur_e)
                epos_path.append(int(cur_e))

                nxt_v = other_vertex(cur_e, cur_v)
                vids_path.append(int(nxt_v))

                dnext = int(deg.get(int(nxt_v), 0))
                if dnext != 2:
                    break

                inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
                if not inc:
                    break
                cur_v = int(nxt_v)
                cur_e = int(sorted(inc)[0])

            pts = np.array([network.vertex_coords(v) for v in vids_path], float)
            seg = pts[1:] - pts[:-1]
            segL = np.sqrt(np.sum(seg * seg, axis=1))
            L = float(np.sum(segL))
            if L <= 0:
                continue

            v_start = int(vids_path[0])
            v_end = int(vids_path[-1])

            # Orientation policy: keep tips at the start when possible
            if int(deg.get(v_start, 0)) == 1 and int(deg.get(v_end, 0)) > 1:
                vids_path = list(reversed(vids_path))
                epos_path = list(reversed(epos_path))
                segL = segL[::-1].copy()
                v_start = int(vids_path[0])
                v_end = int(vids_path[-1])

            for ep in epos_path:
                edge_to_polyline[int(ep)] = int(pid)

            polylines.append(dict(
                kind="polyline",
                path_vertex_ids=[int(v) for v in vids_path],
                path_edge_indices=[int(ep) for ep in epos_path],
                segment_lengths=[float(x) for x in segL.tolist()],
                total_length=float(L),
                crack_mode="half",
                v_start=int(v_start),
                v_end=int(v_end),
                deg_start=int(deg.get(v_start, 0)),
                deg_end=int(deg.get(v_end, 0)),
            ))
            pid += 1

    # Second pass: closed loops in LINE subgraph
    for ep0 in sorted(line_epos):
        if int(ep0) in used_edges:
            continue
        e0 = network.edges[int(ep0)]
        v0 = int(e0.v0)

        vids_path: List[int] = [v0]
        epos_path: List[int] = []
        cur_v = v0
        cur_e = int(ep0)

        while True:
            if cur_e in used_edges:
                break
            used_edges.add(cur_e)
            epos_path.append(int(cur_e))

            nxt_v = other_vertex(cur_e, cur_v)
            vids_path.append(int(nxt_v))

            if nxt_v == v0:
                break

            inc = [int(ei) for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
            if not inc:
                break
            cur_v = int(nxt_v)
            cur_e = int(sorted(inc)[0])

        if vids_path[-1] != v0:
            continue

        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        seg = pts[1:] - pts[:-1]
        segL = np.sqrt(np.sum(seg * seg, axis=1))
        L = float(np.sum(segL))
        if L <= 0:
            continue

        for ep in epos_path:
            edge_to_polyline[int(ep)] = int(pid)

        polylines.append(dict(
            kind="polyline",
            path_vertex_ids=[int(v) for v in vids_path],
            path_edge_indices=[int(ep) for ep in epos_path],
            segment_lengths=[float(x) for x in segL.tolist()],
            total_length=float(L),
            crack_mode="half",
            v_start=int(v0),
            v_end=int(v0),
            deg_start=2,
            deg_end=2,
        ))
        pid += 1

    # --- ARC edges: standalone branches
    arc_epos = _arc_edge_positions(network)
    for ep in arc_epos:
        e = network.edges[int(ep)]
        ctrls = _edge_ctrl_vids(e)
        if len(ctrls) != 1:
            raise ValueError(f"Arc edge {int(getattr(e,'id',ep))} must have exactly one ctrl_vid (center).")
        v0 = int(e.v0); v1 = int(e.v1)
        c = network.vertex_coords(int(ctrls[0]))
        p0 = network.vertex_coords(v0)
        p1 = network.vertex_coords(v1)
        meta = make_arc_polyline(p0, p1, c, crack_mode="half", v_start=v0, v_end=v1)
        meta["deg_start"] = int(deg.get(v0, 0))
        meta["deg_end"] = int(deg.get(v1, 0))
        meta["path_edge_indices"] = [int(ep)]
        polylines.append(meta)
        edge_to_polyline[int(ep)] = int(pid)
        pid += 1

    if not polylines:
        raise ValueError("No valid half polylines were built.")
    return polylines, edge_to_polyline