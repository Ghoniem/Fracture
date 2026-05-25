"""
Curved-geometry utilities for VCM/DCE crack polylines.

This module is intentionally geometry-only:
- Circular arcs (center-defined)
- Natural cubic splines (interpolating through vertices)
- Evaluation helpers: x(s), tangent t(s), normal n(s)

No network traversal / branch construction lives here.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


# ============================================================
# Circular arc support
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
    """
    Create a polyline dict representing a circular arc.

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
        Polyline record compatible with discretizers that accept (kind='arc').
    """
    p0 = np.asarray(p0, float).reshape(2)
    p1 = np.asarray(p1, float).reshape(2)
    c = np.asarray(center, float).reshape(2)

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
        path_edge_indices=[], # optional, may be filled by caller
        v_start=int(v_start),
        v_end=int(v_end),
    )


def arc_point_and_frame(p: dict, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# Natural cubic spline support (interpolating through vertices)
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
    A = np.zeros((n - 2, n - 2), float)
    rhs = np.zeros((n - 2,), float)
    for i in range(1, n - 1):
        hi1 = h[i - 1]
        hi = h[i]
        row = i - 1
        if row - 1 >= 0:
            A[row, row - 1] = hi1
        A[row, row] = 2.0 * (hi1 + hi)
        if row + 1 <= n - 3:
            A[row, row + 1] = hi
        rhs[row] = 6.0 * ((y[i + 1] - y[i]) / hi - (y[i] - y[i - 1]) / hi1)

    M_inner = np.linalg.solve(A, rhs) if (n - 2) > 0 else np.array([], float)
    M = np.zeros(n, float)
    M[1:n - 1] = M_inner
    return M


def _eval_cubic_spline(t: np.ndarray, y: np.ndarray, M: np.ndarray, tt: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    M = np.asarray(M, float)
    tt = np.asarray(tt, float)

    idx = np.searchsorted(t, tt, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)

    h = t[idx + 1] - t[idx]
    a = (t[idx + 1] - tt) / h
    b = (tt - t[idx]) / h
    S = (
        a * y[idx]
        + b * y[idx + 1]
        + ((a**3 - a) * M[idx] + (b**3 - b) * M[idx + 1]) * (h**2) / 6.0
    )
    return S


def _eval_cubic_spline_deriv(t: np.ndarray, y: np.ndarray, M: np.ndarray, tt: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    M = np.asarray(M, float)
    tt = np.asarray(tt, float)

    idx = np.searchsorted(t, tt, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)

    h = t[idx + 1] - t[idx]
    a = (t[idx + 1] - tt) / h
    b = (tt - t[idx]) / h
    dS = (y[idx + 1] - y[idx]) / h + (h / 6.0) * (-(3 * a * a - 1.0) * M[idx] + (3 * b * b - 1.0) * M[idx + 1])
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
    """
    Build a 'cspline' polyline record that interpolates given points.

    Notes
    -----
    - We store a natural cubic spline parameterized by chord length (t in [0,1]).
    - We approximate arc-length by sampling the spline densely (n_len_samples).
    """
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


def cspline_point_and_frame(p: dict, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate cspline position x(s), unit tangent t(s), unit normal n(s) for s in [0,L].

    Important: we use a crude mapping tt = s/L into the spline parameter (tt in [0,1]).
    If you later want true arc-length parameterization, we can add a precomputed lookup.
    """
    L = float(p.get("total_length", 0.0))
    if L <= 0.0:
        return np.array([0.0, 0.0], float), np.array([1.0, 0.0], float), np.array([0.0, 1.0], float)

    ss = float(np.clip(s, 0.0, L))
    tt = ss / L

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
    if nrm <= 0.0:
        tvec = np.array([1.0, 0.0], float)
        nrm = 1.0
    tvec /= nrm
    nvec = np.array([-tvec[1], tvec[0]], float)
    return np.array([x, y], float), tvec, nvec
