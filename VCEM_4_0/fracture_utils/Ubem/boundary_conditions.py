"""
boundary_conditions.py
======================

Utilities to:
  1) define an external boundary (circle, rectangle, polygon),
  2) discretize it into N_boundary linear segments,
  3) read/encode imposed boundary conditions as:
        - total force on a set of segments,
        - traction on segments,
        - displacement on segments,
  4) prepare boundary elements for bem_solver.BEMSolver2D (add_element calls).

Design goals
------------
- Minimal assumptions about "selection" of segments. You can select by:
    * explicit segment indices,
    * predicate on element midpoint (x,y),
    * angle window for circular boundaries.
- Boundary conditions are assembled as per-segment BCs compatible with:
      solver.add_element(x1,y1,x2,y2,is_traction,bc_x,bc_y)

Conventions
-----------
- Boundary normals are OUTWARD.
- For a compressive normal pressure p>0 applied to the boundary, traction is
      t = -p * n_out
  (same convention used in your Brazilian disk cell).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

BCType = Literal["traction", "displacement"]
BCLoadType = Literal["traction", "displacement", "total_force", "pressure_normal"]

SelectorKind = Literal["all", "indices", "predicate", "theta_deg_range"]

@dataclass(frozen=True)
class BoundaryMesh:
    """Piecewise-linear boundary mesh (n segments)."""
    x1: np.ndarray  # (n,)
    y1: np.ndarray  # (n,)
    x2: np.ndarray  # (n,)
    y2: np.ndarray  # (n,)
    xm: np.ndarray  # (n,)
    ym: np.ndarray  # (n,)
    nx: np.ndarray  # (n,) outward normal at segment (from local tangent)
    ny: np.ndarray  # (n,)
    length: np.ndarray  # (n,)
    theta_deg: Optional[np.ndarray] = None  # (n,) midpoint polar angle for circle-like boundaries

    @property
    def n_seg(self) -> int:
        return int(self.x1.shape[0])


# -------------------------
# Geometry builders
# -------------------------

def build_circle_boundary(R: float, n_boundary: int, center: Tuple[float, float]=(0.0, 0.0)) -> BoundaryMesh:
    """Circle boundary discretized into n_boundary straight segments."""
    cx, cy = center
    theta = np.linspace(0.0, 2.0*np.pi, n_boundary + 1)[:-1]
    x_nodes = cx + R*np.cos(theta)
    y_nodes = cy + R*np.sin(theta)

    x1 = x_nodes
    y1 = y_nodes
    x2 = np.roll(x_nodes, -1)
    y2 = np.roll(y_nodes, -1)

    return _segments_to_mesh(x1, y1, x2, y2, compute_theta=True, center=center)


def build_circle_graded_boundary(
    R: float,
    n_boundary: int,
    center: Tuple[float, float] = (0.0, 0.0),
    *,
    focal_angles_deg: Sequence[float] = (90.0, -90.0),
    concentration: float = 8.0,
    taper_exponent: float = 4.0,
) -> BoundaryMesh:
    """Circle boundary with non-uniform spacing concentrated near focal angles.

    Density of nodes at angle theta is

        d(theta) = 1 + (concentration - 1) * sum_k max(0, cos(theta - phi_k))^p

    where phi_k = focal_angles_deg (in radians) and p = taper_exponent. Nodes
    are placed by inverse-CDF of d(theta), so segment count per arc equals
    the density-weighted arc length. For the Brazilian-disk default
    (phi = +/-90 deg, p = 4), d(theta) simplifies to 1 + (C-1)*sin^4(theta):
    peak density C at the platens, baseline 1 at the equator, smooth taper.

    Symmetry is exact by construction: density is even about each focal
    angle and about the midpoint between consecutive focals, so the CDF
    inversion places nodes whose mirror images across both the x and y
    axes are also nodes (provided n_boundary % 4 == 0 when focals are at
    +/-90 deg, which preserves the quadrant nodes).

    Parameters
    ----------
    R, center : circle geometry.
    n_boundary : total number of straight segments (recommend even; for
        focals at +/-90 deg, multiples of 4 keep the quadrant landings).
    focal_angles_deg : where to concentrate (default: top + bottom platens).
    concentration : density at a focal point relative to baseline (default 8).
    taper_exponent : how fast the density returns to baseline; larger = sharper
        peak (default 4 gives a ~+/-45 deg effective width).
    """
    cx, cy = center
    n = int(n_boundary)
    if n < 4:
        raise ValueError("n_boundary must be >= 4 for a meaningful circle")

    focal_rad = np.array([np.radians(float(a)) for a in focal_angles_deg], dtype=float)
    C = float(concentration)
    p = float(taper_exponent)
    if C < 1.0:
        raise ValueError("concentration must be >= 1.0")
    if p <= 0.0:
        raise ValueError("taper_exponent must be > 0")

    # Build a fine angular grid and integrate d(theta) to its CDF.
    # n_fine = max(20000, 200*n) keeps interpolation error well below the
    # node spacing even for n_boundary up to ~1000.
    n_fine = max(20000, 200 * n)
    th = np.linspace(0.0, 2.0 * np.pi, n_fine + 1)

    density = np.ones_like(th)
    for phi in focal_rad:
        c = np.cos(th - phi)
        density += (C - 1.0) * np.maximum(0.0, c) ** p

    # CDF via trapezoid rule
    dtheta = np.diff(th)
    increments = 0.5 * (density[:-1] + density[1:]) * dtheta
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    cdf = cdf / cdf[-1]   # normalize to [0, 1]

    # Invert at equal CDF intervals to place n+1 node angles (closing point
    # duplicates the start). Drop the duplicate so we have n segments.
    target = np.linspace(0.0, 1.0, n + 1)
    node_angles = np.interp(target, cdf, th)
    node_angles = node_angles[:-1]   # length n

    x_nodes = cx + R * np.cos(node_angles)
    y_nodes = cy + R * np.sin(node_angles)
    x1 = x_nodes
    y1 = y_nodes
    x2 = np.roll(x_nodes, -1)
    y2 = np.roll(y_nodes, -1)

    return _segments_to_mesh(x1, y1, x2, y2, compute_theta=True, center=center)

def build_rectangle_boundary(width: float, height: float, n_boundary: int,
                            center: Tuple[float, float]=(0.0, 0.0)) -> BoundaryMesh:
    """
    Axis-aligned rectangle centered at `center`.
    Discretization distributes segments proportional to side lengths.
    """
    cx, cy = center
    w = width
    h = height
    # Corners CCW starting from (w/2, -h/2) is arbitrary; choose (w/2, -h/2) -> (w/2,h/2) -> ...
    corners = np.array([
        [cx + w/2, cy - h/2],
        [cx + w/2, cy + h/2],
        [cx - w/2, cy + h/2],
        [cx - w/2, cy - h/2],
    ], dtype=float)

    # Side lengths (all straight)
    side_lengths = np.array([h, w, h, w], dtype=float)
    n_side = _allocate_segments(n_boundary, side_lengths)

    # Build points along each side (excluding the last point to avoid duplication)
    pts: List[np.ndarray] = []
    for k in range(4):
        p0 = corners[k]
        p1 = corners[(k+1) % 4]
        nk = int(n_side[k])
        t = np.linspace(0.0, 1.0, nk + 1)[:-1]
        pts.append((1.0 - t)[:, None]*p0[None, :] + t[:, None]*p1[None, :])
    nodes = np.vstack(pts)
    x1, y1 = nodes[:, 0], nodes[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    return _segments_to_mesh(x1, y1, x2, y2, compute_theta=False)

def build_polygon_boundary(vertices: np.ndarray, n_boundary: int) -> BoundaryMesh:
    """
    Polygon boundary, vertices given in CCW order, shape (nV,2).
    Discretization distributes segments proportional to edge lengths.
    """
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("vertices must be (nV,2)")
    # Ensure closed loop edges
    V2 = np.vstack([V, V[0]])
    edges = V2[1:] - V2[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    n_edge = _allocate_segments(n_boundary, edge_lengths)

    pts: List[np.ndarray] = []
    for k in range(V.shape[0]):
        p0 = V[k]
        p1 = V[(k+1) % V.shape[0]]
        nk = int(n_edge[k])
        t = np.linspace(0.0, 1.0, nk + 1)[:-1]
        pts.append((1.0 - t)[:, None]*p0[None, :] + t[:, None]*p1[None, :])
    nodes = np.vstack(pts)
    x1, y1 = nodes[:, 0], nodes[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    return _segments_to_mesh(x1, y1, x2, y2, compute_theta=False)

def build_boundary(spec: Dict) -> BoundaryMesh:
    """
    Unified boundary builder.

    Examples
    --------
    spec = {"type":"circle", "R":0.1, "n_boundary":400, "center":[0,0]}
    spec = {"type":"circle_graded", "R":0.0127, "n_boundary":120,
            "focal_angles_deg":[90,-90], "concentration":8, "taper_exponent":4}
    spec = {"type":"rectangle","width":1.0,"height":0.5,"n_boundary":200}
    spec = {"type":"polygon","vertices":[[0,0],[1,0],[1,1],[0,1]],"n_boundary":300}
    """
    btype = spec.get("type", None)
    if btype is None:
        raise ValueError("boundary spec must include 'type'")
    n = int(spec.get("n_boundary", 0))
    if n <= 0:
        raise ValueError("boundary spec must include positive n_boundary")

    if btype == "circle":
        R = float(spec["R"])
        center = tuple(spec.get("center", (0.0, 0.0)))
        return build_circle_boundary(R=R, n_boundary=n, center=center)
    if btype == "circle_graded":
        R = float(spec["R"])
        center = tuple(spec.get("center", (0.0, 0.0)))
        focal = spec.get("focal_angles_deg", (90.0, -90.0))
        concentration  = float(spec.get("concentration", 8.0))
        taper_exponent = float(spec.get("taper_exponent", 4.0))
        return build_circle_graded_boundary(
            R=R, n_boundary=n, center=center,
            focal_angles_deg=focal,
            concentration=concentration,
            taper_exponent=taper_exponent,
        )
    if btype == "rectangle":
        width = float(spec["width"])
        height = float(spec["height"])
        center = tuple(spec.get("center", (0.0, 0.0)))
        return build_rectangle_boundary(width=width, height=height, n_boundary=n, center=center)
    if btype == "polygon":
        vertices = np.asarray(spec["vertices"], dtype=float)
        return build_polygon_boundary(vertices=vertices, n_boundary=n)

    raise ValueError(f"Unknown boundary type: {btype!r}")


# -------------------------
# Boundary-condition encoding
# -------------------------

@dataclass(frozen=True)
class BCSpec:
    """
    A boundary condition "block" applied to a subset of boundary segments.

    load_type:
      - "traction": bc_value = (tx, ty) [Pa] (constant traction vector)
      - "displacement": bc_value = (ux, uy) [m] (constant displacement vector)
      - "total_force": bc_value = (Fx, Fy) [N] total force distributed uniformly over selected segments
      - "pressure_normal": bc_value = p [Pa], applied as traction t = -p * n_out

    selector:
      - kind="all"
      - kind="indices": data = list[int]
      - kind="predicate": data = callable(xm,ym)->bool
      - kind="theta_deg_range": data=(theta_min_deg, theta_max_deg) for circle
    """
    load_type: BCLoadType
    bc_value: Union[Tuple[float, float], float]
    selector_kind: SelectorKind = "all"
    selector_data: Optional[Union[Sequence[int], Tuple[float, float], Callable[[float, float], bool]]] = None

def assemble_segment_bcs(
    mesh: BoundaryMesh,
    bc_specs: Sequence[BCSpec],
    default: Tuple[BCType, float, float] = ("traction", 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-segment:
      - is_traction (bool array, True => traction BC known, displacement unknown)
      - bc_x, bc_y (known values for traction or displacement depending on is_traction)
    """
    n = mesh.n_seg
    is_traction = np.full(n, default[0] == "traction", dtype=bool)
    bc_x = np.full(n, float(default[1]), dtype=float)
    bc_y = np.full(n, float(default[2]), dtype=float)

    for spec in bc_specs:
        idx = _select_indices(mesh, spec.selector_kind, spec.selector_data)
        if len(idx) == 0:
            continue

        if spec.load_type == "traction":
            tx, ty = _as_vec2(spec.bc_value)
            is_traction[idx] = True
            bc_x[idx] = tx
            bc_y[idx] = ty

        elif spec.load_type == "displacement":
            ux, uy = _as_vec2(spec.bc_value)
            is_traction[idx] = False
            bc_x[idx] = ux
            bc_y[idx] = uy

        elif spec.load_type == "total_force":
            Fx, Fy = _as_vec2(spec.bc_value)
            L = float(np.sum(mesh.length[idx]))
            if L <= 0:
                raise ValueError("Selected segments have zero total length for total_force BC.")
            tx = Fx / L
            ty = Fy / L
            is_traction[idx] = True
            bc_x[idx] = tx
            bc_y[idx] = ty

        elif spec.load_type == "pressure_normal":
            p = float(spec.bc_value)
            # traction = -p * n_out
            is_traction[idx] = True
            bc_x[idx] = -p * mesh.nx[idx]
            bc_y[idx] = -p * mesh.ny[idx]

        else:
            raise ValueError(f"Unknown BC load_type: {spec.load_type!r}")

    return is_traction, bc_x, bc_y

def add_boundary_to_solver(solver, mesh: BoundaryMesh, is_traction: np.ndarray, bc_x: np.ndarray, bc_y: np.ndarray) -> None:
    """Push boundary segments + BCs into BEMSolver2D."""
    n = mesh.n_seg
    if not (len(is_traction) == len(bc_x) == len(bc_y) == n):
        raise ValueError("BC arrays must have same length as number of segments.")
    for i in range(n):
        solver.add_element(
            float(mesh.x1[i]), float(mesh.y1[i]),
            float(mesh.x2[i]), float(mesh.y2[i]),
            bool(is_traction[i]),
            float(bc_x[i]), float(bc_y[i]),
        )


# -------------------------
# Helpers
# -------------------------

def _segments_to_mesh(x1, y1, x2, y2, compute_theta: bool, center: Tuple[float, float]=(0.0, 0.0)) -> BoundaryMesh:
    x1 = np.asarray(x1, dtype=float).copy()
    y1 = np.asarray(y1, dtype=float).copy()
    x2 = np.asarray(x2, dtype=float).copy()
    y2 = np.asarray(y2, dtype=float).copy()
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    # Tangent
    tx = dx / length
    ty = dy / length
    # Outward normal for CCW boundary: n = (ty, -tx)
    nx = ty
    ny = -tx
    xm = 0.5*(x1 + x2)
    ym = 0.5*(y1 + y2)

    theta_deg = None
    if compute_theta:
        cx, cy = center
        theta_deg = np.degrees(np.arctan2(ym - cy, xm - cx))
        # map to (-180,180]
        theta_deg = ((theta_deg + 180.0) % 360.0) - 180.0

    return BoundaryMesh(x1=x1, y1=y1, x2=x2, y2=y2, xm=xm, ym=ym, nx=nx, ny=ny, length=length, theta_deg=theta_deg)

def _allocate_segments(n_total: int, weights: np.ndarray) -> np.ndarray:
    """Allocate integer segments to edges proportional to weights, with at least 1 each."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    m = w.size
    # start with proportional allocation
    raw = w / (np.sum(w) + 1e-30) * n_total
    n = np.maximum(1, np.floor(raw).astype(int))
    # fix sum via distributing remainder
    while n.sum() < n_total:
        k = int(np.argmax(raw - n))
        n[k] += 1
    while n.sum() > n_total:
        k = int(np.argmax(n - 1))
        if n[k] > 1:
            n[k] -= 1
        else:
            break
    return n

def _as_vec2(v: Union[Tuple[float, float], float]) -> Tuple[float, float]:
    if isinstance(v, (int, float, np.floating)):
        raise TypeError("Expected a 2-vector (x,y), got scalar.")
    if len(v) != 2:
        raise ValueError("Expected length-2 tuple for vector BC.")
    return float(v[0]), float(v[1])

def _select_indices(mesh: BoundaryMesh, kind: SelectorKind, data) -> np.ndarray:
    n = mesh.n_seg
    if kind == "all":
        return np.arange(n, dtype=int)

    if kind == "indices":
        if data is None:
            raise ValueError("indices selector requires selector_data=list[int]")
        return np.asarray(list(data), dtype=int)

    if kind == "predicate":
        if data is None or not callable(data):
            raise ValueError("predicate selector requires selector_data=callable(xm,ym)->bool")
        mask = np.fromiter((bool(data(float(mesh.xm[i]), float(mesh.ym[i]))) for i in range(n)), dtype=bool, count=n)
        return np.flatnonzero(mask).astype(int)

    if kind == "theta_deg_range":
        if mesh.theta_deg is None:
            raise ValueError("theta_deg_range selector requires a boundary built with compute_theta=True (circle).")
        if data is None or len(data) != 2:
            raise ValueError("theta_deg_range selector requires selector_data=(theta_min_deg, theta_max_deg)")
        tmin, tmax = float(data[0]), float(data[1])
        th = mesh.theta_deg
        # arctan2-derived theta_deg can land ~1e-14 deg off integer
        # endpoints (e.g. -105.00000000000001 for the segment intended at
        # -105 deg). A strict comparison drops that segment, producing
        # asymmetric BC selection and a measurably asymmetric stress field.
        # Absorb the noise with a 1e-9 deg tolerance (far smaller than any
        # realistic loaded-arc resolution).
        eps = 1.0e-9
        if tmin <= tmax:
            mask = (th >= tmin - eps) & (th <= tmax + eps)
        else:
            mask = (th >= tmin - eps) | (th <= tmax + eps)
        return np.flatnonzero(mask).astype(int)

    raise ValueError(f"Unknown selector_kind: {kind!r}")
