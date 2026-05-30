"""Random straight-crack generator.

Builds a network of `N` disconnected straight cracks, each a single edge
between two tip vertices. Crack lengths follow a (truncated) normal
distribution; centers are drawn uniformly inside the chosen domain
('disk' or 'rect'); orientations are uniform in `angle_range_deg`.

Returned arrays match the CRACK_NETWORK sheet schema consumed by
`input.excel_io._load_crack_network`:
    vertices : (2*N, 3) float -- columns [id, x_m, y_m]
    edges    : (N,   2) int   -- columns [v0, v1]

Use via `from fracture_utils.Ugenerator import random_straight_cracks`.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.stats import truncnorm

DomainSize = Union[float, Tuple[float, float]]


def _sample_lengths(
    rng: np.random.Generator,
    n: int,
    mean: float,
    std: float,
    lo: float,
    hi: float | None,
) -> np.ndarray:
    if std <= 0.0:
        return np.full(n, mean, dtype=float)
    a = (lo - mean) / std
    b = np.inf if hi is None else (hi - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=rng)


def _sample_centers_rect(
    rng: np.random.Generator, n: int, w: float, h: float, margin: float
) -> np.ndarray:
    xs = rng.uniform(-w / 2 + margin, w / 2 - margin, size=n)
    ys = rng.uniform(-h / 2 + margin, h / 2 - margin, size=n)
    return np.column_stack([xs, ys])


def _sample_centers_disk(
    rng: np.random.Generator, n: int, R: float, margin: float
) -> np.ndarray:
    r_max = max(R - margin, 0.0)
    r = r_max * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _endpoints_inside(
    p0: np.ndarray, p1: np.ndarray, shape: str, size, margin: float
) -> bool:
    if shape == "disk":
        R = float(size) - margin
        return (p0[0] ** 2 + p0[1] ** 2 <= R ** 2) and (
            p1[0] ** 2 + p1[1] ** 2 <= R ** 2
        )
    if shape == "rect":
        w, h = size
        hx, hy = w / 2 - margin, h / 2 - margin
        return (-hx <= p0[0] <= hx) and (-hy <= p0[1] <= hy) and (
            -hx <= p1[0] <= hx
        ) and (-hy <= p1[1] <= hy)
    raise ValueError(f"unknown domain_shape {shape!r}")


def random_straight_cracks(
    n_cracks: int,
    length_mean: float,
    length_std: float,
    *,
    domain_shape: str = "disk",
    domain_size: DomainSize = 12.7e-3,
    length_min: float = 1.0e-4,
    length_max: float | None = None,
    angle_range_deg: Tuple[float, float] = (0.0, 180.0),
    margin: float = 0.0,
    seed: int | None = None,
    max_attempts_per_crack: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, edges) for `n_cracks` disconnected straight cracks.

    Parameters
    ----------
    n_cracks : int
        Number of cracks (= number of edges; vertex count is 2*n_cracks).
    length_mean, length_std : float
        Mean and standard deviation of the (truncated) normal length
        distribution, in metres.
    domain_shape : {'disk', 'rect'}
        Shape of the containment region the crack endpoints must lie in.
    domain_size : float or (float, float)
        Radius (disk) or (width, height) (rect), in metres. Centred at origin.
    length_min : float
        Lower truncation for the length distribution; also guards against
        zero/negative lengths when `length_std > length_mean`.
    length_max : float, optional
        Upper truncation. None = no upper bound (still capped by the
        endpoints-inside-domain rejection step).
    angle_range_deg : (lo, hi)
        Uniform orientation range, in degrees. Crack lines are unoriented,
        so [0, 180) is the natural default.
    margin : float
        Buffer between the domain boundary and crack endpoints, in metres.
    seed : int, optional
        Seed for the RNG. None = nondeterministic.
    max_attempts_per_crack : int
        Per-crack rejection-sampling cap. Cracks exhausting attempts are
        skipped (with a warning) -- final edge count may be < n_cracks.

    Returns
    -------
    vertices : (2*N, 3) float ndarray  -- [id, x, y]
    edges    : (N, 2) int ndarray      -- [v0, v1]
    """
    if n_cracks <= 0:
        return np.zeros((0, 3), float), np.zeros((0, 2), int)
    rng = np.random.default_rng(seed)
    angle_lo, angle_hi = float(angle_range_deg[0]), float(angle_range_deg[1])

    verts: list[list[float]] = []
    edges: list[list[int]] = []
    skipped = 0
    for k in range(n_cracks):
        accepted = False
        for _ in range(max_attempts_per_crack):
            length = float(
                _sample_lengths(rng, 1, length_mean, length_std, length_min, length_max)[0]
            )
            if domain_shape == "disk":
                center = _sample_centers_disk(rng, 1, float(domain_size), margin)[0]
            else:
                w, h = domain_size  # type: ignore[misc]
                center = _sample_centers_rect(rng, 1, float(w), float(h), margin)[0]
            theta = np.deg2rad(rng.uniform(angle_lo, angle_hi))
            half = 0.5 * length
            dx, dy = half * np.cos(theta), half * np.sin(theta)
            p0 = np.array([center[0] - dx, center[1] - dy])
            p1 = np.array([center[0] + dx, center[1] + dy])
            if _endpoints_inside(p0, p1, domain_shape, domain_size, margin):
                v0 = len(verts)
                v1 = v0 + 1
                verts.append([float(v0), float(p0[0]), float(p0[1])])
                verts.append([float(v1), float(p1[0]), float(p1[1])])
                edges.append([v0, v1])
                accepted = True
                break
        if not accepted:
            skipped += 1

    if skipped:
        print(
            f"[random_straight_cracks] warning: {skipped}/{n_cracks} cracks "
            f"skipped after {max_attempts_per_crack} attempts (too long for domain?)"
        )

    vertices = np.asarray(verts, dtype=float).reshape(-1, 3)
    edges_arr = np.asarray(edges, dtype=int).reshape(-1, 2)
    return vertices, edges_arr
