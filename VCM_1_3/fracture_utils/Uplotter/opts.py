"""Plot options and shared helpers for DCE plotting."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict
import numpy as np

@dataclass
class StressPlotOptsV4:
    extent_factor: float = 5.0
    n_grid: int = 200
    add_remote: bool = True

# NEW: explicit domain control
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    
    # masking
    mask_cracks: bool = True
    mask_crack: Optional[bool] = None  # alias accepted by notebooks
    mask_width_factor: float = 2e-3

    # styling
    cmap: str = "jet"
    n_bands: int = 20
    label_contours: bool = False
    label_fmt: str = "%.2g"

    # range control (legacy)
    vmax_factor: float = 2.0
    robust_percentile: float = 99.0

    # explicit limits
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # optional percentile clipping
    clip_percentiles: Optional[Tuple[float, float]] = None

    # normalization / scaling
    norm: str = "linear"
    symlog_linthresh: float = 1.0
    symlog_linscale: float = 1.0
    symlog_base: float = 10.0

    extend: str = "both"
    dpi: int = 150

    def __post_init__(self):
        if self.mask_crack is not None:
            self.mask_cracks = bool(self.mask_crack)


def ensure_dir(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, out_dir: Optional[Path], name: str, dpi: int = 150):
    if out_dir is None:
        return
    out_dir = ensure_dir(out_dir)
    fig.savefig(out_dir / f"{name}.png", dpi=int(dpi), bbox_inches="tight")


def apply_clip_percentiles(Z: np.ndarray, clip: Optional[Tuple[float, float]]) -> np.ndarray:
    if clip is None:
        return Z
    try:
        p_lo, p_hi = float(clip[0]), float(clip[1])
    except Exception:
        return Z
    if not (0.0 <= p_lo < p_hi <= 100.0):
        return Z
    lo, hi = np.nanpercentile(Z, [p_lo, p_hi])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return np.clip(Z, lo, hi)
    return Z


def robust_vmin_vmax(Z: np.ndarray, opts: StressPlotOptsV4, ref: float = 0.0) -> Tuple[float, float]:
    if opts.vmin is not None or opts.vmax is not None:
        vmin = float(opts.vmin) if opts.vmin is not None else float(np.nanmin(Z))
        vmax = float(opts.vmax) if opts.vmax is not None else float(np.nanmax(Z))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
            a = float(np.nanmax(np.abs(Z)))
            a = max(a, 1e-12)
            return -a, a
        return vmin, vmax

    ref = float(abs(ref))
    if ref > 0:
        vmax = float(opts.vmax_factor) * ref
    else:
        vmax = float(np.nanpercentile(np.abs(Z), float(opts.robust_percentile)))
    vmax = max(vmax, 1e-12)
    return -vmax, vmax


def symlog_levels(vmin: float, vmax: float, linthresh: float, n: int) -> np.ndarray:
    vmin, vmax = float(vmin), float(vmax)
    linthresh = max(float(linthresh), 1e-12)
    n = int(max(8, n))
    n_core = max(3, n // 3)
    n_wing = max(2, (n - n_core) // 2)
    core = np.linspace(-linthresh, linthresh, n_core, endpoint=True)
    pos_max = max(vmax, linthresh * 1.01)
    pos = np.geomspace(linthresh, pos_max, n_wing + 1)[1:]
    neg = -pos[::-1]
    levels = np.unique(np.concatenate([neg, core, pos]))
    levels[0] = min(levels[0], vmin)
    levels[-1] = max(levels[-1], vmax)
    return levels


def rot_from_tangent(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float).reshape(2,)
    nrm = float(np.hypot(t[0], t[1]))
    if nrm == 0:
        return np.eye(2)
    ex = t / nrm
    ey = np.array([-ex[1], ex[0]])
    return np.column_stack([ex, ey])


def network_extent(vertices: Sequence) -> float:
    xy = np.array([[float(v.x), float(v.y)] for v in vertices], float)
    if xy.size == 0:
        return 1.0
    xspan = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    yspan = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    return max(xspan, yspan, 1e-12)
