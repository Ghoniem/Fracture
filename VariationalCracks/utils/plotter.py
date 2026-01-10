"""utils/dce_plots_parametrized_v1_5.py

Stable plotting utilities for DCE crack-network v4.

This module is intentionally self-contained and aligned with the notebook calls:

    plotter = DCEPlotterV4(res, out_dir=out_dir)
    plotter.plot_network_graph()
    plotter.plot_deformed_network(scale=1e6, n_pts_per_edge=400, show_faces=True)
    plotter.plot_displacement_vector_field(extent_factor=1.2, n_grid=41, scale=1.0,
                                           disp_scale=1e6, mask_cracks=True)
    opts = StressPlotOptsV4(...)
    plotter.plot_stress_components_global(opts=opts, components=("sxx","syy","sxy"))

Key points:
- Stress is plotted in MPa.
- vmin/vmax capping via vmax_factor (relative to remote component) or robust percentile.
- Deformed network uses the NETWORK vertices/edges for geometry (not per-edge crack centers),
  so topology matches the graph.

"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Options
# -------------------------

@dataclass
class StressPlotOptsV4:
    extent_factor: float = 5.0
    n_grid: int = 141
    add_remote: bool = True

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

    # explicit limits (override robust scaling if provided)
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # optional percentile clipping
    clip_percentiles: Optional[Tuple[float, float]] = None

    # normalization / scaling: "linear" (default), "symlog", "log" (positive-only)
    norm: str = "linear"
    symlog_linthresh: float = 1.0
    symlog_linscale: float = 1.0
    symlog_base: float = 10.0

    # colorbar extension: "neither","min","max","both"
    extend: str = "both"

    # output
    dpi: int = 150

    def __post_init__(self):
        if self.mask_crack is not None:
            self.mask_cracks = bool(self.mask_crack)


def _ensure_dir(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig, out_dir: Optional[Path], name: str, dpi: int = 150):
    if out_dir is None:
        return
    out_dir = _ensure_dir(out_dir)
    fig.savefig(out_dir / f"{name}.png", dpi=int(dpi), bbox_inches="tight")



def _apply_clip_percentiles(Z: np.ndarray, clip: Optional[Tuple[float, float]]) -> np.ndarray:
    """Clip Z by percentiles while respecting NaNs."""
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


def _robust_vmin_vmax(Z: np.ndarray, opts: StressPlotOptsV4, ref: float = 0.0) -> Tuple[float, float]:
    """Determine (vmin, vmax) using explicit limits or robust scaling."""
    if getattr(opts, "vmin", None) is not None or getattr(opts, "vmax", None) is not None:
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


def _symlog_levels(vmin: float, vmax: float, linthresh: float, n: int) -> np.ndarray:
    """Generate symmetric contour levels suitable for symlog plots."""
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


def _rot_from_tangent(t: np.ndarray) -> np.ndarray:
    """Return rotation matrix mapping local->global, local x aligned with tangent."""
    t = np.asarray(t, float).reshape(2,)
    nrm = float(np.hypot(t[0], t[1]))
    if nrm == 0:
        return np.eye(2)
    ex = t / nrm
    ey = np.array([-ex[1], ex[0]])
    return np.column_stack([ex, ey])


def _network_extent(vertices: Sequence) -> float:
    xy = np.array([[float(v.x), float(v.y)] for v in vertices], float)
    if xy.size == 0:
        return 1.0
    xspan = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    yspan = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    return max(xspan, yspan, 1e-12)


# -------------------------
# Smoothing helpers
# -------------------------

def _moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware moving-average smoother with reflect padding.

    - Preserves array length.
    - If window <= 1, returns y unchanged.
    - If window is even, it is promoted to the next odd integer.
    """
    y = np.asarray(y, float).reshape(-1,)
    window = int(window)
    if window <= 1 or y.size == 0:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2

    # Reflect padding reduces end effects without imposing zero-slope.
    yp = np.pad(y, pad_width=pad, mode="reflect")
    valid = np.isfinite(yp).astype(float)
    yp0 = np.where(np.isfinite(yp), yp, 0.0)

    k = np.ones(window, float)
    num = np.convolve(yp0, k, mode="valid")
    den = np.convolve(valid, k, mode="valid")
    den = np.where(den > 0, den, np.nan)
    return num / den


def _resolve_smooth_window(value, ne_half: int, *, C: int = 50, wmin: int = 3) -> int:
    """Resolve smoothing window specification.

    value:
      - int/float: direct window in panels (<=1 disables)
      - 'auto': window ~= C / ne_half (decreases with refinement), min wmin
      - None: disables

    Returns an odd integer window; 0 disables.
    """
    if value is None:
        return 0
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            nh = max(1, int(ne_half))
            w = int(round(float(C) / float(nh)))
            w = max(int(wmin), w)
        else:
            try:
                w = int(round(float(value)))
            except Exception:
                return 0
    else:
        try:
            w = int(round(float(value)))
        except Exception:
            return 0

    if w <= 1:
        return 0
    if w % 2 == 0:
        w += 1
    return int(w)


def _reconstruct_cod_csd_parametrized_smoothed(res, edge_index: int, n_pts: int = 4000,
                                              enforce_global_tip_zero: bool = True,
                                              cod_window_panels: int = 0,
                                              csd_window_panels: int = 0):
    """Reconstruct (x_edge, COD, CSD) for parametrized_crack, smoothing at the *panel* level.

    The window sizes are interpreted in number of *panels* (i.e., the discretization that depends on ne_half),
    not in dense plotting samples. This targets the panel-scale sawtooth artifacts directly.

    - cod_window_panels smooths bI (normal Burgers density -> primarily affects COD)
    - csd_window_panels smooths bII (tangential Burgers density -> primarily affects CSD)
    """
    edge_index = int(edge_index)
    sol = getattr(res, "sol", None)
    if not isinstance(sol, dict) or str(sol.get("solver_option", "")).lower() != "parametrized_crack":
        raise AttributeError("This reconstruction is only valid for solver_option='parametrized_crack'.")

    # Map edge -> polyline id
    e2p = sol.get("parametrized_edge_to_polyline", {})
    pid = int(e2p.get(edge_index, -1))
    polylines = list(sol.get("parametrized_polylines", []))
    poly_solutions = list(sol.get("polyline_solutions", []))
    if pid < 0 or pid >= len(polylines) or pid >= len(poly_solutions):
        raise ValueError("polyline metadata/solution missing or inconsistent.")

    meta = dict(polylines[pid])
    solp = dict(poly_solutions[pid])

    path_vids = [int(v) for v in meta.get("path_vertex_ids", [])]
    path_edges = [int(i) for i in meta.get("path_edge_indices", [])]
    segL = np.asarray(meta.get("segment_lengths", []), float)
    Ltot = float(meta.get("total_length", np.sum(segL)))

    if edge_index not in path_edges:
        raise ValueError(f"Edge {edge_index} is not in polyline path for pid={pid}.")

    # Build cumulative arc-length at path vertices
    s_vert = np.zeros(len(path_vids), float)
    if len(segL) == len(path_vids) - 1:
        s_vert[1:] = np.cumsum(segL)

    # Determine this edge's position in the path
    k = path_edges.index(edge_index)
    s0 = float(s_vert[k])
    Le = float(segL[k])
    a = 0.5 * Le

    # Determine path direction relative to stored edge orientation
    edge = res.calc.network.edges[edge_index]
    v_start = int(path_vids[k])
    v_end = int(path_vids[k + 1])
    if int(edge.v0) == v_start and int(edge.v1) == v_end:
        dir_sign = +1
    elif int(edge.v1) == v_start and int(edge.v0) == v_end:
        dir_sign = -1
    else:
        dir_sign = +1  # fallback

    # Sample points along the edge in its local x in [-a,a]
    n_pts = int(max(200, n_pts))
    x_edge = np.linspace(-a, a, n_pts)

    # Map x_edge -> arc-length s along the polyline
    if dir_sign == +1:
        s = s0 + (x_edge + a)
    else:
        s = s0 + (a - x_edge)

    # Panel data (solved Burgers densities)
    s_nodes = np.asarray(solp.get("s_nodes"), float)
    bI = np.asarray(solp.get("bI"), float)
    bII = np.asarray(solp.get("bII"), float)
    t_col = np.asarray(solp.get("t_col"), float)
    n_col = np.asarray(solp.get("n_col"), float)
    ds = np.asarray(solp.get("ds"), float)
    Np = int(solp.get("Np", len(bI)))
    if Np <= 0:
        raise ValueError("Invalid number of panels (Np).")

    # Defensive shape fixes
    bI = bI.reshape(-1,)[:Np]
    bII = bII.reshape(-1,)[:Np]
    ds = ds.reshape(-1,)[:Np]
    t_col = t_col.reshape(-1, 2)[:Np, :]
    n_col = n_col.reshape(-1, 2)[:Np, :]

    if len(s_nodes) != Np + 1:
        s_nodes = np.linspace(0.0, Ltot, Np + 1)

    # -------- Panel-level smoothing (the key fix) --------
    if int(cod_window_panels) and int(cod_window_panels) > 1:
        bI = _moving_average_nan(bI, int(cod_window_panels))
    if int(csd_window_panels) and int(csd_window_panels) > 1:
        bII = _moving_average_nan(bII, int(csd_window_panels))

    # cumulative jump at panel nodes
    J_nodes = np.zeros((Np + 1, 2), float)
    for i in range(Np):
        dJ = (bII[i] * t_col[i] + bI[i] * n_col[i]) * float(ds[i])
        J_nodes[i + 1] = J_nodes[i] + dJ

    if enforce_global_tip_zero and Ltot > 0:
        # Remove affine gauge so J(0)=J(L)=0
        J_end = J_nodes[-1].copy()
        if np.linalg.norm(J_end) > 0:
            alpha = (s_nodes / float(Ltot)).reshape(-1, 1)
            J_nodes = J_nodes - alpha * J_end.reshape(1, 2)

    # interpolate J at sample s
    s_clip = np.clip(s, 0.0, float(Ltot))
    idx = np.searchsorted(s_nodes, s_clip, side="right") - 1
    idx = np.clip(idx, 0, Np - 1)
    sL = s_nodes[idx]
    sR = s_nodes[idx + 1]
    w = np.where(sR > sL, (s_clip - sL) / (sR - sL), 0.0)
    J = (1.0 - w).reshape(-1, 1) * J_nodes[idx] + w.reshape(-1, 1) * J_nodes[idx + 1]

    # Convert global jump to edge-local (CSD,COD)
    v0 = res.calc.network.V(edge.v0)
    v1 = res.calc.network.V(edge.v1)
    p0 = np.array([float(v0.x), float(v0.y)], float)
    p1 = np.array([float(v1.x), float(v1.y)], float)
    t = p1 - p0
    L = float(np.hypot(t[0], t[1]))
    if L <= 0:
        R = np.eye(2)
    else:
        ex = t / L
        ey = np.array([-ex[1], ex[0]])
        R = np.column_stack([ex, ey])  # local (t,n) -> global
    loc = (R.T @ J.T).T
    CSD = loc[:, 0]
    COD = loc[:, 1]
    return x_edge, COD, CSD

# -------------------------
# Plotter
# -------------------------

class DCEPlotterV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results          # DCEResultsV4 (network)
        self.calc = results.calc    # DCENetworkStaticV4
        self.out_dir = _ensure_dir(out_dir)

    # ---- graph / geometry ----

    def plot_network_graph(self, units: str = "mm", annotate: bool = True):
        V = self.calc.network.vertices
        E = self.calc.network.edges

        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Crack Network (Graph)")

        # edges
        for e in E:
            v0 = next(v for v in V if int(v.id) == int(e.v0))
            v1 = next(v for v in V if int(v.id) == int(e.v1))
            ax.plot([v0.x * s, v1.x * s], [v0.y * s, v1.y * s], color="k", lw=2)
            if annotate:
                xm = 0.5 * (v0.x + v1.x) * s
                ym = 0.5 * (v0.y + v1.y) * s
                ax.text(xm, ym, f"e{int(e.id)}", ha="center", va="center")

        # vertices
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, v in enumerate(V):
            ax.scatter([v.x * s], [v.y * s], s=50, color=colors[i % len(colors)] if colors else None)
            if annotate:
                ax.text(v.x * s, v.y * s, f"v{int(v.id)}", ha="left", va="bottom")

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        _save_fig(fig, self.out_dir, "network_graph", dpi=150)
        return fig

    def plot_deformed_network(
        self,
        n_theta: int = 4000,
        scale: float = 1e6,
        show_faces: bool = True,
        n_pts_per_edge: int = 400,
        units: str = "mm",
        *,
        use_panel_midpoints: bool = True,
        cod_smooth_window: int = 0,
        csd_smooth_window: int = 0,
    ):
        """Plot deformed crack network using network vertices/edges for geometry.

        scale magnifies COD/CSD for visibility (unitless multiplier applied to displacement).

        cod_smooth_window / csd_smooth_window apply an optional moving-average smoother
        (NaN-aware, reflect-padded). Set to 0 or 1 to disable.
        """
        # -------------------------------------------------
# Resolve smoothing windows (panel-level)
# -------------------------------------------------
        ne_half_eff = int(
            getattr(getattr(self.res, "sol", {}), "get", lambda k, d=None: d)("ne_half", 0) or 0
        )
        if ne_half_eff <= 0:
            ne_half_eff = 10  # safe fallback

        cod_win = _resolve_smooth_window(cod_smooth_window, ne_half_eff)
        csd_win = _resolve_smooth_window(csd_smooth_window, ne_half_eff)

        
        V = self.calc.network.vertices
        E = self.calc.network.edges
        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Deformed Crack Network (scale={scale:.0e})")

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not colors:
            colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

        for edge_idx, edge in enumerate(E):
            # edge endpoints from NETWORK
            v0 = next(v for v in V if int(v.id) == int(edge.v0))
            v1 = next(v for v in V if int(v.id) == int(edge.v1))
            p0 = np.array([float(v0.x), float(v0.y)])
            p1 = np.array([float(v1.x), float(v1.y)])
            c = 0.5 * (p0 + p1)
            t = p1 - p0
            L = float(np.hypot(t[0], t[1]))
            a = 0.5 * L
            if a <= 0:
                continue
            R = _rot_from_tangent(t)  # local->global

            # -------------------------------------------------
            # Midpoint-based face plotting (default for parametrized_crack)
            # -------------------------------------------------
            if (
                bool(use_panel_midpoints)
                and show_faces
                and hasattr(self.res, "crack_face_coords_panel_midpoints")
            ):
                try:
                    xyU, xyL, _ex = self.res.crack_face_coords_panel_midpoints(
                        edge_index=int(edge_idx),
                        scale=float(scale),
                        enforce_global_tip_zero=True,
                    )
                    xyU = np.asarray(xyU, float)
                    xyL = np.asarray(xyL, float)
                    # Ensure connectivity at edge endpoints (kinks):
                    # Use displaced endpoint points computed by the post-processor
                    # (upper/lower endpoints generally differ at interior junctions).
                    if xyU.ndim == 2 and xyU.shape[1] == 2 and xyU.shape[0] >= 1:
                        pU0 = np.asarray(_ex.get("xyU0", p0), float)
                        pL0 = np.asarray(_ex.get("xyL0", p0), float)
                        pU1 = np.asarray(_ex.get("xyU1", p1), float)
                        pL1 = np.asarray(_ex.get("xyL1", p1), float)
                        xyU = np.vstack([pU0.reshape(1, 2), xyU, pU1.reshape(1, 2)])
                        xyL = np.vstack([pL0.reshape(1, 2), xyL, pL1.reshape(1, 2)])
                    if xyU.ndim == 2 and xyU.shape[1] == 2 and xyU.shape[0] >= 2:
                        ax.plot(
                            xyU[:, 0] * s,
                            xyU[:, 1] * s,
                            lw=1.5,
                            color=colors[edge_idx % len(colors)],
                        )
                        ax.plot(
                            xyL[:, 0] * s,
                            xyL[:, 1] * s,
                            lw=1.5,
                            color=colors[edge_idx % len(colors)],
                        )
                        # undeformed centerline overlay
                        ax.plot(
                            [p0[0] * s, p1[0] * s],
                            [p0[1] * s, p1[1] * s],
                            ls=":",
                            lw=1.0,
                            color="k",
                            alpha=0.6,
                        )
                        continue
                except Exception:
                    # Fall back to dense reconstruction below
                    pass

            # per-edge results

            # If this solution was obtained via solver_option="parametrized_crack",
            # reconstruct COD/CSD as a *single* arc-length field along the ordered polyline
            # to avoid artificial COD=0 at interior degree-2 vertices.
            if getattr(self.res, "is_parametrized", lambda: False)() and hasattr(self.res, "reconstruct_cod_csd_parametrized"):
                # Use a high-resolution reconstruction for correct near-tip geometry,
                # then plot directly (no need to downsample).
                n_theta_eff = int(max(int(n_theta), 8000))
                x, COD, CSD = _reconstruct_cod_csd_parametrized_smoothed(
                    self.res,
                    edge_index=int(edge_idx),
                    n_pts=n_theta_eff,
                    enforce_global_tip_zero=True,
                    cod_window_panels=int(cod_win),
                    csd_window_panels=int(csd_win),
                )
            else:
                edge_res = self.res.edge(int(edge_idx))
                x, COD, CSD = edge_res.reconstruct_cod_csd(n_theta=int(n_theta), enforce_tip_zero=True)

            x = np.asarray(x, float)
            COD = np.asarray(COD, float)
            CSD = np.asarray(CSD, float)

            # Optional smoothing: for non-parametrized edges, this acts on reconstructed samples.
            # For parametrized_crack, smoothing is applied at the *panel* level inside
            # _reconstruct_cod_csd_parametrized_smoothed (recommended).
            if not (isinstance(getattr(self.res, 'sol', None), dict) and str(getattr(self.res, 'sol', {}).get('solver_option','')).lower()=='parametrized_crack'):
                if int(cod_win) and int(cod_win) > 1:
                    COD = _moving_average_nan(COD, int(cod_win))
                if int(csd_win) and int(csd_win) > 1:
                    CSD = _moving_average_nan(CSD, int(csd_win))

            # local coordinates
            X0 = x
            Y0 = np.zeros_like(x)

            # displacement (local) for upper/lower faces
            if show_faces:
                du = 0.5 * COD
                dt = 0.5 * CSD
                # Scale should magnify only the displacement components (COD/CSD),
                # not the base geometry coordinates.
                dt_s = dt * float(scale)
                du_s = du * float(scale)

                Yup = du_s
                Ylo = -du_s
                Xup = X0 + dt_s
                Xlo = X0 - dt_s

                Pup = (R @ np.vstack([Xup, Yup])) + c.reshape(2, 1)
                Plo = (R @ np.vstack([Xlo, Ylo])) + c.reshape(2, 1)

                ax.plot(Pup[0] * s, Pup[1] * s, lw=1.5, color=colors[edge_idx % len(colors)])
                ax.plot(Plo[0] * s, Plo[1] * s, lw=1.5, color=colors[edge_idx % len(colors)])
            else:
                P0 = (R @ np.vstack([X0, Y0])) + c.reshape(2, 1)
                ax.plot(P0[0] * s, P0[1] * s, lw=2.0, color=colors[edge_idx % len(colors)])

            # undeformed centerline overlay
            ax.plot([p0[0] * s, p1[0] * s], [p0[1] * s, p1[1] * s], ls=":", lw=1.0, color="k", alpha=0.6)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        _save_fig(fig, self.out_dir, "network_deformed", dpi=150)
        return fig

    # ---- displacement field ----

    def plot_displacement_vector_field(
        self,
        extent_factor: float = 1.2,
        n_grid: int = 41,
        scale: float = 1.0,
        disp_scale: float = 1e6,
        units: str = "mm",
        mask_cracks: bool = True,
    ):
        """Quiver of displacement field in global domain.

        disp_scale multiplies displacement values before plotting (for visibility).
        """
        V = self.calc.network.vertices
        ext = _network_extent(V)
        half_w = 0.5 * float(extent_factor) * ext

        # grid
        x = np.linspace(-half_w, half_w, int(n_grid))
        y = np.linspace(-half_w, half_w, int(n_grid))
        Xg, Yg = np.meshgrid(x, y)

        ux, uy = self.res.displacement_field_global(Xg, Yg, add_remote=False)
        ux = np.asarray(ux, float) * float(disp_scale)
        uy = np.asarray(uy, float) * float(disp_scale)

        # ------------------------------------------------------------
        # Parametrized-crack visualization: enforce a *single* branch
        # convention for the displacement field.
        #
        # The dislocation-based displacement is multi-valued (log/atan2
        # terms). Summing per-edge kernels can yield an artificial global
        # jump (a "branch cut") that is unrelated to the physical crack.
        # For solver_option="parametrized_crack" we use the stored
        # equivalent crack frame to classify points into two half-planes
        # and remove the spurious jump by applying a constant shift to
        # one side. This is a visualization gauge choice and does not
        # modify stresses or COD/CSD.
        sol = getattr(self.res, "sol", None)
        if isinstance(sol, dict) and str(sol.get("solver_option", "")).lower() == "parametrized_crack":
            prm = sol.get("parametrized", {}) if isinstance(sol.get("parametrized", {}), dict) else {}
            ang = float(prm.get("equivalent_angle", 0.0))
            cen = np.asarray(prm.get("equivalent_center", [0.0, 0.0]), float).reshape(2,)

            # Equivalent frame: local x along tangent, local y is signed distance
            t_eq = np.array([math.cos(ang), math.sin(ang)], float)
            Q = _rot_from_tangent(t_eq)          # local->global
            QT = Q.T                             # global->local

            P = np.stack([Xg, Yg], axis=0)  # (2, Ny, Nx)
            Pl = QT @ (P.reshape(2, -1) - cen.reshape(2, 1))
            yl = Pl[1, :].reshape(Xg.shape)

            # Use only points sufficiently far from the cut for robust offset estimation
            far = np.abs(yl) > (0.25 * half_w)
            finite = np.isfinite(ux) & np.isfinite(uy)
            pos = far & finite & (yl >= 0.0)
            neg = far & finite & (yl < 0.0)

            if np.any(pos) and np.any(neg):
                # Robust constant shift to align the two branches
                dux = float(np.nanmedian(ux[pos]) - np.nanmedian(ux[neg]))
                duy = float(np.nanmedian(uy[pos]) - np.nanmedian(uy[neg]))

                # Shift the negative side to match the positive side
                ux = np.where(yl < 0.0, ux + dux, ux)
                uy = np.where(yl < 0.0, uy + duy, uy)

        if mask_cracks:
            # mask near each edge segment
            for e in self.calc.network.edges:
                v0 = next(v for v in V if int(v.id) == int(e.v0))
                v1 = next(v for v in V if int(v.id) == int(e.v1))
                p0 = np.array([float(v0.x), float(v0.y)])
                p1 = np.array([float(v1.x), float(v1.y)])
                d = p1 - p0
                L2 = float(d @ d)
                if L2 <= 0:
                    continue
                # distance from point to segment
                PX = np.stack([Xg, Yg], axis=0).reshape(2, -1)
                t = ((PX.T - p0) @ d) / L2
                t = np.clip(t, 0.0, 1.0)
                proj = p0.reshape(1, 2) + t.reshape(-1, 1) * d.reshape(1, 2)
                dist = np.hypot(PX.T[:, 0] - proj[:, 0], PX.T[:, 1] - proj[:, 1]).reshape(Xg.shape)
                mask = dist < (2e-3 * ext)
                ux = np.where(mask, np.nan, ux)
                uy = np.where(mask, np.nan, uy)

        # units
        sxy = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(7, 6))
        mag = np.sqrt(ux * ux + uy * uy)
        im = ax.pcolormesh(Xg * sxy, Yg * sxy, mag, shading="auto")
        fig.colorbar(im, ax=ax, label=f"|u| * {disp_scale:g} [m]")

        ax.quiver(Xg * sxy, Yg * sxy, ux, uy, angles="xy", scale_units="xy", scale=scale)
        ax.set_title("Displacement vector field (global)")
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        _save_fig(fig, self.out_dir, "displacement_vector_field", dpi=150)
        return fig

    # ---- stress field ----

    def plot_stress_components_global(
        self,
        opts: StressPlotOptsV4,
        components: Sequence[str] = ("sxx", "syy", "sxy"),
    ) -> Dict[str, plt.Figure]:
        opts = StressPlotOptsV4(**vars(opts))  # normalize alias

        V = self.calc.network.vertices
        ext = _network_extent(V)
        half_w = 0.5 * float(opts.extent_factor) * ext

        x = np.linspace(-half_w, half_w, int(opts.n_grid))
        y = np.linspace(-half_w, half_w, int(opts.n_grid))
        Xg, Yg = np.meshgrid(x, y)

        sxx, syy, sxy = self.res.stress_field_global(Xg, Yg, add_remote=bool(opts.add_remote))
        fields = {"sxx": sxx / 1e6, "syy": syy / 1e6, "sxy": sxy / 1e6}  # MPa

        # remote reference
        sig = getattr(self.calc, "applied_tensor", None)
        if callable(sig):
            remote = np.asarray(sig(), float) / 1e6
        else:
            # fall back
            ap = self.calc.applied
            remote = np.array([[ap.sigma_xx, ap.sigma_xy], [ap.sigma_xy, ap.sigma_yy]], float) / 1e6

        ref_map = {"sxx": float(remote[0, 0]), "syy": float(remote[1, 1]), "sxy": float(remote[0, 1])}

        figs: Dict[str, plt.Figure] = {}
        for comp in components:
            comp = comp.lower()
            if comp not in fields:
                continue
            Z = np.array(fields[comp], float)

            # mask near cracks
            if opts.mask_cracks:
                for e in self.calc.network.edges:
                    v0 = next(v for v in V if int(v.id) == int(e.v0))
                    v1 = next(v for v in V if int(v.id) == int(e.v1))
                    p0 = np.array([float(v0.x), float(v0.y)])
                    p1 = np.array([float(v1.x), float(v1.y)])
                    d = p1 - p0
                    L2 = float(d @ d)
                    if L2 <= 0:
                        continue
                    PX = np.stack([Xg, Yg], axis=0).reshape(2, -1)
                    t = ((PX.T - p0) @ d) / L2
                    t = np.clip(t, 0.0, 1.0)
                    proj = p0.reshape(1, 2) + t.reshape(-1, 1) * d.reshape(1, 2)
                    dist = np.hypot(PX.T[:, 0] - proj[:, 0], PX.T[:, 1] - proj[:, 1]).reshape(Xg.shape)
                                        # Ensure mask is at least a few grid cells wide so inclined segments
                    # do not appear as sparse dots on the Cartesian grid
                    dx = x[1] - x[0]
                    dy = y[1] - y[0]
                    h = max(abs(dx), abs(dy))
                    width = max(opts.mask_width_factor * ext, 2.0 * h)
                    Z[dist < width] = np.nan


            # Optional tail clipping to improve visibility (does not change the underlying solution).
            Zp = _apply_clip_percentiles(Z, getattr(opts, "clip_percentiles", None))

            ref = abs(ref_map.get(comp, 0.0))
            vmin, vmax = _robust_vmin_vmax(Zp, opts, ref=ref)

            norm = None
            norm_name = str(getattr(opts, "norm", "linear") or "linear").lower()
            if norm_name == "symlog":
                from matplotlib.colors import SymLogNorm
                norm = SymLogNorm(
                    linthresh=float(getattr(opts, "symlog_linthresh", 1.0)),
                    linscale=float(getattr(opts, "symlog_linscale", 1.0)),
                    vmin=float(vmin),
                    vmax=float(vmax),
                    base=float(getattr(opts, "symlog_base", 10.0)),
                )
                levels = _symlog_levels(
                    vmin, vmax,
                    float(getattr(opts, "symlog_linthresh", 1.0)),
                    int(opts.n_bands) + 1,
                )
            elif norm_name == "log":
                from matplotlib.colors import LogNorm
                Zpos = Zp[np.isfinite(Zp) & (Zp > 0)]
                vmin_pos = float(np.nanmin(Zpos)) if Zpos.size else 1e-12
                vmax_pos = float(np.nanmax(Zpos)) if Zpos.size else max(1.0, vmin_pos * 10.0)
                vmin_pos = max(vmin_pos, 1e-12)
                vmax_pos = max(vmax_pos, vmin_pos * 1.01)
                norm = LogNorm(vmin=vmin_pos, vmax=vmax_pos)
                levels = np.geomspace(vmin_pos, vmax_pos, int(opts.n_bands) + 1)
            else:
                levels = np.linspace(vmin, vmax, int(opts.n_bands) + 1)

            fig, ax = plt.subplots(figsize=(6.2, 5.5))
            cf = ax.contourf(
                Xg * 1e3, Yg * 1e3, Zp,
                levels=levels,
                cmap=opts.cmap,
                norm=norm,
                extend=str(getattr(opts, "extend", "both")),
            )
            c = ax.contour(Xg * 1e3, Yg * 1e3, Zp, levels=levels, colors="k", linewidths=0.5, alpha=0.55)
            if opts.label_contours:
                ax.clabel(c, inline=True, fontsize=8, fmt=opts.label_fmt)
            fig.colorbar(cf, ax=ax, label="Stress [MPa]")

            ax.set_title(comp)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.axis("equal")
            ax.grid(True, alpha=0.15)

            _save_fig(fig, self.out_dir, f"stress_{comp}", dpi=opts.dpi)
            figs[comp] = fig

        return figs


# Backwards-compatible aliases used in some notebooks
StressPlotOpts = StressPlotOptsV4