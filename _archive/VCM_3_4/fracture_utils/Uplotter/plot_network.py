"""Network plotting utilities (graph + basic geometry).

This version includes optional overlay of parametrized arc polylines stored in:
    res.sol["parametrized_polylines"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from .opts import ensure_dir, save_fig


def _sample_arc_xy(meta: Dict, n: int = 200) -> np.ndarray:
    c = np.asarray(meta.get("arc_center", [0.0, 0.0]), float).reshape(2,)
    r = float(meta.get("arc_radius", 0.0))
    th0 = float(meta.get("arc_theta0", 0.0))
    dth = float(meta.get("arc_dtheta", 0.0))
    t = np.linspace(0.0, 1.0, max(2, int(n)))
    th = th0 + dth * t
    xy = c.reshape(1, 2) + r * np.c_[np.cos(th), np.sin(th)]
    return xy


class DCEPlotterNetworkV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_network_graph(
        self,
        units: str = "mm",
        annotate: bool = True,
        *,
        show: bool = True,
        save: bool = True,
        font_size: int = 16,
        label_offset_frac: float = 0.025,
    ):
        V = self.calc.network.vertices
        E = self.calc.network.edges
        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(7, 7))

        vmap = {int(v.id): v for v in V}

        # Bounding-box-based offset so labels never overlap the edges.
        xs_all = np.array([v.x * s for v in V], dtype=float)
        ys_all = np.array([v.y * s for v in V], dtype=float)
        span = float(max(np.ptp(xs_all), np.ptp(ys_all), 1.0))
        d_off = label_offset_frac * span  # absolute label-offset distance

        bbox_kwargs = dict(
            boxstyle="round,pad=0.18",
            fc="white",
            ec="0.6",
            lw=0.6,
            alpha=0.92,
        )

        # Draw network edges + perpendicular edge labels
        for e in E:
            v0 = vmap[int(e.v0)]
            v1 = vmap[int(e.v1)]
            x0, y0 = v0.x * s, v0.y * s
            x1, y1 = v1.x * s, v1.y * s
            ax.plot([x0, x1], [y0, y1], color="k", lw=2)
            if annotate:
                xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                dx, dy = x1 - x0, y1 - y0
                L = max(np.hypot(dx, dy), 1e-12)
                # unit perpendicular (rotated +90 deg)
                px, py = -dy / L, dx / L
                ax.text(
                    xm + d_off * px,
                    ym + d_off * py,
                    f"e{int(e.id)}",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    bbox=bbox_kwargs,
                )

        # Vertex incidence (so we can offset vertex labels away from the
        # incident edges instead of dumping them on top of the line).
        incidence: Dict[int, list] = {int(v.id): [] for v in V}
        for e in E:
            incidence[int(e.v0)].append(int(e.v1))
            incidence[int(e.v1)].append(int(e.v0))

        # Draw vertices and offset their labels away from incident edges
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, v in enumerate(V):
            vx, vy = v.x * s, v.y * s
            ax.scatter([vx], [vy], s=70,
                       color=colors[i % len(colors)] if colors else None,
                       zorder=4)
            if annotate:
                # Average unit vector to all neighbours
                ux, uy = 0.0, 0.0
                for nb_id in incidence.get(int(v.id), []):
                    nb_v = vmap.get(nb_id)
                    if nb_v is None:
                        continue
                    dx = nb_v.x * s - vx
                    dy = nb_v.y * s - vy
                    L = max(np.hypot(dx, dy), 1e-12)
                    ux += dx / L
                    uy += dy / L
                norm = np.hypot(ux, uy)
                if norm < 1e-9:
                    # isolated vertex or symmetric incidence: default upper-right
                    ox, oy = 1.0, 1.0
                else:
                    # place the label opposite the average incident direction
                    ox, oy = -ux / norm, -uy / norm
                ax.text(
                    vx + d_off * ox,
                    vy + d_off * oy,
                    f"v{int(v.id)}",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight="bold",
                    bbox=bbox_kwargs,
                    zorder=5,
                )

        # Overlay parametrized arcs (if present)
        sol = getattr(self.res, "sol", None)
        if isinstance(sol, dict):
            polylines = list(sol.get("parametrized_polylines", []))
            for meta in polylines:
                if str(meta.get("kind", "polyline")).lower() == "arc":
                    xy = _sample_arc_xy(meta, n=300)
                    ax.plot(xy[:, 0] * s, xy[:, 1] * s, color="tab:orange", lw=2.0, alpha=0.9)

        ax.set_xlabel(f"x [{units}]", fontsize=font_size)
        ax.set_ylabel(f"y [{units}]", fontsize=font_size)
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        if save:
            save_fig(fig, self.out_dir, "network_graph", dpi=150)
        if show:
            plt.show()
        return fig, ax
