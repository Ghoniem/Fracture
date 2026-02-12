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
    ):
        V = self.calc.network.vertices
        E = self.calc.network.edges
        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Crack Network (Graph)")

        vmap = {int(v.id): v for v in V}

        # Draw network edges
        for e in E:
            v0 = vmap[int(e.v0)]
            v1 = vmap[int(e.v1)]
            ax.plot([v0.x * s, v1.x * s], [v0.y * s, v1.y * s], color="k", lw=2)
            if annotate:
                xm = 0.5 * (v0.x + v1.x) * s
                ym = 0.5 * (v0.y + v1.y) * s
                ax.text(xm, ym, f"e{int(e.id)}", ha="center", va="center")

        # Draw vertices
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, v in enumerate(V):
            ax.scatter([v.x * s], [v.y * s], s=50, color=colors[i % len(colors)] if colors else None)
            if annotate:
                ax.text(v.x * s, v.y * s, f"v{int(v.id)}", ha="left", va="bottom")

        # Overlay parametrized arcs (if present)
        sol = getattr(self.res, "sol", None)
        if isinstance(sol, dict):
            polylines = list(sol.get("parametrized_polylines", []))
            for meta in polylines:
                if str(meta.get("kind", "polyline")).lower() == "arc":
                    xy = _sample_arc_xy(meta, n=300)
                    ax.plot(xy[:, 0] * s, xy[:, 1] * s, color="tab:orange", lw=2.0, alpha=0.9)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        if save:
            save_fig(fig, self.out_dir, "network_graph", dpi=150)
        if show:
            plt.show()
        return fig, ax
