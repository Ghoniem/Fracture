"""Network plotting utilities (graph + basic geometry)."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Sequence, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .opts import (
    StressPlotOptsV4, ensure_dir, save_fig, apply_clip_percentiles,
    robust_vmin_vmax, symlog_levels, rot_from_tangent, network_extent
)
from .smooth import moving_average_nan, resolve_smooth_window
from ..Uprocessor.PK import PKProcessor


class DCEPlotterNetworkV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_network_graph(self, units: str = "mm", annotate: bool = True, *, show: bool = True, save: bool = True):
        V = self.calc.network.vertices
        E = self.calc.network.edges
        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Crack Network (Graph)")

        vmap = {int(v.id): v for v in V}

        for e in E:
            v0 = vmap[int(e.v0)]
            v1 = vmap[int(e.v1)]
            ax.plot([v0.x * s, v1.x * s], [v0.y * s, v1.y * s], color="k", lw=2)
            if annotate:
                xm = 0.5 * (v0.x + v1.x) * s
                ym = 0.5 * (v0.y + v1.y) * s
                ax.text(xm, ym, f"e{int(e.id)}", ha="center", va="center")

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, v in enumerate(V):
            ax.scatter([v.x * s], [v.y * s], s=50, color=colors[i % len(colors)] if colors else None)
            if annotate:
                ax.text(v.x * s, v.y * s, f"v{int(v.id)}", ha="left", va="bottom")

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        if save:
            save_fig(fig, self.out_dir, "network_graph", dpi=150)
        if show:
            plt.show()
        return fig
