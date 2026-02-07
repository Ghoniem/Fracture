"""Displacement field plotting utilities."""
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


class DCEPlotterDisplacementV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_displacement_vector_field(
        self,
        extent_factor: float = 1.2,
        n_grid: int = 41,
        scale: float = 1.0,
        disp_scale: float = 5e4,
        units: str = "mm",
        mask_cracks: bool = False,
        *,
        show: bool = True,
        save: bool = True,
    ):
        V = self.calc.network.vertices
        ext = network_extent(V)
        half_w = 0.5 * float(extent_factor) * ext

        x = np.linspace(-half_w, half_w, int(n_grid))
        y = np.linspace(-half_w, half_w, int(n_grid))
        Xg, Yg = np.meshgrid(x, y)

        ux, uy = self.res.displacement_field_global(Xg, Yg, add_remote=False)
        ux = np.asarray(ux, float) * float(disp_scale)
        uy = np.asarray(uy, float) * float(disp_scale)

        sol = getattr(self.res, "sol", None)
        if isinstance(sol, dict) and str(sol.get("solver_option", "")).lower() == "parametrized_crack":
            prm = sol.get("parametrized", {}) if isinstance(sol.get("parametrized", {}), dict) else {}
            ang = float(prm.get("equivalent_angle", 0.0))
            cen = np.asarray(prm.get("equivalent_center", [0.0, 0.0]), float).reshape(2,)
            t_eq = np.array([math.cos(ang), math.sin(ang)], float)
            Q = rot_from_tangent(t_eq)
            QT = Q.T

            P = np.stack([Xg, Yg], axis=0)
            Pl = QT @ (P.reshape(2, -1) - cen.reshape(2, 1))
            yl = Pl[1, :].reshape(Xg.shape)

            far = np.abs(yl) > (0.25 * half_w)
            finite = np.isfinite(ux) & np.isfinite(uy)
            pos = far & finite & (yl >= 0.0)
            neg = far & finite & (yl < 0.0)

            if np.any(pos) and np.any(neg):
                dux = float(np.nanmedian(ux[pos]) - np.nanmedian(ux[neg]))
                duy = float(np.nanmedian(uy[pos]) - np.nanmedian(uy[neg]))
                ux = np.where(yl < 0.0, ux + dux, ux)
                uy = np.where(yl < 0.0, uy + duy, uy)

        if mask_cracks:
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
                tt = ((PX.T - p0) @ d) / L2
                tt = np.clip(tt, 0.0, 1.0)
                proj = p0.reshape(1, 2) + tt.reshape(-1, 1) * d.reshape(1, 2)
                dist = np.hypot(PX.T[:, 0] - proj[:, 0], PX.T[:, 1] - proj[:, 1]).reshape(Xg.shape)
                mask = dist < (2e-3 * ext)
                ux = np.where(mask, np.nan, ux)
                uy = np.where(mask, np.nan, uy)

        sxy = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(7, 6))
        mag = np.sqrt(ux * ux + uy * uy)
        im = ax.pcolormesh(Xg * sxy, Yg * sxy, mag, shading="auto")
        fig.colorbar(im, ax=ax, label=f"|u| * {disp_scale:g} [m]")

        # ax.quiver(Xg * sxy, Yg * sxy, ux, uy, angles="xy", scale_units="xy", scale=scale)
        # ax.set_title("Displacement vector field (global)")
        # ax.set_xlabel(f"x [{units}]")
        # ax.set_ylabel(f"y [{units}]")
        # ax.axis("equal")
        # ax.grid(True, alpha=0.25)
        
        # Calculate magnitude for reference (but don't use for coloring)

# Create clean quiver plot
        ax.quiver(Xg * sxy, Yg * sxy, ux, uy, 
                angles="xy", 
                scale_units="xy", 
                scale=scale,
                color='black',         # Solid color
                width=0.004,           # Arrow shaft width
                headwidth=3,           # Arrow head width
                headlength=4,          # Arrow head length
                headaxislength=3.5,    # Arrow head axis length
                alpha=0.7)             # Transparency

        ax.set_title("Displacement vector field (global)", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"x [{units}]", fontsize=11)
        ax.set_ylabel(f"y [{units}]", fontsize=11)
        ax.axis("equal")
        ax.grid(True, alpha=0.25, linestyle='--')

        # Optional: Set white background explicitly
        ax.set_facecolor('white')

        if save:
            save_fig(fig, self.out_dir, "displacement_vector_field", dpi=150)
        if show:
            plt.show()
        return fig
