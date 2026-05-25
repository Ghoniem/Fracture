"""Global stress plotting utilities."""
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


class DCEPlotterStressV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_stress_components_global(
        self,
        opts: StressPlotOptsV4,
        components: Sequence[str] = ("sxx", "syy", "sxy"),
        *,
        show: bool = False,
        save: bool = True,
    ) -> Dict[str, plt.Figure]:
        opts = StressPlotOptsV4(**vars(opts))

        V = self.calc.network.vertices
        ext = network_extent(V)
        half_w = 0.5 * float(opts.extent_factor) * ext

        x = np.linspace(-half_w, half_w, int(opts.n_grid))
        y = np.linspace(-half_w, half_w, int(opts.n_grid))
        Xg, Yg = np.meshgrid(x, y)

        sxx, syy, sxy = self.res.stress_field_global(Xg, Yg, add_remote=bool(opts.add_remote))
        fields = {"sxx": np.asarray(sxx, float) / 1e6,
                  "syy": np.asarray(syy, float) / 1e6,
                  "sxy": np.asarray(sxy, float) / 1e6}

        sig = getattr(self.calc, "applied_tensor", None)
        if callable(sig):
            remote = np.asarray(sig(), float) / 1e6
        else:
            ap = self.calc.applied
            remote = np.array([[ap.sigma_xx, ap.sigma_xy], [ap.sigma_xy, ap.sigma_yy]], float) / 1e6
        ref_map = {"sxx": float(remote[0, 0]), "syy": float(remote[1, 1]), "sxy": float(remote[0, 1])}

        figs: Dict[str, plt.Figure] = {}
        for comp in components:
            comp = str(comp).lower()
            if comp not in fields:
                continue
            Z = np.array(fields[comp], float)

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
                    tt = ((PX.T - p0) @ d) / L2
                    tt = np.clip(tt, 0.0, 1.0)
                    proj = p0.reshape(1, 2) + tt.reshape(-1, 1) * d.reshape(1, 2)
                    dist = np.hypot(PX.T[:, 0] - proj[:, 0], PX.T[:, 1] - proj[:, 1]).reshape(Xg.shape)

                    dx = x[1] - x[0]
                    dy = y[1] - y[0]
                    h = max(abs(dx), abs(dy))
                    width = max(opts.mask_width_factor * ext, 2.0 * h)
                    Z[dist < width] = np.nan

            Zp = apply_clip_percentiles(Z, getattr(opts, "clip_percentiles", None))
            ref = abs(ref_map.get(comp, 0.0))
            vmin, vmax = robust_vmin_vmax(Zp, opts, ref=ref)

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
                levels = symlog_levels(vmin, vmax, float(getattr(opts, "symlog_linthresh", 1.0)), int(opts.n_bands) + 1)
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
            cf = ax.contourf(Xg * 1e3, Yg * 1e3, Zp, levels=levels, cmap=opts.cmap, norm=norm, extend=str(getattr(opts, "extend", "both")))
            c = ax.contour(Xg * 1e3, Yg * 1e3, Zp, levels=levels, colors="k", linewidths=0.5, alpha=0.55)
            if opts.label_contours:
                ax.clabel(c, inline=True, fontsize=8, fmt=opts.label_fmt)
            fig.colorbar(cf, ax=ax, label="Stress [MPa]")

            ax.set_title(comp)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.axis("equal")
            ax.grid(True, alpha=0.15)

            if save:
                save_fig(fig, self.out_dir, f"stress_{comp}", dpi=opts.dpi)
            if show:
                plt.show()
            figs[comp] = fig

        return figs
