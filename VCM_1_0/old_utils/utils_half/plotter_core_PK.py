"""DCEPlotterV4 implementation."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

from .plot_opts import (
    StressPlotOptsV4, ensure_dir, save_fig, apply_clip_percentiles,
    robust_vmin_vmax, symlog_levels, rot_from_tangent, network_extent
)
from .plot_smooth import moving_average_nan, resolve_smooth_window
from .plot_reconstruct import reconstruct_cod_csd_parametrized_smoothed


class DCEPlotterV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_network_graph(self, units: str = "mm", annotate: bool = True):
        V = self.calc.network.vertices
        E = self.calc.network.edges
        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Crack Network (Graph)")

        for e in E:
            v0 = next(v for v in V if int(v.id) == int(e.v0))
            v1 = next(v for v in V if int(v.id) == int(e.v1))
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

        save_fig(fig, self.out_dir, "network_graph", dpi=150)
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
        ne_half_eff = int(getattr(self.res, "sol", {}).get("ne_half", 0) or 0)
        if ne_half_eff <= 0:
            ne_half_eff = 10

        cod_win = resolve_smooth_window(cod_smooth_window, ne_half_eff)
        csd_win = resolve_smooth_window(csd_smooth_window, ne_half_eff)

        V = self.calc.network.vertices
        E = self.calc.network.edges
        s = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Deformed Crack Network (scale={scale:.0e})")

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not colors:
            colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

        for edge_idx, edge in enumerate(E):
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
            R = rot_from_tangent(t)

            if (
                bool(use_panel_midpoints)
                and show_faces
                and hasattr(self.res, "crack_face_coords_panel_midpoints")
            ):
                try:
                    xyU, xyL, extra = self.res.crack_face_coords_panel_midpoints(
                        edge_index=int(edge_idx),
                        scale=float(scale),
                        enforce_global_tip_zero=True,
                    )
                    xyU = np.asarray(xyU, float)
                    xyL = np.asarray(xyL, float)
                    if xyU.ndim == 2 and xyU.shape[1] == 2 and xyU.shape[0] >= 1:
                        pU0 = np.asarray(extra.get("xyU0", p0), float)
                        pL0 = np.asarray(extra.get("xyL0", p0), float)
                        pU1 = np.asarray(extra.get("xyU1", p1), float)
                        pL1 = np.asarray(extra.get("xyL1", p1), float)
                        xyU = np.vstack([pU0.reshape(1, 2), xyU, pU1.reshape(1, 2)])
                        xyL = np.vstack([pL0.reshape(1, 2), xyL, pL1.reshape(1, 2)])
                    if xyU.ndim == 2 and xyU.shape[1] == 2 and xyU.shape[0] >= 2:
                        ax.plot(xyU[:, 0] * s, xyU[:, 1] * s, lw=1.5, color=colors[edge_idx % len(colors)])
                        ax.plot(xyL[:, 0] * s, xyL[:, 1] * s, lw=1.5, color=colors[edge_idx % len(colors)])
                        ax.plot([p0[0] * s, p1[0] * s], [p0[1] * s, p1[1] * s], ls=":", lw=1.0, color="k", alpha=0.6)
                        continue
                except Exception:
                    pass

            if getattr(self.res, "is_parametrized", lambda: False)():
                n_theta_eff = int(max(int(n_theta), 8000))
                x, COD, CSD = reconstruct_cod_csd_parametrized_smoothed(
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

            if not getattr(self.res, "is_parametrized", lambda: False)():
                if int(cod_win) and int(cod_win) > 1:
                    COD = moving_average_nan(COD, int(cod_win))
                if int(csd_win) and int(csd_win) > 1:
                    CSD = moving_average_nan(CSD, int(csd_win))

            X0 = x
            Y0 = np.zeros_like(x)

            if show_faces:
                du = 0.5 * COD
                dt = 0.5 * CSD
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

            ax.plot([p0[0] * s, p1[0] * s], [p0[1] * s, p1[1] * s], ls=":", lw=1.0, color="k", alpha=0.6)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        save_fig(fig, self.out_dir, "network_deformed", dpi=150)
        return fig

    def plot_displacement_vector_field(
        self,
        extent_factor: float = 1.2,
        n_grid: int = 41,
        scale: float = 1.0,
        disp_scale: float = 1e6,
        units: str = "mm",
        mask_cracks: bool = True,
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

        ax.quiver(Xg * sxy, Yg * sxy, ux, uy, angles="xy", scale_units="xy", scale=scale)
        ax.set_title("Displacement vector field (global)")
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        save_fig(fig, self.out_dir, "displacement_vector_field", dpi=150)
        return fig

    def plot_stress_components_global(
        self,
        opts: StressPlotOptsV4,
        components: Sequence[str] = ("sxx", "syy", "sxy"),
    ) -> Dict[str, plt.Figure]:
        opts = StressPlotOptsV4(**vars(opts))

        V = self.calc.network.vertices
        ext = network_extent(V)
        half_w = 0.5 * float(opts.extent_factor) * ext

        x = np.linspace(-half_w, half_w, int(opts.n_grid))
        y = np.linspace(-half_w, half_w, int(opts.n_grid))
        Xg, Yg = np.meshgrid(x, y)

        sxx, syy, sxy = self.res.stress_field_global(Xg, Yg, add_remote=bool(opts.add_remote))
        fields = {"sxx": sxx / 1e6, "syy": syy / 1e6, "sxy": sxy / 1e6}

        sig = getattr(self.calc, "applied_tensor", None)
        if callable(sig):
            remote = np.asarray(sig(), float) / 1e6
        else:
            ap = self.calc.applied
            remote = np.array([[ap.sigma_xx, ap.sigma_xy], [ap.sigma_xy, ap.sigma_yy]], float) / 1e6
        ref_map = {"sxx": float(remote[0, 0]), "syy": float(remote[1, 1]), "sxy": float(remote[0, 1])}

        figs: Dict[str, plt.Figure] = {}
        for comp in components:
            comp = comp.lower()
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

            save_fig(fig, self.out_dir, f"stress_{comp}", dpi=opts.dpi)
            figs[comp] = fig

        return figs

    def plot_b_content_vectors(
        self,
        *,
        vec_scale: float = 1.0,
        units: str = "mm",
        show_tips: bool = True,
        show_junctions: bool = True,
    ):
        """Plot B-content vectors (integrated density) at junctions and tips.

        Notes
        -----
        - This plots the *kinematic* integrated density content (historically printed as "B").
        - It is not a PK force. Use :meth:`plot_pk_force_vectors` for PK-style forces.
        """
        pk = PKProcessor(self.res)
        V = self.calc.network.vertices
        E = self.calc.network.edges
        deg = pk.degree_map()

        sxy = 1e3 if units.lower() == "mm" else 1.0
        vpos = {int(v.id): np.array([float(v.x), float(v.y)], float) for v in V}

        # Build vertex incidence from network edges
        v_inc = {}
        for ei, e in enumerate(E):
            v_inc.setdefault(int(e.v0), []).append((ei, "start"))
            v_inc.setdefault(int(e.v1), []).append((ei, "end"))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("B-content vectors (junctions and tips)")

        # Draw network midlines
        for ei, e in enumerate(E):
            p0 = vpos[int(e.v0)]
            p1 = vpos[int(e.v1)]
            ax.plot([p0[0] * sxy, p1[0] * sxy], [p0[1] * sxy, p1[1] * sxy], ls=":", lw=1.0, color="k", alpha=0.5)

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C3", "C4", "C5"]

        # Tips
        if show_tips:
            for vid, d in deg.items():
                if int(d) != 1:
                    continue
                items = v_inc.get(int(vid), [])
                if not items:
                    continue
                ei, which = items[0]
                B = pk.B_content_edge(int(ei))
                p = vpos[int(vid)]
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=40, color="k")
                if B is None:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{vid}: B unavailable", fontsize=8, ha="left", va="bottom")
                    continue
                B = np.asarray(B, float).reshape(2,) * float(vec_scale)
                ax.arrow(p[0] * sxy, p[1] * sxy, B[0], B[1], head_width=0.18, length_includes_head=True, color="C0", alpha=0.85)

        # Junctions: per-incident vectors and sum
        if show_junctions:
            for vid, d in deg.items():
                if int(d) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=55, color="k")
                ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", ha="left", va="bottom")

                Bsum = np.zeros(2, float)
                items = v_inc.get(int(vid), [])
                for i, (ei, which) in enumerate(items):
                    B = pk.B_content_edge(int(ei))
                    if B is None:
                        continue
                    B = np.asarray(B, float).reshape(2,) * float(vec_scale)
                    Bsum += B
                    ax.arrow(p[0] * sxy, p[1] * sxy, B[0], B[1], head_width=0.16, length_includes_head=True, color=colors[i % len(colors)], alpha=0.85)

                ax.arrow(p[0] * sxy, p[1] * sxy, Bsum[0], Bsum[1], head_width=0.20, length_includes_head=True, color="k", alpha=0.65)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

        save_fig(fig, self.out_dir, "b_content_vectors", dpi=150)
        return fig

    def plot_pk_force_vectors(
        self,
        *,
        force_scale: float = 1.0,
        exclude_self: bool = True,
        units: str = "mm",
        show_tips: bool = True,
        show_junctions: bool = True,
        n_samples_end: int = 32,
    ):
        """Plot PK-style force vectors at junctions and tips.

        This calls :meth:`processor_PK.PKProcessor.F_PK_edge_tip` at the appropriate edge ends.

        Requirements
        ------------
        For `exclude_self=True`, your `results` object must implement:
            `stress_field_global_excluding(edge_index, X, Y, add_remote=True)`
        Otherwise set `exclude_self=False` and provide:
            `stress_field_global(X, Y, add_remote=True)`
        """
        from .processor_PK import PKProcessor

        pk = PKProcessor(self.res)
        V = self.calc.network.vertices
        E = self.calc.network.edges
        deg = pk.degree_map()

        sxy = 1e3 if units.lower() == "mm" else 1.0
        vpos = {int(v.id): np.array([float(v.x), float(v.y)], float) for v in V}

        # Build vertex incidence from network edges
        v_inc = {}
        for ei, e in enumerate(E):
            v_inc.setdefault(int(e.v0), []).append((ei, "start"))
            v_inc.setdefault(int(e.v1), []).append((ei, "end"))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("F_PK vectors (junctions and tips)")

        for ei, e in enumerate(E):
            p0 = vpos[int(e.v0)]
            p1 = vpos[int(e.v1)]
            ax.plot([p0[0] * sxy, p1[0] * sxy], [p0[1] * sxy, p1[1] * sxy], ls=":", lw=1.0, color="k", alpha=0.5)

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C3", "C4", "C5"]

        # Tips
        if show_tips:
            for vid, d in deg.items():
                if int(d) != 1:
                    continue
                items = v_inc.get(int(vid), [])
                if not items:
                    continue
                ei, which = items[0]
                p = vpos[int(vid)]
                F = pk.F_PK_edge_tip(int(ei), at=str(which), exclude_self=bool(exclude_self), n_samples=int(n_samples_end))
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=40, color="k")
                if F is None:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{vid}: F_PK unavailable", fontsize=8, ha="left", va="bottom")
                    continue
                F = np.asarray(F, float).reshape(2,) * float(force_scale)
                ax.arrow(p[0] * sxy, p[1] * sxy, F[0], F[1], head_width=0.18, length_includes_head=True, color="C3", alpha=0.85)

        # Junctions
        if show_junctions:
            for vid, d in deg.items():
                if int(d) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=55, color="k")
                ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", ha="left", va="bottom")

                Fsum = np.zeros(2, float)
                items = v_inc.get(int(vid), [])
                for i, (ei, which) in enumerate(items):
                    F = pk.F_PK_edge_tip(int(ei), at=str(which), exclude_self=bool(exclude_self), n_samples=int(n_samples_end))
                    if F is None:
                        continue
                    F = np.asarray(F, float).reshape(2,) * float(force_scale)
                    Fsum += F
                    ax.arrow(p[0] * sxy, p[1] * sxy, F[0], F[1], head_width=0.16, length_includes_head=True, color=colors[i % len(colors)], alpha=0.85)

                ax.arrow(p[0] * sxy, p[1] * sxy, Fsum[0], Fsum[1], head_width=0.20, length_includes_head=True, color="k", alpha=0.65)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

        save_fig(fig, self.out_dir, "pk_force_vectors", dpi=150)
        return fig
    
    def plot_b_content_vectors(
        self,
        vec_scale: float = 1.0,
        show_tips: bool = True,
        show_junctions: bool = True,
    ):
        """
        Plot Burgers-content (B) vectors at junctions and degree-1 tips.
        Pure post-processing; does not modify solver results.
        """
        
        pk = PKProcessor(self.res)

        V = self.calc.network.vertices
        E = self.calc.network.edges

        # vertex degree
        deg = {int(v.id): 0 for v in V}
        for e in E:
            deg[int(e.v0)] += 1
            deg[int(e.v1)] += 1

        fig, ax = plt.subplots(figsize=(7, 7))

        # --- junctions ---
        if show_junctions:
            for v in V:
                vid = int(v.id)
                if deg.get(vid, 0) < 2:
                    continue

                Bsum = np.zeros(2)
                for ei, e in enumerate(E):
                    if int(e.v0) == vid or int(e.v1) == vid:
                        Bsum += pk.B_content_edge(ei)

                ax.quiver(
                    v.x,
                    v.y,
                    Bsum[0],
                    Bsum[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0 / vec_scale,
                    color="r",
                    width=0.006,
                )

        # --- tips ---
        if show_tips:
            for ei, e in enumerate(E):
                for which, vid in [("start", int(e.v0)), ("end", int(e.v1))]:
                    if deg.get(vid, 0) != 1:
                        continue

                    v = next(vv for vv in V if int(vv.id) == vid)
                    B = pk.B_content_edge(ei)

                    ax.quiver(
                        v.x,
                        v.y,
                        B[0],
                        B[1],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0 / vec_scale,
                        color="k",
                        width=0.004,
                    )

        ax.set_aspect("equal")
        ax.set_title("B-content vectors")
        ax.grid(True, alpha=0.3)

        plt.show()

