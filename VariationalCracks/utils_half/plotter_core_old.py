"""DCEPlotterV4 implementation."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .plot_opts import (
    StressPlotOptsV4, ensure_dir, save_fig, apply_clip_percentiles,
    robust_vmin_vmax, symlog_levels, rot_from_tangent, network_extent
)
from .plot_smooth import moving_average_nan, resolve_smooth_window
from .plot_reconstruct import reconstruct_cod_csd_parametrized_smoothed
from .processor_PK import PKProcessor


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
        """Plot deformed crack network using **jump/COD** kinematics (J), not DeltaJ ("B").

        Rules enforced:
          - Degree-1 vertices (free tips): plotted tip is closed (J=0 at the tip point).
          - Degree>=2 vertices (junctions): all incident branches share a common jump vector J(v).
            Faces are attached to the two endpoints of the junction jump segment using a **cyclic CCW** rule:
              *the face on the CCW-normal side of the outgoing tangent connects to the START point*.
        """

        pk = PKProcessor(self.res)

        ne_half_eff = int(getattr(self.res, "sol", {}).get("ne_half", 0) or 0)
        if ne_half_eff <= 0:
            ne_half_eff = 10

        cod_win = resolve_smooth_window(cod_smooth_window, ne_half_eff)
        csd_win = resolve_smooth_window(csd_smooth_window, ne_half_eff)

        V = self.calc.network.vertices
        E = self.calc.network.edges

        deg = pk.degree_map()
        # Only parametrized option has a reliable polyline->vertex mapping for J(v).
        Jv_map = pk.junction_J_map() if getattr(self.res, "is_parametrized", lambda: False)() else {}

        sxy = 1e3 if units.lower() == "mm" else 1.0

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Deformed Crack Network (scale={scale:.0e})")

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C3", "C4", "C5"]

        def _nearest_end_is_start(xy: np.ndarray, p: np.ndarray) -> bool:
            xy = np.asarray(xy, float)
            if xy.ndim != 2 or xy.shape[0] < 2:
                return True
            d0 = float(np.hypot(*(xy[0, :] - p)))
            d1 = float(np.hypot(*(xy[-1, :] - p)))
            return d0 <= d1

        def _attach_tip_point(U: np.ndarray, L: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Attach p to the nearest end of each face polyline
            if _nearest_end_is_start(U, p):
                U = np.vstack([p.reshape(1, 2), U])
            else:
                U = np.vstack([U, p.reshape(1, 2)])
            if _nearest_end_is_start(L, p):
                L = np.vstack([p.reshape(1, 2), L])
            else:
                L = np.vstack([L, p.reshape(1, 2)])
            return U, L

        def _ensure_U_is_left_face(U: np.ndarray, L: np.ndarray, p: np.ndarray, t_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
            """Ensure U is the CCW-left face at the junction end.

            Returns (U,L,U_start) where U_start indicates whether the junction end is the START of the arrays.
            """
            U = np.asarray(U, float)
            L = np.asarray(L, float)
            p = np.asarray(p, float).reshape(2,)
            t_out = np.asarray(t_out, float).reshape(2,)
            nt = float(np.hypot(t_out[0], t_out[1]))
            if nt <= 0:
                return U, L, True
            t_out = t_out / nt
            n_left = np.array([-t_out[1], t_out[0]], float)

            U_start = _nearest_end_is_start(U, p)
            L_start = _nearest_end_is_start(L, p)
            if U_start != L_start:
                U_start = True

            d_sep = (U[0, :] - L[0, :]) if U_start else (U[-1, :] - L[-1, :])
            if float(np.dot(d_sep, n_left)) < 0.0:
                U, L = L, U
            return U, L, U_start

        def _attach_junction_endpoints_cyclic(
            U: np.ndarray,
            L: np.ndarray,
            p: np.ndarray,
            t_out: np.ndarray,
            Jv: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Attach junction endpoints using the cyclic CCW rule.

            Define junction jump segment endpoints:
              p_start = p + 0.5*Jv*scale
              p_end   = p - 0.5*Jv*scale

            Rule:
              - the CCW-left face (relative to outgoing tangent) connects to p_start
              - the other face connects to p_end
            """
            U, L, U_start = _ensure_U_is_left_face(U, L, p, t_out)
            Jv = np.asarray(Jv, float).reshape(2,)
            p = np.asarray(p, float).reshape(2,)

            p_start = p + 0.5 * Jv * float(scale)
            p_end = p - 0.5 * Jv * float(scale)

            if U_start:
                U = np.vstack([p_start.reshape(1, 2), U])
                L = np.vstack([p_end.reshape(1, 2), L])
            else:
                U = np.vstack([U, p_start.reshape(1, 2)])
                L = np.vstack([L, p_end.reshape(1, 2)])
            return U, L

        for edge_idx, edge in enumerate(E):
            v0 = next(v for v in V if int(v.id) == int(edge.v0))
            v1 = next(v for v in V if int(v.id) == int(edge.v1))
            p0 = np.array([float(v0.x), float(v0.y)], float)
            p1 = np.array([float(v1.x), float(v1.y)], float)

            v0_id = int(edge.v0)
            v1_id = int(edge.v1)

            # Gauge choice for reconstruction (plotting only):
            # We want **closed tips** (J_tip -> 0) for any degree-1 endpoint.
            # It is safe to apply a tip-zero gauge for *any* edge that has at least one free tip,
            # because the junction endpoint is later overridden/attached using the common J(v).
            enforce_gauge = (deg.get(v0_id, 0) == 1) or (deg.get(v1_id, 0) == 1)

            P, J = pk.jump_along_edge(
                edge_index=int(edge_idx),
                n_theta=int(max(400, n_theta)),
                enforce_global_tip_zero=bool(enforce_gauge),
                cod_window_panels=int(cod_win),
                csd_window_panels=int(csd_win),
            )

            U = P + 0.5 * J * float(scale)
            Lw = P - 0.5 * J * float(scale)

            # Apply endpoint rules
            for vid, p, tag in ((v0_id, p0, "start"), (v1_id, p1, "end")):
                d = int(deg.get(int(vid), 0))
                if d == 1:

                    # Enforce a *closed* tip in the plotted geometry.
                    #
                    # Simply appending the vertex point can create an unphysical triangular "wedge"
                    # when the last reconstructed point has a small but nonzero jump (J_tip ~ 1e-6).
                    # Instead, we *overwrite* the nearest end-point on BOTH faces so that:
                    #   U_tip == L_tip == p_tip
                    # which guarantees COD/J -> 0 exactly at the drawn tip.
                    mid = 0.5 * (U + Lw)
                    tip_is_start = _nearest_end_is_start(mid, p)

                    # Force exact closure at the tip point.
                    if tip_is_start:
                        U[0, :] = p
                        Lw[0, :] = p
                    else:
                        U[-1, :] = p
                        Lw[-1, :] = p

                    # Optional (plotting-only) tip taper:
                    # Even with a correct solver, the last *interior* sample may have a small residual
                    # J_tip (~1e-6), which becomes visually prominent after scaling. To avoid the
                    # persistent "wedge" artifact at degree-1 tips, we smoothly ramp the opening
                    # from zero at the tip to the reconstructed value over a few samples.
                    npts = int(min(len(U), len(Lw)))
                    k_tip = int(min(25, max(4, round(0.01 * npts))))
                    if npts >= k_tip + 2:
                        if tip_is_start:
                            idxs = np.arange(0, k_tip)
                        else:
                            idxs = np.arange(npts - k_tip, npts)
                        # alpha=0 at the tip, alpha=1 at the k_tip-1 point
                        alpha = np.linspace(0.0, 1.0, len(idxs)).reshape(-1, 1)
                        m = mid[idxs, :]
                        U[idxs, :] = m + alpha * (U[idxs, :] - m)
                        Lw[idxs, :] = m + alpha * (Lw[idxs, :] - m)

                elif d >= 2:
                    Jv = Jv_map.get(int(vid), None)
                    if Jv is None:
                        # Fallback: use nearest end separation (already scaled); convert back to unscaled jump
                        Jv = (U[0, :] - Lw[0, :]) / float(scale) if _nearest_end_is_start(U, p) else (U[-1, :] - Lw[-1, :]) / float(scale)
                    # outgoing tangent from junction along this edge
                    t_out = (p1 - p0) if tag == "start" else (p0 - p1)
                    U, Lw = _attach_junction_endpoints_cyclic(U, Lw, p, t_out, Jv)

            ax.plot([p0[0] * sxy, p1[0] * sxy], [p0[1] * sxy, p1[1] * sxy], ls=":", lw=1.0, color="k", alpha=0.6)

            if show_faces:
                ax.plot(U[:, 0] * sxy, U[:, 1] * sxy, lw=1.5, color=colors[edge_idx % len(colors)])
                ax.plot(Lw[:, 0] * sxy, Lw[:, 1] * sxy, lw=1.5, color=colors[edge_idx % len(colors)])
            else:
                ax.plot(P[:, 0] * sxy, P[:, 1] * sxy, lw=2.0, color=colors[edge_idx % len(colors)])

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
