"""DCEPlotterV4 implementation."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

from .opts import (
    StressPlotOptsV4, ensure_dir, save_fig, apply_clip_percentiles,
    robust_vmin_vmax, symlog_levels, rot_from_tangent, network_extent
)
from .smooth import moving_average_nan, resolve_smooth_window
from .reconstruct import reconstruct_cod_csd_parametrized_smoothed
from ..Uprocessor.PK import PKProcessor  

class DCEPlotterV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_network_graph(
        self,
        units: str = "mm",
        annotate: bool = False,
        *,
        font_size: int = 16,
        label_offset_frac: float = 0.07,
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
        d_off = label_offset_frac * span

        bbox_kwargs = dict(
            boxstyle="round,pad=0.32",
            fc="white",
            ec="0.6",
            lw=0.6,
            alpha=0.92,
        )

        # Edges + perpendicular edge labels
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
                px, py = -dy / L, dx / L  # unit perpendicular
                ax.text(
                    xm + d_off * px,
                    ym + d_off * py,
                    f"e{int(e.id)}",
                    ha="center", va="center",
                    fontsize=font_size,
                    bbox=bbox_kwargs,
                )

        # Build vertex incidence so vertex labels can be offset
        # away from every incident edge.
        incidence: Dict[int, list] = {int(v.id): [] for v in V}
        for e in E:
            incidence[int(e.v0)].append(int(e.v1))
            incidence[int(e.v1)].append(int(e.v0))

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, v in enumerate(V):
            vx, vy = v.x * s, v.y * s
            ax.scatter([vx], [vy], s=70,
                       color=colors[i % len(colors)] if colors else None,
                       zorder=4)
            if annotate:
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
                    ox, oy = 1.0, 1.0
                else:
                    ox, oy = -ux / norm, -uy / norm
                ax.text(
                    vx + d_off * ox,
                    vy + d_off * oy,
                    f"v{int(v.id)}",
                    ha="center", va="center",
                    fontsize=font_size,
                    fontweight="bold",
                    bbox=bbox_kwargs,
                    zorder=5,
                )

        ax.set_xlabel(f"x [{units}]", fontsize=font_size)
        ax.set_ylabel(f"y [{units}]", fontsize=font_size)
        ax.tick_params(axis="both", which="major", labelsize=font_size)
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
        show: bool = True,
        save: bool = True,
    ):
        """Plot deformed crack network using *geometric* jump (COD/CSD) only.

        Policy (per your request)
        ------------------------
        - No plotter-level enforcement of tip closure (no endpoint clamping, no affine ramps).
        - Geometry is built strictly from reconstructed COD/CSD mapped to a global jump vector.
        - Junctions (degree>=2) are attached using the common junction jump J(v) when available.

        This function relies on :class:`processor_PK.PKProcessor` for a consistent mapping between
        local COD/CSD and global coordinates, particularly for parametrized cracks where the
        polyline segment direction may differ from the raw network edge ordering.
        """

        pk = PKProcessor(self.res)

        n_crack_elements_eff = int(getattr(self.res, "sol", {}).get("n_crack_elements", 0) or 0)
        if n_crack_elements_eff <= 0:
            n_crack_elements_eff = 10

        cod_win = resolve_smooth_window(cod_smooth_window, n_crack_elements_eff)
        csd_win = resolve_smooth_window(csd_smooth_window, n_crack_elements_eff)

        V = self.calc.network.vertices
        E = self.calc.network.edges

        deg = pk.degree_map()

        # Junction model affects only the kinematic anchoring used for visualization.
        # - strict: legacy common-junction visualization may be used
        # - core/soft: junctions are intentionally open; do NOT force common J(v) in plotter
        sol = getattr(self.res, "sol", {}) if isinstance(getattr(self.res, "sol", {}), dict) else {}
        constraints_meta = sol.get("constraints", {}) if isinstance(sol.get("constraints", {}), dict) else {}
        junction_model = str(constraints_meta.get("junction_model", "strict")).lower().strip()
        if junction_model not in ("strict", "core", "soft"):
            junction_model = "strict"

        Jv_map = {}
        if junction_model == "strict" and getattr(self.res, "is_parametrized", lambda: False)():
            Jv_map = pk.junction_J_map()

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

        def _maybe_swap_faces(U: np.ndarray, L: np.ndarray, n_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Swap U/L so the mean separation points along +n_hat."""
            U = np.asarray(U, float)
            L = np.asarray(L, float)
            if U.ndim != 2 or L.ndim != 2 or U.shape[1] != 2 or L.shape[1] != 2:
                return U, L
            m = int(min(U.shape[0], L.shape[0]))
            if m < 3:
                return U, L
            i0 = m // 5
            i1 = m - m // 5
            if i1 <= i0 + 1:
                i0, i1 = 0, m
            d = np.mean(U[i0:i1, :] - L[i0:i1, :], axis=0)
            if float(np.dot(d, n_hat)) < 0.0:
                return L, U
            return U, L

        def _attach_junction_endpoints(U: np.ndarray, L: np.ndarray, p: np.ndarray, Jv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Attach the junction jump segment endpoints to the face endpoints.

            Uses a stable geometric pairing evaluated at the junction end of the polylines.
            """
            U = np.asarray(U, float)
            L = np.asarray(L, float)
            p = np.asarray(p, float).reshape(2,)
            Jv = np.asarray(Jv, float).reshape(2,)

            p_plus = p + 0.5 * Jv * float(scale)
            p_minus = p - 0.5 * Jv * float(scale)

            mid = 0.5 * (U + L)
            end_is_start = _nearest_end_is_start(mid, p)
            U_end = U[0, :] if end_is_start else U[-1, :]
            L_end = L[0, :] if end_is_start else L[-1, :]

            c1 = float(np.hypot(*(U_end - p_plus))) + float(np.hypot(*(L_end - p_minus)))
            c2 = float(np.hypot(*(U_end - p_minus))) + float(np.hypot(*(L_end - p_plus)))
            if c1 <= c2:
                pU, pL = p_plus, p_minus
            else:
                pU, pL = p_minus, p_plus

            if end_is_start:
                U = np.vstack([pU.reshape(1, 2), U])
                L = np.vstack([pL.reshape(1, 2), L])
            else:
                U = np.vstack([U, pU.reshape(1, 2)])
                L = np.vstack([L, pL.reshape(1, 2)])
            return U, L


        # Deformed crack faces:
        # If using CORE/SOFT junction model with parametrized solution, plot by polyline so that:
        #   - degree-1 tips are closed (solver enforces J_tip ≈ 0)
        #   - junctions remain open (branch-specific J0 at polyline start)
        if junction_model in ("core", "soft") and isinstance(sol, dict) and "polyline_solutions" in sol and "parametrized_polylines" in sol:
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
            polys = sol.get("parametrized_polylines", [])
            solps = sol.get("polyline_solutions", [])
            vpos = {int(v.id): np.array([float(v.x), float(v.y)], float) for v in V}

            for pid, meta in enumerate(polys):
                if int(pid) >= len(solps):
                    continue
                s = solps[int(pid)]

                # Midpoints along this polyline (meters)
                xy_mid = np.asarray(s.get("xy_mid", s.get("xy_col", None)), float)
                if xy_mid.ndim != 2 or xy_mid.shape[1] != 2:
                    continue

                bI  = np.asarray(s.get("bI", []), float).reshape(-1)
                bII = np.asarray(s.get("bII", []), float).reshape(-1)
                ds  = np.asarray(s.get("ds", []), float).reshape(-1)
                t   = np.asarray(s.get("t_mid", s.get("t_col", None)), float)
                n   = np.asarray(s.get("n_mid", s.get("n_col", None)), float)

                if bI.size == 0 or ds.size != bI.size or bII.size != bI.size:
                    continue
                if t is None or n is None or t.shape[0] != bI.size or n.shape[0] != bI.size:
                    continue

                J0 = np.asarray(s.get("J0", [0.0, 0.0]), float).reshape(2,)
                beta = bII[:, None] * t + bI[:, None] * n
                Bcum = np.cumsum(beta * ds[:, None], axis=0)
                Jmid = J0[None, :] - Bcum
                J_end = (J0 - Bcum[-1]) if Bcum.size else J0

                vids = meta.get("path_vertex_ids", [])
                if not vids:
                    continue
                v_start = int(meta.get("v_start", vids[0]))
                v_end   = int(meta.get("v_end", vids[-1]))
                p_start = vpos.get(v_start, None)
                p_end   = vpos.get(v_end, None)
                if p_start is None or p_end is None:
                    continue

                Pline = np.vstack([p_start, xy_mid, p_end])
                Jline = np.vstack([J0[None, :], Jmid, J_end[None, :]])

                Pplot = Pline * sxy
                Jplot = Jline * float(scale) * sxy

                U = Pplot + 0.5 * Jplot
                Lw = Pplot - 0.5 * Jplot

                col = colors[int(pid) % len(colors)]
                ax.plot(Pplot[:, 0], Pplot[:, 1], ls=":", lw=1.0, color="k", alpha=0.35)
                if show_faces:
                    ax.plot(U[:, 0], U[:, 1], lw=2.0, color=col, alpha=0.95)
                    ax.plot(Lw[:, 0], Lw[:, 1], lw=2.0, color=col, alpha=0.95)

        else:
            # STRICT (legacy) or non-parametrized: plot per-edge using reconstruction
            for edge_idx, edge in enumerate(E):
                v0 = next(v for v in V if int(v.id) == int(edge.v0))
                v1 = next(v for v in V if int(v.id) == int(edge.v1))
                p0 = np.array([float(v0.x), float(v0.y)], float)
                p1 = np.array([float(v1.x), float(v1.y)], float)

                v0_id = int(edge.v0)
                v1_id = int(edge.v1)

                # Plot geometry from *geometric* jump only (COD/CSD -> J).
                # We do not clamp/taper in the plotter. However, the jump reconstruction
                # is defined up to an additive constant (integration constant). For any
                # edge that has a degree-1 endpoint, we fix that constant by requiring
                # J=0 at the physical tip point. This is a gauge choice for reconstruction,
                # not a solver modification.
                tip_point = None
                if int(deg.get(v0_id, 0)) == 1:
                    tip_point = p0
                elif int(deg.get(v1_id, 0)) == 1:
                    tip_point = p1

                # Tip-zero gauge for reconstruction: if this edge has a degree-1 endpoint
                # (a physical tip), request tip-zero from the reconstruction backend.
                P, J = pk.jump_along_edge(
                    edge_index=int(edge_idx),
                    n_theta=int(max(400, n_theta)),
                    enforce_global_tip_zero=False,
                    tip_point=tip_point,
                    cod_window_panels=int(cod_win),
                    csd_window_panels=int(csd_win),
                )

                U = P + 0.5 * J * float(scale)
                Lw = P - 0.5 * J * float(scale)

                # Visual consistency: choose a deterministic U/L ordering by the geometric normal.
                t = p1 - p0
                Lt = float(np.hypot(t[0], t[1]))
                if Lt > 0:
                    t_hat = t / Lt
                    n_hat = np.array([-t_hat[1], t_hat[0]], float)
                    U, Lw = _maybe_swap_faces(U, Lw, n_hat)

                # Junction attachments (degree>=2) using common J(v) where available.
                for vid, p in ((v0_id, p0), (v1_id, p1)):
                    if int(deg.get(int(vid), 0)) >= 2:
                        Jv = Jv_map.get(int(vid), None)
                        if Jv is not None:
                            U, Lw = _attach_junction_endpoints(U, Lw, p, Jv)

                if show_faces:
                    ax.plot(U[:, 0] * sxy, U[:, 1] * sxy, lw=1.5, color=colors[edge_idx % len(colors)])
                    ax.plot(Lw[:, 0] * sxy, Lw[:, 1] * sxy, lw=1.5, color=colors[edge_idx % len(colors)])
                else:
                    ax.plot(P[:, 0] * sxy, P[:, 1] * sxy, lw=2.0, color=colors[edge_idx % len(colors)])

                ax.plot([p0[0] * sxy, p1[0] * sxy], [p0[1] * sxy, p1[1] * sxy], ls=":", lw=1.0, color="k", alpha=0.6)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        if save:
            try:
                save_fig(fig, self.out_dir, "network_deformed", dpi=150)
            except Exception:
                pass
        if show:
            plt.show()
        return fig

    def plot_displacement_vector_field(
        self,
        extent_factor: float = 1.2,
        n_grid: int = 41,
        scale: float = 1.0,
        disp_scale: float = 1e6,
        units: str = "mm",
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
        *,
        grid: Optional[tuple[np.ndarray, np.ndarray]] = None,
        save_arrays: bool = False,
        return_arrays: bool = False,
        arrays_prefix: str = "stress",
        show: bool = True,
        save: bool = True,
    ) -> Dict[str, plt.Figure] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, plt.Figure]]:
        """Plot global stress components on a regular grid.

        Additions (for Coupled VCM–BEM workflows)
        ----------------------------------------
        - grid=(xs, ys): override plotting grid using 1D coordinate arrays (meters)
        - save_arrays: save xs/ys and component arrays (Pa) to out_dir as:
            {arrays_prefix}_xs.npy, {arrays_prefix}_ys.npy,
            {arrays_prefix}_sxx.npy, {arrays_prefix}_syy.npy, {arrays_prefix}_sxy.npy
        - return_arrays: return (xs, ys, Sxx, Syy, Sxy, figs) where stresses are in Pa
        """
        opts = StressPlotOptsV4(**vars(opts))

        V = self.calc.network.vertices
        ext = network_extent(V)

        if grid is None:
            half_w = 0.5 * float(opts.extent_factor) * ext
            xs = np.linspace(-half_w, half_w, int(opts.n_grid))
            ys = np.linspace(-half_w, half_w, int(opts.n_grid))
        else:
            xs = np.asarray(grid[0], float).reshape(-1)
            ys = np.asarray(grid[1], float).reshape(-1)

        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

        # Stress field in Pa from results object
        sxx, syy, sxy = self.res.stress_field_global(Xg, Yg, add_remote=bool(opts.add_remote))
        Sxx_pa = np.asarray(sxx, float)
        Syy_pa = np.asarray(syy, float)
        Sxy_pa = np.asarray(sxy, float)

        # For plotting, convert to MPa
        fields = {"sxx": Sxx_pa / 1e6, "syy": Syy_pa / 1e6, "sxy": Sxy_pa / 1e6}

        # Reference magnitudes for robust color limits
        ref_map = {"sxx": 0.0, "syy": 0.0, "sxy": 0.0}
        try:
            ap = self.calc.applied
            is_spatial = bool(getattr(ap, "sigma_func", None))
            if not is_spatial:
                remote = np.array([[ap.sigma_xx, ap.sigma_xy], [ap.sigma_xy, ap.sigma_yy]], float) / 1e6
                ref_map = {"sxx": float(remote[0, 0]), "syy": float(remote[1, 1]), "sxy": float(remote[0, 1])}
        except Exception:
            pass

        figs: Dict[str, plt.Figure] = {}
        for comp in components:
            comp = str(comp).lower()
            if comp not in fields:
                continue
            Z = np.array(fields[comp], float)

            Zp = apply_clip_percentiles(Z, getattr(opts, "clip_percentiles", None))
            ref = abs(float(ref_map.get(comp, 0.0)))
            vmin, vmax = robust_vmin_vmax(Zp, opts, ref=ref)

            norm = None
            norm_name = str(getattr(opts, "norm", "linear") or "linear").lower()
            if norm_name == "symlog":
                from matplotlib.colors import SymLogNorm
                linth = float(getattr(opts, "symlog_linthresh", 1.0))
                norm = SymLogNorm(
                    linthresh=linth,
                    linscale=float(getattr(opts, "symlog_linscale", 1.0)),
                    vmin=float(vmin),
                    vmax=float(vmax),
                    base=float(getattr(opts, "symlog_base", 10.0)),
                )
                levels = symlog_levels(vmin, vmax, linth, int(opts.n_bands) + 1)
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

        if save_arrays:
            np.save(self.out_dir / f"{arrays_prefix}_xs.npy", xs)
            np.save(self.out_dir / f"{arrays_prefix}_ys.npy", ys)
            np.save(self.out_dir / f"{arrays_prefix}_sxx.npy", Sxx_pa)
            np.save(self.out_dir / f"{arrays_prefix}_syy.npy", Syy_pa)
            np.save(self.out_dir / f"{arrays_prefix}_sxy.npy", Sxy_pa)

        if return_arrays:
            return xs, ys, Sxx_pa, Syy_pa, Sxy_pa, figs

        return figs
