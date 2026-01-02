
"""
dce_plots_v0_banded.py
Plotting utilities for dce_calcs_v0*.

Adds publication-style stress contour plots:
- Discrete/banded filled contours for ALL components
- Contour lines overlaid with numeric labels (MPa)
- Controllable upper range via vmax_mpa or vmax_factor * (remote component)
- Optional symmetric scaling about zero

No analytical LEFM/Williams fields are used.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# -------------------------
# Styling defaults
# -------------------------
CRACK_REF_STYLE = dict(color="k", linestyle=":", linewidth=1.0, zorder=10)  # undeformed crack
CRACK_UP_STYLE  = dict(color="b", linestyle="-", linewidth=2.0, zorder=11)
CRACK_LO_STYLE  = dict(color="r", linestyle="--", linewidth=2.0, zorder=11)


def _sample_x_local(a: float, n: int = 800, eps_factor: float = 1e-6):
    eps = max(eps_factor * a, 1e-18)
    return np.linspace(-a + eps, a - eps, n)


def _robust_abs_vmax(Z: np.ndarray, clip_percentile: float = 99.5) -> float:
    Z = np.asarray(Z, float).ravel()
    Z = Z[np.isfinite(Z)]
    if Z.size == 0:
        return 1.0
    if clip_percentile is None:
        vmax = float(np.max(np.abs(Z)))
    else:
        vmax = float(np.percentile(np.abs(Z), float(clip_percentile)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


class DCEPlotterV0:
    def __init__(self, calc, results: dict):
        self.calc = calc
        self.results = results

    # -------------------------
    # Burgers density and COD/CSD
    # -------------------------
    def plot_burgers(
        self,
        n: int = 1200,
        eps_factor: float = 1e-6,
        yscale: str = "symlog",
        linthresh: float | None = None,
        clip_percentile: float | None = 99.0,
        show_g: bool = True,
    ):
        a = self.calc.crack.half_length
        x = _sample_x_local(a, n=n, eps_factor=eps_factor)

        bI = self.calc.evaluate_b_mode(x, self.results, mode="I")
        bII = self.calc.evaluate_b_mode(x, self.results, mode="II")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        def _style(ax, y):
            if clip_percentile is not None:
                ymax = float(np.percentile(np.abs(y[np.isfinite(y)]), float(clip_percentile))) if np.any(np.isfinite(y)) else 1.0
                if not np.isfinite(ymax) or ymax <= 0:
                    ymax = 1.0
                ax.set_ylim(-ymax, ymax)
            if yscale == "symlog":
                if linthresh is None:
                    lt = max(1e-30, 0.01 * (np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0))
                else:
                    lt = float(linthresh)
                ax.set_yscale("symlog", linthresh=lt)

        axs[0].plot(x, bI, lw=2)
        axs[0].set_title("Mode I Burgers density $b_I(x)$")
        axs[0].set_xlabel("local $x_1$ [m]")
        axs[0].set_ylabel("$b_I$ [m]")
        axs[0].grid(True, alpha=0.3)
        _style(axs[0], bI)

        axs[1].plot(x, bII, lw=2)
        axs[1].set_title("Mode II Burgers density $b_{II}(x)$")
        axs[1].set_xlabel("local $x_1$ [m]")
        axs[1].set_ylabel("$b_{II}$ [m]")
        axs[1].grid(True, alpha=0.3)
        _style(axs[1], bII)

        figs = [fig]

        if show_g:
            gI = bI * np.sqrt(np.maximum(a * a - x * x, 0.0))
            gII = bII * np.sqrt(np.maximum(a * a - x * x, 0.0))

            fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
            axs2[0].plot(x, gI, lw=2)
            axs2[0].set_title(r"Mode I: $g_I(x)=b_I(x)\sqrt{a^2-x^2}$")
            axs2[0].set_xlabel("local $x_1$ [m]")
            axs2[0].set_ylabel("$g_I$ [m$^2$]")
            axs2[0].grid(True, alpha=0.3)

            axs2[1].plot(x, gII, lw=2)
            axs2[1].set_title(r"Mode II: $g_{II}(x)=b_{II}(x)\sqrt{a^2-x^2}$")
            axs2[1].set_xlabel("local $x_1$ [m]")
            axs2[1].set_ylabel("$g_{II}$ [m$^2$]")
            axs2[1].grid(True, alpha=0.3)

            figs.append(fig2)

        return figs

    def plot_cod_csd(self, n: int = 1200):
        x, COD, CSD = self.calc.reconstruct_cod_csd(self.results, n=n)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(x, COD, lw=2)
        axs[0].set_title("COD (normal opening)")
        axs[0].set_xlabel("local $x_1$ [m]")
        axs[0].set_ylabel("COD [m]")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(x, CSD, lw=2)
        axs[1].set_title("CSD (tangential sliding)")
        axs[1].set_xlabel("local $x_1$ [m]")
        axs[1].set_ylabel("CSD [m]")
        axs[1].grid(True, alpha=0.3)

        return fig, axs

    def plot_crack_displacement_geometry(self, n: int = 1200, scale: float | None = None):
        a = self.calc.crack.half_length
        x, COD, CSD = self.calc.reconstruct_cod_csd(self.results, n=n)

        ut = 0.5 * CSD
        un = 0.5 * COD

        x1 = x
        x2 = np.zeros_like(x1)

        upper = np.vstack([x1 + ut, x2 + un]).T
        lower = np.vstack([x1 - ut, x2 - un]).T

        R = self.calc.R
        cx, cy = self.calc.crack.center
        upper_g = (upper @ R.T) + np.array([cx, cy])
        lower_g = (lower @ R.T) + np.array([cx, cy])

        base = np.vstack([x1, x2]).T
        base_g = (base @ R.T) + np.array([cx, cy])

        if scale is None:
            max_disp = max(np.max(np.abs(COD)), np.max(np.abs(CSD)), 1e-30)
            scale = 0.05 * (2 * a) / max_disp

        upper_g_disp = base_g + scale * (upper_g - base_g)
        lower_g_disp = base_g + scale * (lower_g - base_g)

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(base_g[:, 0] * 1e3, base_g[:, 1] * 1e3, label="Undeformed crack", **CRACK_REF_STYLE)
        ax.plot(upper_g_disp[:, 0] * 1e3, upper_g_disp[:, 1] * 1e3, label="Upper face", **CRACK_UP_STYLE)
        ax.plot(lower_g_disp[:, 0] * 1e3, lower_g_disp[:, 1] * 1e3, label="Lower face", **CRACK_LO_STYLE)
        ax.fill_between(
            upper_g_disp[:, 0] * 1e3, upper_g_disp[:, 1] * 1e3, lower_g_disp[:, 1] * 1e3, alpha=0.15
        )

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.grid(True, alpha=0.3)

        rep = self.results.get("representation", "")
        ax.set_title(f"Crack displacement | rep={rep} | scale={scale:.4g}×")
        ax.legend(loc="best")
        return fig, ax, upper_g_disp, lower_g_disp, base_g

    # -------------------------
    # Stress fields with banded contours + labels
    # -------------------------
    def plot_stress_field_components(
        self,
        extent_factor: float = 1.5,
        n_grid: int = 301,
        add_remote: bool = True,
        mask_crack: bool = True,
        r_min_factor: float = 0.03,
        output_prefix: str | None = None,
        coords: str = "global",
        overlay_deformed: bool = True,
        deform_scale: float | None = None,
        component_set: str = "global",
        cmap: str = "jet",
        n_bands: int = 13,
        contour_line_color: str = "k",
        contour_linewidth: float = 0.7,
        label_contours: bool = True,
        label_fmt: str = "%.0f",
        label_fontsize: int = 7,
        symmetric: bool = True,
        # Range control:
        vmax_mpa: float | None = None,
        vmax_factor: float = 10.0,
        vref_component: str = "syy",
        clip_percentile: float | None = 99.5,
        # Optional explicit axis limits in meters:
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ):
        """
        Stress contour plots for all requested components, with discrete color bands + labeled contour lines.

        Range control:
          - If vmax_mpa is provided: uses +/-vmax_mpa.
          - Else if add_remote=True: uses vmax_factor * |remote component| (component-specific).
          - Else: uses robust percentile-based scaling from the computed field.

        vref_component:
          - for global: "sxx","syy","sxy" (default "syy")
          - for local:  "s11","s22","s12"
        """
        a = self.calc.crack.half_length
        L = extent_factor * a

        # local rectangular grid
        x = np.linspace(-L, L, n_grid)
        y = np.linspace(-L, L, n_grid)
        Xl, Yl = np.meshgrid(x, y)

        # mask region near tips and along crack line
        if mask_crack:
            rmin = r_min_factor * a
            tip1 = np.sqrt((Xl + a) ** 2 + Yl**2)
            tip2 = np.sqrt((Xl - a) ** 2 + Yl**2)
            mask = (tip1 < rmin) | (tip2 < rmin) | ((np.abs(Yl) < 0.01 * a) & (np.abs(Xl) < a))
        else:
            mask = np.zeros_like(Xl, dtype=bool)

        # stresses local [Pa]
        s11, s22, s12 = self.calc.evaluate_stress_field_local(Xl, Yl, self.results, add_remote=add_remote)

        # rotate to global [Pa]
        R = self.calc.R
        sxx = R[0, 0] * R[0, 0] * s11 + 2 * R[0, 0] * R[0, 1] * s12 + R[0, 1] * R[0, 1] * s22
        syy = R[1, 0] * R[1, 0] * s11 + 2 * R[1, 0] * R[1, 1] * s12 + R[1, 1] * R[1, 1] * s22
        sxy = (
            R[0, 0] * R[1, 0] * s11
            + (R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]) * s12
            + R[0, 1] * R[1, 1] * s22
        )

        coords_l = coords.lower()
        if coords_l == "local":
            Xp, Yp = Xl, Yl
            xlabel, ylabel = "local $x_1$ [mm]", "local $x_2$ [mm]"
        elif coords_l == "global":
            cx, cy = self.calc.crack.center
            XYg = np.stack([Xl, Yl], axis=-1) @ R.T
            Xp = XYg[..., 0] + cx
            Yp = XYg[..., 1] + cy
            xlabel, ylabel = "X [mm]", "Y [mm]"
        else:
            raise ValueError("coords must be 'local' or 'global'.")

        def _ma(Z):
            return np.ma.array(Z, mask=mask)

        fields = []
        if component_set in ("both", "local"):
            fields += [(_ma(s11), r"$\sigma_{11}$ (local)", "s11"),
                       (_ma(s22), r"$\sigma_{22}$ (local)", "s22"),
                       (_ma(s12), r"$\sigma_{12}$ (local)", "s12")]
        if component_set in ("both", "global"):
            fields += [(_ma(sxx), r"$\sigma_{xx}$ (global)", "sxx"),
                       (_ma(syy), r"$\sigma_{yy}$ (global)", "syy"),
                       (_ma(sxy), r"$\sigma_{xy}$ (global)", "sxy")]

        # Overlay deformed geometry
        if overlay_deformed:
            _, _, upper_g, lower_g, _ = self.plot_crack_displacement_geometry(scale=deform_scale)
            plt.close()
            if coords_l == "local":
                RT = self.calc.RT
                c = np.array(self.calc.crack.center)
                upper_plot = (upper_g - c) @ RT.T
                lower_plot = (lower_g - c) @ RT.T
            else:
                upper_plot = upper_g
                lower_plot = lower_g
        else:
            upper_plot = lower_plot = None

        # Remote reference values [MPa] (best-effort)
        remote = getattr(self.calc, "applied_stress", None)
        remote_map_global = {}
        remote_map_local = {}
        if remote is not None:
            try:
                remote_map_global = {
                    "sxx": float(getattr(remote, "sigma_xx", 0.0)) / 1e6,
                    "syy": float(getattr(remote, "sigma_yy", 0.0)) / 1e6,
                    "sxy": float(getattr(remote, "sigma_xy", 0.0)) / 1e6,
                }
            except Exception:
                remote_map_global = {"sxx": 0.0, "syy": 0.0, "sxy": 0.0}

            sxx0 = remote_map_global["sxx"] * 1e6
            syy0 = remote_map_global["syy"] * 1e6
            sxy0 = remote_map_global["sxy"] * 1e6
            Sg = np.array([[sxx0, sxy0], [sxy0, syy0]], dtype=float)
            Sl = R.T @ Sg @ R
            remote_map_local = {"s11": Sl[0, 0] / 1e6, "s22": Sl[1, 1] / 1e6, "s12": Sl[0, 1] / 1e6}

        figs = []
        for fld_pa, title, fname in fields:
            Z = fld_pa / 1e6  # MPa
            Zf = np.asarray(Z.filled(np.nan))

            # determine vmax in MPa
            if vmax_mpa is not None:
                vmax = float(vmax_mpa)
            elif add_remote:
                if fname in ("sxx", "syy", "sxy"):
                    vref = abs(remote_map_global.get(vref_component, remote_map_global.get(fname, 0.0)))
                    if vref <= 0:
                        vref = abs(remote_map_global.get(fname, 0.0))
                else:
                    vref = abs(remote_map_local.get(vref_component, remote_map_local.get(fname, 0.0)))
                    if vref <= 0:
                        vref = abs(remote_map_local.get(fname, 0.0))
                if vref <= 0:
                    vmax = _robust_abs_vmax(Zf, clip_percentile=clip_percentile)
                else:
                    vmax = float(vmax_factor) * float(vref)
            else:
                vmax = _robust_abs_vmax(Zf, clip_percentile=clip_percentile)

            vmax = max(vmax, 1e-12)

            if symmetric:
                vmin = -vmax
                levels = np.linspace(vmin, vmax, int(n_bands))
            else:
                vmin = float(np.nanmin(Zf))
                levels = np.linspace(vmin, vmax, int(n_bands))

            norm = Normalize(vmin=float(levels[0]), vmax=float(levels[-1]))

            fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
            cf = ax.contourf(Xp * 1e3, Yp * 1e3, Z, levels=levels, cmap=cmap, norm=norm, extend="both")
            cl = ax.contour(Xp * 1e3, Yp * 1e3, Z, levels=levels, colors=contour_line_color,
                            linewidths=float(contour_linewidth), alpha=0.9)

            if label_contours:
                ax.clabel(cl, inline=True, fontsize=int(label_fontsize), fmt=label_fmt)

            fig.colorbar(cf, ax=ax, label="Stress [MPa]")

            # Undeformed crack reference (thin dotted)
            if coords_l == "local":
                ax.plot(np.array([-a, a]) * 1e3, np.zeros(2), **CRACK_REF_STYLE)
            else:
                p1 = np.array([-a, 0.0]) @ R.T + np.array(self.calc.crack.center)
                p2 = np.array([ a, 0.0]) @ R.T + np.array(self.calc.crack.center)
                ax.plot(np.array([p1[0], p2[0]]) * 1e3, np.array([p1[1], p2[1]]) * 1e3, **CRACK_REF_STYLE)

            if overlay_deformed:
                ax.plot(upper_plot[:, 0] * 1e3, upper_plot[:, 1] * 1e3, **CRACK_UP_STYLE)
                ax.plot(lower_plot[:, 0] * 1e3, lower_plot[:, 1] * 1e3, **CRACK_LO_STYLE)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.2)

            if xlim is not None:
                ax.set_xlim(xlim[0] * 1e3, xlim[1] * 1e3)
            if ylim is not None:
                ax.set_ylim(ylim[0] * 1e3, ylim[1] * 1e3)

            figs.append(fig)
            if output_prefix:
                fig.savefig(f"{output_prefix}_{coords_l}_{fname}.png", dpi=220, bbox_inches="tight")

        return figs
