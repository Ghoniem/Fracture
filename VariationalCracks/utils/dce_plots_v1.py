
"""
dce_plots_v0.py
Plotting utilities for dce_calcs_v0.

Additions (v0.1):
- Robust Burgers-density plotting near tips (symlog or percentile clipping) + optional plot of g(x)=b(x)*sqrt(a^2-x^2)
- Stress fields plotted either in crack-local or global coordinates
- Optionally overlay *deformed crack geometry* on stress contour plots

No analytical LEFM/Williams fields are used.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from utils.dce_results_v1 import DCEResultsV0


def _sample_x_local(a: float, n: int = 800, eps_factor: float = 1e-6):
    eps = max(eps_factor * a, 1e-18)
    return np.linspace(-a + eps, a - eps, n)


def _robust_limits(y: np.ndarray, hi: float = 99.0):
    """Percentile-based symmetric limits about 0."""
    y = np.asarray(y, float).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return -1.0, 1.0
    a = float(np.percentile(np.abs(y), hi))
    if not np.isfinite(a) or a <= 0:
        a = float(np.max(np.abs(y))) if float(np.max(np.abs(y))) > 0 else 1.0
    return -a, a


class DCEPlotterV0:
    def __init__(self, calc, results):
        self.calc = calc
        self.results = results
        # Backward-compatible alias
        self.R = results
        # Solver output dict
        self.sol = results.sol if hasattr(results, 'sol') else results

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
        """
        Plot b_I(x), b_II(x) robustly near the tips.

        - yscale="symlog" helps when b(x) grows strongly near tips.
        - clip_percentile sets y-limits using symmetric +/- percentile(abs(y)).
          Set to None to disable clipping.
        - show_g plots g(x)=b(x)*sqrt(a^2-x^2), which removes the tip-weight singularity
          for the cheb_* representations (useful diagnostic).
        """
        a = self.calc.crack.half_length
        x = _sample_x_local(a, n=n, eps_factor=eps_factor)

        bI = self._burgers(x, mode="I")
        bII = self._burgers(x, mode="II")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        def _style(ax, y):
            if clip_percentile is not None:
                ymin, ymax = _robust_limits(y, hi=float(clip_percentile))
                ax.set_ylim(ymin, ymax)
            if yscale == "symlog":
                if linthresh is None:
                    _, ymax = _robust_limits(y, hi=99.0)
                    lt = max(1e-30, 0.01 * abs(ymax))
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
            axs2[0].set_title("Mode I: $g_I(x)=b_I(x)\\sqrt{a^2-x^2}$")
            axs2[0].set_xlabel("local $x_1$ [m]")
            axs2[0].set_ylabel("$g_I$ [m$^2$]")
            axs2[0].grid(True, alpha=0.3)

            axs2[1].plot(x, gII, lw=2)
            axs2[1].set_title("Mode II: $g_{II}(x)=b_{II}(x)\\sqrt{a^2-x^2}$")
            axs2[1].set_xlabel("local $x_1$ [m]")
            axs2[1].set_ylabel("$g_{II}$ [m$^2$]")
            axs2[1].grid(True, alpha=0.3)

            figs.append(fig2)

        return figs

    def plot_cod_csd(self, n: int = 1200):
        x, COD, CSD = self.results.reconstruct_cod_csd(n_theta=n, enforce_tip_zero=True)
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
        """
        Deformed crack faces in global coordinates from reconstructed COD/CSD.
        Returns (fig, ax, upper_g_disp, lower_g_disp, base_g).
        """
        a = self.calc.crack.half_length
        x, COD, CSD = self.results.reconstruct_cod_csd(n_theta=n, enforce_tip_zero=True)

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
        ax.plot(base_g[:, 0] * 1e3, base_g[:, 1] * 1e3, "k--", lw=2, label="Undeformed crack")
        ax.plot(upper_g_disp[:, 0] * 1e3, upper_g_disp[:, 1] * 1e3, "b-", lw=2, label="Upper face")
        ax.plot(lower_g_disp[:, 0] * 1e3, lower_g_disp[:, 1] * 1e3, "r--", lw=2, label="Lower face")
        ax.fill_between(
            upper_g_disp[:, 0] * 1e3, upper_g_disp[:, 1] * 1e3, lower_g_disp[:, 1] * 1e3, alpha=0.15
        )

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.grid(True, alpha=0.3)

        rep = self.results.get("representation", "")
        ax.set_title(f"Crack displacement | rep={rep} | scale={scale:.3g}×")
        ax.legend(loc="best")
        return fig, ax, upper_g_disp, lower_g_disp, base_g

    # -------------------------
    # Stress fields
    # -------------------------
    def plot_stress_field_components(
        self,
        extent_factor: float = 1.5,
        n_grid: int = 301,
        add_remote: bool = True,
        mask_crack: bool = True,
        r_min_factor: float = 0.02,
        output_prefix: str | None = None,
        coords: str = "local",
        overlay_deformed: bool = True,
        deform_scale: float | None = None,
        component_set: str = "both",
    ):
        """
        Numerical stress field around the crack.

        coords:
          - "local": axes in local x1,x2 (rectangular grid)
          - "global": axes in global X,Y (rotated grid; contoured using Xg,Yg arrays)

        overlay_deformed:
          If True, overlays deformed crack faces on the contour plots.

        component_set:
          - "local": plot (s11,s22,s12)
          - "global": plot (sxx,syy,sxy)
          - "both": plot all six
        """
        a = self.calc.crack.half_length
        L = extent_factor * a

        # local rectangular grid
        x = np.linspace(-L, L, n_grid)
        y = np.linspace(-L, L, n_grid)
        Xl, Yl = np.meshgrid(x, y)

        # mask region near tips and along the crack line
        if mask_crack:
            rmin = r_min_factor * a
            tip1 = np.sqrt((Xl + a) ** 2 + Yl**2)
            tip2 = np.sqrt((Xl - a) ** 2 + Yl**2)
            mask = (tip1 < rmin) | (tip2 < rmin) | ((np.abs(Yl) < 0.01 * a) & (np.abs(Xl) < a))
        else:
            mask = np.zeros_like(Xl, dtype=bool)

        # stresses local
        s11, s22, s12 = self._stress_field_local(Xl, Yl, add_remote=add_remote)

        # rotate to global components
        R = self.calc.R
        sxx = R[0, 0] * R[0, 0] * s11 + 2 * R[0, 0] * R[0, 1] * s12 + R[0, 1] * R[0, 1] * s22
        syy = R[1, 0] * R[1, 0] * s11 + 2 * R[1, 0] * R[1, 1] * s12 + R[1, 1] * R[1, 1] * s22
        sxy = (
            R[0, 0] * R[1, 0] * s11
            + (R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]) * s12
            + R[0, 1] * R[1, 1] * s22
        )

        # plotting coordinates
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

        # overlay geometry
        if overlay_deformed:
            _, _, upper_g, lower_g, base_g = self.plot_crack_displacement_geometry(scale=deform_scale)
            plt.close()

            if coords_l == "local":
                RT = self.calc.RT
                c = np.array(self.calc.crack.center)
                upper_plot = (upper_g - c) @ RT.T
                lower_plot = (lower_g - c) @ RT.T
            else:
                upper_plot = upper_g
                lower_plot = lower_g

        figs = []
        for fld, title, fname in fields:
            fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.4))
            cs = ax.contourf(Xp * 1e3, Yp * 1e3, fld / 1e6, levels=30)
            fig.colorbar(cs, ax=ax, label="Stress [MPa]")

            # undeformed crack line
            if coords_l == "local":
                ax.plot(np.array([-a, a]) * 1e3, np.zeros(2), "k-", lw=2)
            else:
                p1 = np.array([-a, 0.0]) @ R.T + np.array(self.calc.crack.center)
                p2 = np.array([ a, 0.0]) @ R.T + np.array(self.calc.crack.center)
                ax.plot(np.array([p1[0], p2[0]]) * 1e3, np.array([p1[1], p2[1]]) * 1e3, "k-", lw=2)

            if overlay_deformed:
                ax.plot(upper_plot[:, 0] * 1e3, upper_plot[:, 1] * 1e3, "b-", lw=2)
                ax.plot(lower_plot[:, 0] * 1e3, lower_plot[:, 1] * 1e3, "r--", lw=2)
                ax.fill_between(upper_plot[:, 0] * 1e3, upper_plot[:, 1] * 1e3, lower_plot[:, 1] * 1e3, alpha=0.10)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.2)

            figs.append(fig)
            if output_prefix:
                fig.savefig(f"{output_prefix}_{coords_l}_{fname}.png", dpi=200, bbox_inches="tight")

        return figs