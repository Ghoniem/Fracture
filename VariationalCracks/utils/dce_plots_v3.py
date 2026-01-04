"""
utils.dce_plots_v3

Plotting for multi-crack results.
- Stress components in GLOBAL coordinates
- Overlay all cracks (undeformed dotted line) + optional deformed outlines (per crack)
- Save figures to output directory
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class StressPlotOpts:
    def __init__(
        self,
        extent_factor=2.0,
        n_grid=401,
        add_remote=True,
        mask_crack=True,
        cmap="jet",
        n_bands=12,
        label_contours=False,
        label_fmt="%.0f",
        vmax_factor=10.0,
    ):
        self.extent_factor = float(extent_factor)
        self.n_grid = int(n_grid)
        self.add_remote = bool(add_remote)
        self.mask_crack = bool(mask_crack)
        self.cmap = str(cmap)
        self.n_bands = int(n_bands)
        self.label_contours = bool(label_contours)
        self.label_fmt = str(label_fmt)
        self.vmax_factor = float(vmax_factor)


def _applied_tensor(applied) -> np.ndarray:
    """
    Robustly obtain 2x2 applied stress tensor in GLOBAL coords [Pa].
    Supports:
      - AppliedStress with .tensor()
      - AppliedStress with attributes sigma_xx/sigma_yy/sigma_xy
      - direct 2x2 ndarray
    """
    if hasattr(applied, "tensor") and callable(getattr(applied, "tensor")):
        T = np.asarray(applied.tensor(), float)
        if T.shape != (2, 2):
            raise ValueError("applied.tensor() must return shape (2,2)")
        return T
    if all(hasattr(applied, k) for k in ("sigma_xx", "sigma_yy", "sigma_xy")):
        return np.array([[float(applied.sigma_xx), float(applied.sigma_xy)],
                         [float(applied.sigma_xy), float(applied.sigma_yy)]], dtype=float)
    T = np.asarray(applied, float)
    if T.shape == (2, 2):
        return T
    raise TypeError("Unsupported applied stress object; expected AppliedStress-like or 2x2 array.")


class DCEPlotterV3:
    def __init__(self, results_v3, out_dir=None):
        self.res = results_v3
        self.calc = results_v3.calc
        self.sol = results_v3.sol

        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig, name):
        if self.out_dir is not None:
            fig.savefig(self.out_dir / name, dpi=300, bbox_inches="tight")

    def plot_stress_components_global(self, opts: StressPlotOpts, components=("sxx", "syy", "sxy")):
        # Build domain based on max crack half-length
        amax = max([ck.half_length for ck in self.calc.cracks])
        L = opts.extent_factor * amax

        x = np.linspace(-L, L, opts.n_grid)
        y = np.linspace(-L, L, opts.n_grid)
        Xg, Yg = np.meshgrid(x, y)

        # Compute stress field [Pa] and convert to MPa
        sxx, syy, sxy = self.res.stress_field_global(Xg, Yg, add_remote=opts.add_remote)
        fields = {"sxx": sxx/1e6, "syy": syy/1e6, "sxy": sxy/1e6}

        # Remote tensor (for scale reference)
        sig = _applied_tensor(self.calc.applied) / 1e6  # MPa

        figs = {}

        for comp in components:
            Z = np.asarray(fields[comp], float).copy()

            # mask points on each crack line segment to avoid singular pixels
            if opts.mask_crack:
                for ck in self.calc.cracks:
                    center = np.asarray(ck.center, float)
                    R = np.array([[np.cos(ck.angle), -np.sin(ck.angle)],
                                  [np.sin(ck.angle),  np.cos(ck.angle)]], float)
                    Rt = R.T
                    Pl = Rt @ (np.vstack([Xg.ravel(), Yg.ravel()]) - center.reshape(2,1))
                    xl = Pl[0].reshape(Xg.shape)
                    yl = Pl[1].reshape(Xg.shape)
                    mask = (np.abs(yl) < 1e-12) & (np.abs(xl) <= ck.half_length)
                    Z[mask] = np.nan

            # color scale cap: reference remote component if available; else robust max
            ref = 0.0
            if comp == "sxx":
                ref = abs(sig[0,0])
            elif comp == "syy":
                ref = abs(sig[1,1])
            elif comp == "sxy":
                ref = abs(sig[0,1])

            base = ref if ref > 0 else float(np.nanmax(np.abs(Z)))
            vmax = max(1e-12, opts.vmax_factor * base)

            levels = np.linspace(-vmax, vmax, opts.n_bands + 1)

            fig, ax = plt.subplots(figsize=(6,5))
            csf = ax.contourf(Xg*1e3, Yg*1e3, Z, levels=levels, cmap=opts.cmap, extend="both")
            plt.colorbar(csf, ax=ax, label="Stress [MPa]")

            # contours
            cs = ax.contour(Xg*1e3, Yg*1e3, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.6)
            if opts.label_contours:
                ax.clabel(cs, inline=True, fontsize=8, fmt=opts.label_fmt)

            # overlay undeformed crack lines (dotted thin)
            for ck in self.calc.cracks:
                a = ck.half_length
                center = np.asarray(ck.center, float).reshape(2,)
                R = np.array([[np.cos(ck.angle), -np.sin(ck.angle)],
                              [np.sin(ck.angle),  np.cos(ck.angle)]], float)
                pL = center + (R @ np.array([-a, 0.0]))
                pR = center + (R @ np.array([+a, 0.0]))
                ax.plot([pL[0]*1e3, pR[0]*1e3], [pL[1]*1e3, pR[1]*1e3], ls=":", lw=1.0, color="k")

            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title(comp)

            self._save_fig(fig, f"stress_{comp}.png")
            figs[comp] = fig

        return figs
