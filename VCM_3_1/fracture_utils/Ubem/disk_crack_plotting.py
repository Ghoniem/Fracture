
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fracture_utils.Uplotter.core import DCEPlotterV4, StressPlotOptsV4


# ============================================================
# Plot parameter container
# ============================================================
class PlotParams:
    def __init__(
        self,
        cmap="jet",
        n_bands=40,
        vmax_factor=2.0,
        dpi=300,
        match_limits_to_crack=True,
        robust=True,
        robust_pct=97.0,
        symmetric=True,
        n_levels=30,
        n_line_levels=20,
    ):
        self.cmap = cmap
        self.n_bands = n_bands
        self.vmax_factor = vmax_factor
        self.dpi = dpi
        self.match_limits_to_crack = match_limits_to_crack
        self.robust = robust
        self.robust_pct = robust_pct
        self.symmetric = symmetric
        self.n_levels = n_levels
        self.n_line_levels = n_line_levels


# ============================================================
# TOTAL FIELD PLOTTING ONLY  (BEM + Crack)
# ============================================================
def plot_total_field(bem_dir, res, out_dir, tag, params: PlotParams, show=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load BEM arrays (Pa)
    xs = np.load(Path(bem_dir) / "xs.npy")
    ys = np.load(Path(bem_dir) / "ys.npy")
    Sxx_bem = np.load(Path(bem_dir) / "Sxx.npy")
    Syy_bem = np.load(Path(bem_dir) / "Syy.npy")
    Sxy_bem = np.load(Path(bem_dir) / "Sxy.npy")

    # --- Crack stress on SAME grid (Pa)
    plotter = DCEPlotterV4(res, out_dir=out_dir)

    opts = StressPlotOptsV4(
        extent_factor=3,
        n_grid=len(xs),
        add_remote=False,
        mask_cracks=False,
        cmap=params.cmap,
        n_bands=params.n_bands,
        label_contours=False,
        vmax_factor=params.vmax_factor,
        dpi=params.dpi,
    )

    plotter.plot_stress_components_global(
        opts=opts,
        components=("sxx", "syy", "sxy"),
        grid=(xs, ys),
        save_arrays=True,
        arrays_prefix="crack_tmp",
        show=False,
        save=False,
    )

    Sxx_cr = np.load(out_dir / "crack_tmp_sxx.npy")
    Syy_cr = np.load(out_dir / "crack_tmp_syy.npy")
    Sxy_cr = np.load(out_dir / "crack_tmp_sxy.npy")

    Sxx_tot = Sxx_bem + Sxx_cr
    Syy_tot = Syy_bem + Syy_cr
    Sxy_tot = Sxy_bem + Sxy_cr

    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

    def _robust_vlim(Z_ref_pa):
        Z = np.asarray(Z_ref_pa, float)
        Z = Z[np.isfinite(Z)]
        if Z.size == 0:
            return None
        if params.robust:
            q = np.nanpercentile(np.abs(Z), float(params.robust_pct))
        else:
            q = np.nanmax(np.abs(Z))
        if not np.isfinite(q) or q <= 0:
            return None
        return float(q) * float(params.vmax_factor)

    def _plot(Z_tot_pa, Z_ref_pa, name):
        # Scaling:
        vlim = _robust_vlim(Z_ref_pa if params.match_limits_to_crack else Z_tot_pa)
        Zm = Z_tot_pa * 1e-6  # MPa

        if vlim is not None:
            vlim_mpa = vlim * 1e-6
            if params.symmetric:
                vmin, vmax = -vlim_mpa, vlim_mpa
            else:
                vmin, vmax = np.nanmin(Zm), np.nanmax(Zm)
            levels = np.linspace(vmin, vmax, int(params.n_levels))
            line_levels = np.linspace(vmin, vmax, int(params.n_line_levels))
        else:
            levels = int(params.n_levels)
            line_levels = int(params.n_line_levels)

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        cf = ax.contourf(
            Xg * 1e3,
            Yg * 1e3,
            Zm,
            levels=levels,
            cmap=params.cmap,
            extend="both",
        )
        ax.contour(
            Xg * 1e3,
            Yg * 1e3,
            Zm,
            levels=line_levels,
            colors="k",
            linewidths=0.5,
            alpha=0.6,
        )

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"TOTAL {name} = BEM + Crack ({tag})")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Stress [MPa]")
        fig.savefig(out_dir / f"total_{name}_{tag}.png", dpi=params.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    _plot(Sxx_tot, Sxx_cr, "sxx")
    _plot(Syy_tot, Syy_cr, "syy")
    _plot(Sxy_tot, Sxy_cr, "sxy")

    return Sxx_tot, Syy_tot, Sxy_tot


# ============================================================
# Hook factory (TOTAL ONLY)
# ============================================================
def make_standard_plot_hook(
    bem_dir,
    combined_out_dir,
    plot_params: PlotParams,
    show_initial=True,
    show_cycles=False,
    show_final=True,
    only_post_simplify: bool = True,
):
    combined_out_dir = Path(combined_out_dir)

    def _hook(tag, res, step_dir):

        # ---- Initial
        if tag == "initial":
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag="initial",
                params=plot_params,
                show=show_initial,
            )
            return

        # ---- Cycle
        if tag.startswith("cycle_"):
            if only_post_simplify and (not tag.endswith("post_simplify")):
                return
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag=tag,
                params=plot_params,
                show=show_cycles,
            )
            return

        # ---- Final
        if tag == "final":
            plot_total_field(
                bem_dir,
                res,
                out_dir=combined_out_dir,
                tag="final",
                params=plot_params,
                show=show_final,
            )
            return

    return _hook
