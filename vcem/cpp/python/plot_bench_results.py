"""Plot stress contours from the disk_compression_2 benchmark output.

Matches the Python plotter style used by
fracture_utils.Ubem.bem_stress_plotter.plot_stress_component_contour:
  - contourf with 30 explicit levels in [vmin, vmax]
  - contour-line overlay with 20 line levels
  - symmetric robust range (default), or asymmetric if symmetric=False
  - jet colormap, MPa units, mm axes

Reads {Sxx,Syy,Sxy}_cpp.npy (and optionally _py.npy) plus xs.npy / ys.npy
from the bench output directory. When both engines are present, also
writes a side-by-side (cpp | py | cpp-py) panel per component.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Defaults that mirror brazilian_disk_bem.py ContourOpts ─────────────────
N_LEVELS      = 30
N_LINE_LEVELS = 20
CMAP          = "jet"
ROBUST_PCT    = 97.0
SYMMETRIC     = True   # matches the Python plotter for Sxx/Syy/Sxy in disk runs
EXTEND        = "both"
VALUE_SCALE   = 1e-6   # Pa -> MPa
XY_SCALE      = 1e3    # m  -> mm


def robust_vmin_vmax(arr, pct=ROBUST_PCT, symmetric=SYMMETRIC):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(a, 100 - pct))
    hi = float(np.percentile(a, pct))
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def plot_component(ax, xs, ys, Z, *, title, symmetric=SYMMETRIC):
    Zp = Z * VALUE_SCALE
    vmin, vmax = robust_vmin_vmax(Zp, symmetric=symmetric)
    levels      = np.linspace(vmin, vmax, N_LEVELS)
    line_levels = np.linspace(vmin, vmax, N_LINE_LEVELS)
    XX, YY = np.meshgrid(xs * XY_SCALE, ys * XY_SCALE)

    cf = ax.contourf(XX, YY, Zp, levels=levels, cmap=CMAP, extend=EXTEND)
    ax.contour(XX, YY, Zp, levels=line_levels,
               colors="k", linewidths=0.4, alpha=0.4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}  [MPa]   range=({vmin:.1f}, {vmax:.1f})")
    ax.grid(True, alpha=0.25)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3]
                                / "vcem" / "output"
                                / "bench_disk_compression_2"))
    ap.add_argument("--asymmetric", action="store_true",
                    help="use asymmetric robust range (default: symmetric, matches Python)")
    args = ap.parse_args()
    in_dir   = Path(args.in_dir)
    sym_flag = not args.asymmetric

    xs = np.load(in_dir / "xs.npy")
    ys = np.load(in_dir / "ys.npy")

    have_cpp = all((in_dir / f"{c}_cpp.npy").exists() for c in ("Sxx", "Syy", "Sxy"))
    have_py  = all((in_dir / f"{c}_py.npy").exists()  for c in ("Sxx", "Syy", "Sxy"))
    if not have_cpp and not have_py:
        print(f"ERROR: no S*_cpp.npy or S*_py.npy in {in_dir}")
        sys.exit(1)

    engines = {}
    if have_cpp:
        engines["cpp"] = {c: np.load(in_dir / f"{c}_cpp.npy") for c in ("Sxx", "Syy", "Sxy")}
    if have_py:
        engines["py"]  = {c: np.load(in_dir / f"{c}_py.npy")  for c in ("Sxx", "Syy", "Sxy")}

    # ── Single-engine contours ───────────────────────────────────────────
    for engine, arrs in engines.items():
        for comp in ("Sxx", "Syy", "Sxy"):
            fig, ax = plt.subplots(figsize=(6.4, 5.6))
            cf = plot_component(ax, xs, ys, arrs[comp],
                                title=f"{comp} ({engine})", symmetric=sym_flag)
            fig.colorbar(cf, ax=ax, shrink=0.92, label="Stress [MPa]")
            fig.tight_layout()
            out = in_dir / f"{comp}_{engine}_contour.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

    # ── Side-by-side compare ───────────────────────────────────────────────
    if have_cpp and have_py:
        print("\nside-by-side (cpp | py | cpp - py):")
        for comp in ("Sxx", "Syy", "Sxy"):
            cpp = engines["cpp"][comp]
            pyv = engines["py"][comp]
            diff = cpp - pyv

            fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
            cf_c = plot_component(axes[0], xs, ys, cpp,
                                  title=f"{comp} cpp", symmetric=sym_flag)
            fig.colorbar(cf_c, ax=axes[0], shrink=0.92, label="MPa")
            cf_p = plot_component(axes[1], xs, ys, pyv,
                                  title=f"{comp} python", symmetric=sym_flag)
            fig.colorbar(cf_p, ax=axes[1], shrink=0.92, label="MPa")

            valid = np.isfinite(diff)
            if valid.any():
                v = float(np.nanpercentile(np.abs(diff), ROBUST_PCT))
                if v <= 0: v = 1.0
                XX, YY = np.meshgrid(xs * XY_SCALE, ys * XY_SCALE)
                cf_d = axes[2].contourf(XX, YY, diff,
                                         levels=np.linspace(-v, v, N_LEVELS),
                                         cmap="RdBu_r", extend="both")
                axes[2].set_aspect("equal", adjustable="box")
                axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("y [mm]")
                axes[2].set_title(f"{comp}: cpp - py [Pa]   "
                                  f"max|diff|={float(np.nanmax(np.abs(diff))):.2e}")
                axes[2].grid(True, alpha=0.25)
                fig.colorbar(cf_d, ax=axes[2], shrink=0.92, label="Pa")

            fig.tight_layout()
            out = in_dir / f"{comp}_compare.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  {out}")


if __name__ == "__main__":
    main()
