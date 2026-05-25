"""Plot stress contours from the disk_compression_2 benchmark output.

Reads {Sxx,Syy,Sxy}_cpp.npy (and optionally _py.npy) plus xs.npy/ys.npy from
the bench output directory and writes contour PNGs.

If both engines were run, also writes a difference-map PNG per component.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def robust_lim(arr, pct=97.0):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(a, [100 - pct, pct])
    m = max(abs(lo), abs(hi))
    return -m, m


def plot_contour(ax, xs, ys, Z, *, title, cmap="jet"):
    vmin, vmax = robust_lim(Z, pct=97.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if vmin < 0 < vmax else None
    cf = ax.contourf(xs * 1e3, ys * 1e3, Z * 1e-6,
                     levels=30, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3]
                                / "VCEM_4_0" / "output"
                                / "bench_disk_compression_2"))
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    xs = np.load(in_dir / "xs.npy")
    ys = np.load(in_dir / "ys.npy")

    have_cpp = all((in_dir / f"{c}_cpp.npy").exists() for c in ("Sxx", "Syy", "Sxy"))
    have_py  = all((in_dir / f"{c}_py.npy").exists()  for c in ("Sxx", "Syy", "Sxy"))

    if not have_cpp and not have_py:
        print(f"ERROR: no S*_cpp.npy or S*_py.npy found in {in_dir}")
        sys.exit(1)

    engine_to_arrays = {}
    if have_cpp:
        engine_to_arrays["cpp"] = {
            "Sxx": np.load(in_dir / "Sxx_cpp.npy"),
            "Syy": np.load(in_dir / "Syy_cpp.npy"),
            "Sxy": np.load(in_dir / "Sxy_cpp.npy"),
        }
    if have_py:
        engine_to_arrays["py"] = {
            "Sxx": np.load(in_dir / "Sxx_py.npy"),
            "Syy": np.load(in_dir / "Syy_py.npy"),
            "Sxy": np.load(in_dir / "Sxy_py.npy"),
        }

    for engine, arrs in engine_to_arrays.items():
        for comp in ("Sxx", "Syy", "Sxy"):
            fig, ax = plt.subplots(figsize=(6, 5))
            cf = plot_contour(ax, xs, ys, arrs[comp],
                              title=f"{comp} [MPa] -- {engine}")
            fig.colorbar(cf, ax=ax)
            fig.tight_layout()
            out = in_dir / f"{comp}_{engine}_contour.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"wrote {out}")

    if have_cpp and have_py:
        print("\nDifference maps (cpp - py):")
        for comp in ("Sxx", "Syy", "Sxy"):
            diff = engine_to_arrays["cpp"][comp] - engine_to_arrays["py"][comp]
            fig, ax = plt.subplots(figsize=(6, 5))
            valid = np.isfinite(diff)
            if not valid.any():
                plt.close(fig); continue
            v = float(np.nanmax(np.abs(diff)))
            cf = ax.contourf(xs * 1e3, ys * 1e3, diff,
                             levels=30, cmap="RdBu_r",
                             vmin=-v, vmax=v)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"{comp}: cpp - py [Pa]  (max|diff|={v:.2e})")
            fig.colorbar(cf, ax=ax)
            fig.tight_layout()
            out = in_dir / f"{comp}_diff.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"  {out}: max|diff|={v:.3e} Pa  rms={np.sqrt(np.nanmean(diff**2)):.3e} Pa")


if __name__ == "__main__":
    main()
