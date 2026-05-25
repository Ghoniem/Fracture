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


def robust_vmin_vmax(arr, pct_lo=2.0, pct_hi=98.0):
    """Robust min/max from finite data only — respects the actual data sign.

    Returns (vmin, vmax) clipped to symmetric range *only if* the data
    straddles zero. Otherwise returns the asymmetric robust range so that
    a strongly one-sided distribution (e.g. compressive Syy in a Brazilian
    disk) maps across the full colormap instead of collapsing into the
    centre.
    """
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(a, pct_lo))
    hi = float(np.percentile(a, pct_hi))
    if lo < 0.0 < hi:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def plot_field(ax, xs, ys, Z, *, title, cmap="jet"):
    """pcolormesh handles NaN cleanly (transparent); contourf does not."""
    Z_mpa = Z * 1e-6
    vmin, vmax = robust_vmin_vmax(Z_mpa)
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    else:
        norm = None  # plain linear vmin/vmax — pcolormesh accepts them directly

    pcm = ax.pcolormesh(xs * 1e3, ys * 1e3, Z_mpa,
                        shading="auto", cmap=cmap,
                        vmin=vmin if norm is None else None,
                        vmax=vmax if norm is None else None,
                        norm=norm)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}  [MPa]\nrange=({vmin:.1f}, {vmax:.1f})")
    return pcm


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
            pcm = plot_field(ax, xs, ys, arrs[comp], title=f"{comp} ({engine})")
            fig.colorbar(pcm, ax=ax, label="MPa")
            fig.tight_layout()
            out = in_dir / f"{comp}_{engine}_contour.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"wrote {out}")

    # If both engines available, write a 1x3 row of side-by-side (cpp | py | diff) per component.
    if have_cpp and have_py:
        print("\nside-by-side (cpp | py | cpp-py) panels:")
        for comp in ("Sxx", "Syy", "Sxy"):
            cpp = engine_to_arrays["cpp"][comp]
            pyv = engine_to_arrays["py"][comp]
            diff = cpp - pyv

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            pcm_c = plot_field(axes[0], xs, ys, cpp, title=f"{comp} cpp")
            fig.colorbar(pcm_c, ax=axes[0], label="MPa")
            pcm_p = plot_field(axes[1], xs, ys, pyv, title=f"{comp} python")
            fig.colorbar(pcm_p, ax=axes[1], label="MPa")

            valid = np.isfinite(diff)
            if valid.any():
                v = float(np.nanpercentile(np.abs(diff), 98.0))
                if v <= 0:
                    v = 1.0
                pcm_d = axes[2].pcolormesh(xs * 1e3, ys * 1e3, diff,
                                            shading="auto", cmap="RdBu_r",
                                            vmin=-v, vmax=v)
                axes[2].set_aspect("equal")
                axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("y [mm]")
                axes[2].set_title(f"{comp} cpp - py [Pa]\nmax|diff|={float(np.nanmax(np.abs(diff))):.2e}")
                fig.colorbar(pcm_d, ax=axes[2], label="Pa")

            fig.tight_layout()
            out = in_dir / f"{comp}_compare.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"  {out}")


if __name__ == "__main__":
    main()
