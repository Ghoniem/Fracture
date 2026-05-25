"""Visualize the boundary mesh produced by build_boundary.

Reads boundary_{x1,y1,x2,y2}.npy from a bench output directory and writes
two diagnostic plots:
  1. mesh_polar.png  -- xy view with each segment drawn + endpoint markers
  2. mesh_density.png -- element angular size (deg) vs midpoint angle, plus
                         a polar bar chart of node density vs angle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=str)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    x1 = np.load(in_dir / "boundary_x1.npy")
    y1 = np.load(in_dir / "boundary_y1.npy")
    x2 = np.load(in_dir / "boundary_x2.npy")
    y2 = np.load(in_dir / "boundary_y2.npy")
    n_seg = len(x1)

    seg_length = np.hypot(x2 - x1, y2 - y1)
    xm = 0.5 * (x1 + x2); ym = 0.5 * (y1 + y2)
    theta_deg = np.degrees(np.arctan2(ym, xm))
    R = float(np.hypot(xm, ym).mean())
    angular_size_deg = np.degrees(seg_length / R)   # exact for tiny arcs; off by <1% for the largest

    # ── 1. xy view ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n_seg):
        ax.plot([x1[i] * 1e3, x2[i] * 1e3], [y1[i] * 1e3, y2[i] * 1e3],
                "-", color="C0", lw=0.7)
    ax.plot(xm * 1e3, ym * 1e3, ".", color="C3", ms=3, label="midpoints")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Boundary mesh: N = {n_seg} segments")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out1 = in_dir / "mesh_polar.png"
    fig.savefig(out1, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    # ── 2. element angular size vs theta ──────────────────────────────────
    order = np.argsort(theta_deg)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(theta_deg[order], angular_size_deg[order], "o-", ms=3, lw=1)
    ax.axvline( 90, color="r", linestyle="--", alpha=0.4, label="platen +y")
    ax.axvline(-90, color="b", linestyle="--", alpha=0.4, label="platen -y")
    ax.axvline( 75, color="r", linestyle=":",  alpha=0.3)
    ax.axvline(105, color="r", linestyle=":",  alpha=0.3)
    ax.axvline(-75, color="b", linestyle=":",  alpha=0.3)
    ax.axvline(-105,color="b", linestyle=":",  alpha=0.3, label="loaded arc edges")
    ax.set_xlabel("midpoint angle [deg]")
    ax.set_ylabel("segment angular extent [deg]")
    ax.set_title(f"Element angular size (min={angular_size_deg.min():.2f}°, "
                 f"max={angular_size_deg.max():.2f}°, "
                 f"ratio={angular_size_deg.max()/angular_size_deg.min():.1f}x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out2 = in_dir / "mesh_density.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
