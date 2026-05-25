"""Symmetry diagnostic for the disk_compression_2 stress field.

For a Brazilian disk with the platens centered on +/- y and zero shear at
the equator, every stress component must obey three reflection symmetries:

  Sxx(-x, y)  == Sxx(x, y)
  Sxx( x,-y)  == Sxx(x, y)
  Syy(-x, y)  == Syy(x, y)
  Syy( x,-y)  == Syy(x, y)
  Sxy(-x, y)  == -Sxy(x, y)   (antisymmetric)
  Sxy( x,-y)  == -Sxy(x, y)   (antisymmetric)

Anything beyond panel-discretization error suggests a real bug. Reports the
asymmetry as both absolute (Pa) and as a fraction of the field's robust
range; also compares Python (engine='py') vs C++ (engine='cpp') on the
same discretization so we can distinguish (a) BC/mesh-induced artifacts
common to both implementations from (b) a C++-only problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load(engine: str, in_dir: Path):
    return {comp: np.load(in_dir / f"{comp}_{engine}.npy")
            for comp in ("Sxx", "Syy", "Sxy")}


def symmetry_metrics(name: str, Z: np.ndarray, *, antisymmetric: bool = False):
    """Report (y-axis flip), (x-axis flip), and (point reflection) deltas."""
    Zh = Z[:, ::-1]            # x -> -x
    Zv = Z[::-1, :]            # y -> -y
    Zhv = Z[::-1, ::-1]        # (x,y) -> (-x,-y)

    sign = -1.0 if antisymmetric else 1.0
    d_y = Z - sign * Zh
    d_x = Z - sign * Zv
    d_pt = Z - Zhv             # point reflection: both axes flipped, sign**2 = 1

    valid = np.isfinite(Z) & np.isfinite(Zh) & np.isfinite(Zv) & np.isfinite(Zhv)
    scale = float(np.nanmax(np.abs(Z))) if np.any(valid) else 1.0

    def stat(d):
        v = d[valid]
        if v.size == 0:
            return 0.0, 0.0
        return float(np.max(np.abs(v))), float(np.sqrt(np.mean(v ** 2)))

    yax_max, yax_rms = stat(d_y)
    xax_max, xax_rms = stat(d_x)
    pt_max,  pt_rms  = stat(d_pt)

    print(f"  {name}  scale={scale:.3e}  (antisymmetric={antisymmetric})")
    print(f"    y-axis flip  max={yax_max:.3e}  rms={yax_rms:.3e}  "
          f"(rel max {yax_max/scale:.3e})")
    print(f"    x-axis flip  max={xax_max:.3e}  rms={xax_rms:.3e}  "
          f"(rel max {xax_max/scale:.3e})")
    print(f"    pt reflect   max={pt_max:.3e}  rms={pt_rms:.3e}  "
          f"(rel max {pt_max/scale:.3e})")
    return {
        "scale": scale,
        "y_axis_max_rel": yax_max / scale,
        "x_axis_max_rel": xax_max / scale,
        "pt_max_rel":     pt_max  / scale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="Directory containing Sxx_{cpp,py}.npy etc.")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    have_cpp = all((in_dir / f"{c}_cpp.npy").exists() for c in ("Sxx", "Syy", "Sxy"))
    have_py  = all((in_dir / f"{c}_py.npy").exists()  for c in ("Sxx", "Syy", "Sxy"))

    if have_cpp:
        print(f"\n=== C++ engine ({in_dir}) ===")
        arrs = load("cpp", in_dir)
        for comp, anti in [("Sxx", False), ("Syy", False), ("Sxy", True)]:
            symmetry_metrics(comp, arrs[comp], antisymmetric=anti)

    if have_py:
        print(f"\n=== Python engine ({in_dir}) ===")
        arrs = load("py", in_dir)
        for comp, anti in [("Sxx", False), ("Syy", False), ("Sxy", True)]:
            symmetry_metrics(comp, arrs[comp], antisymmetric=anti)

    if have_cpp and have_py:
        print("\n=== INTERPRETATION ===")
        print("  - If cpp and python show similar asymmetry magnitudes, it's a")
        print("    discretization/BC artifact common to both implementations.")
        print("  - If cpp is much worse than python, there's a real bug in C++.")


if __name__ == "__main__":
    main()
