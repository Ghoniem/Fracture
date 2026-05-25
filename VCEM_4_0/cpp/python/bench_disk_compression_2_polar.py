"""Brazilian-disk benchmark on a disk-conforming polar evaluation grid.

Same problem as bench_disk_compression_2.py, but the interior stress
evaluation uses a polar (r, theta) grid that fits the disk exactly and
extends right up to the boundary -- no rectangular raster, no
inside-disk mask staircase. Plot via matplotlib tricontourf on the
scattered (x, y) points; no clip path needed because the outermost ring
of points already traces the disk outline.

Usage:
    python bench_disk_compression_2_polar.py \
        [--n-elem 500] [--n-r 60] [--n-theta 240] \
        [--graded] [--concentration 8] [--taper-exponent 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))
sys.path.insert(0, str(REPO / "VCEM_4_0" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Ubem.boundary_conditions import (
    build_boundary, BCSpec, assemble_segment_bcs,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# Problem definition mirrors bench_disk_compression_2.py.
R           = 25.4e-3 / 2.0
P_TOTAL     = 3.8e3
E           = 231.52e9
NU          = 0.3
PLANE_STRAIN = True
H_THICK     = 6.35e-3
GAUSS_N_SOLVE  = 4
GAUSS_N_STRESS = 24      # bumped from 12: with a polar grid hugging the
                         # boundary, the interior-point quadrature has to
                         # resolve a near-singular kernel for points
                         # within ~one-element-length of a panel.
ARC_HALF_ANGLE = 15.0
BOUNDARY_PAD   = 5.0e-3   # fraction of R kept clear of the boundary.
                          # The interior-point Somigliana formula loses accuracy
                          # when field points are closer to a boundary element
                          # than the element's own length (near-singular kernel,
                          # Gauss-Legendre under-resolves). For N=500 graded with
                          # 0.33 deg platen elements (~0.073 mm), a pad of
                          # 0.005*R ~ 0.063 mm sits ~1 element-length inside;
                          # for N=120 graded with 1.36 deg elements (~0.30 mm),
                          # the same fraction sits well inside the safe zone.


def build_problem(n_elem, graded, concentration, taper_exponent):
    if graded:
        mesh = build_boundary({
            "type": "circle_graded",
            "R": R, "n_boundary": n_elem, "center": (0.0, 0.0),
            "focal_angles_deg": (90.0, -90.0),
            "concentration": concentration,
            "taper_exponent": taper_exponent,
        })
    else:
        mesh = build_boundary({
            "type": "circle", "R": R, "n_boundary": n_elem, "center": (0.0, 0.0),
        })

    theta_deg = np.asarray(mesh.theta_deg, dtype=float)
    L         = np.asarray(mesh.length,    dtype=float)
    arc = ARC_HALF_ANGLE
    eps = 1.0e-9
    top = (theta_deg >=  90 - arc - eps) & (theta_deg <=  90 + arc + eps)
    bot = (theta_deg >= -90 - arc - eps) & (theta_deg <= -90 + arc + eps)
    L_top = float(np.sum(L[top])); L_bot = float(np.sum(L[bot]))
    pressure_top = P_TOTAL / (L_top * H_THICK)
    pressure_bot = P_TOTAL / (L_bot * H_THICK)

    bc_specs = [
        BCSpec("pressure_normal", pressure_top, "theta_deg_range", ( 90 - arc,  90 + arc)),
        BCSpec("pressure_normal", pressure_bot, "theta_deg_range", (-90 - arc, -90 + arc)),
    ]
    is_traction, bc_x, bc_y = assemble_segment_bcs(
        mesh, bc_specs=bc_specs, default=("traction", 0.0, 0.0))
    return mesh, is_traction, bc_x, bc_y


def make_solver(mesh, is_traction, bc_x, bc_y):
    s = bem_cpp.BEMSolver2D(E=E, nu=NU, h=H_THICK, plane_strain=PLANE_STRAIN)
    for k in range(mesh.n_seg):
        s.add_element(float(mesh.x1[k]), float(mesh.y1[k]),
                      float(mesh.x2[k]), float(mesh.y2[k]),
                      bool(is_traction[k]),
                      float(bc_x[k]), float(bc_y[k]))
    return s


def polar_disk_grid(n_r: int, n_theta: int):
    """(xs, ys) flat arrays for a polar grid covering the disk.

    Outermost ring at r = R*(1 - BOUNDARY_PAD); innermost ring at the same
    spacing as adjacent rings (so r=0 is NOT included as a duplicated
    n_theta-point cluster). One extra point at the centre is appended so
    Delaunay triangulation in tricontourf fills the interior smoothly.

    Angular distribution is also graded -- denser near +/-90 deg -- using
    the same cos^p density as the boundary mesh. This keeps the polar
    evaluation grid as well-resolved at the platens as the boundary mesh.
    """
    r_max = R * (1.0 - BOUNDARY_PAD)
    dr = r_max / n_r          # spacing so first ring sits at dr
    r = np.linspace(dr, r_max, n_r)

    # Graded angular sampling: cos^4 bump at +/- 90 deg (same shape as the
    # boundary mesh density) so plotting fidelity matches the boundary
    # resolution under the loaded arcs.
    th_fine = np.linspace(0.0, 2*np.pi, max(20000, 200*n_theta) + 1)
    density = 1.0 + 7.0 * (np.maximum(0, np.cos(th_fine - np.pi/2))**4
                          + np.maximum(0, np.cos(th_fine + np.pi/2))**4)
    cdf = np.concatenate([[0.0], np.cumsum(0.5*(density[:-1]+density[1:])*np.diff(th_fine))])
    cdf /= cdf[-1]
    target = np.linspace(0.0, 1.0, n_theta, endpoint=False)
    th = np.interp(target, cdf, th_fine)

    R_grid, T_grid = np.meshgrid(r, th, indexing="ij")
    xs = (R_grid * np.cos(T_grid)).ravel()
    ys = (R_grid * np.sin(T_grid)).ravel()

    # Plus a single centre point so the triangulation has no central hole.
    xs = np.concatenate([[0.0], xs])
    ys = np.concatenate([[0.0], ys])
    return xs, ys


# ── Plot helpers (tricontourf on scattered polar points) ─────────────────
def robust_vmin_vmax(arr, pct=97.0, symmetric=True):
    a = arr[np.isfinite(arr)]
    if a.size == 0: return -1.0, 1.0
    lo, hi = np.percentile(a, [100-pct, pct])
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return float(lo), float(hi)


def plot_polar_field(xs_m, ys_m, Z, *, title, out_path,
                     symmetric=True, n_levels=30, n_line_levels=20):
    Zp = Z * 1e-6  # Pa -> MPa
    vmin, vmax = robust_vmin_vmax(Zp, symmetric=symmetric)
    levels      = np.linspace(vmin, vmax, n_levels)
    line_levels = np.linspace(vmin, vmax, n_line_levels)

    # Triangulate the polar point cloud once.
    triang = mtri.Triangulation(xs_m * 1e3, ys_m * 1e3)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cf = ax.tricontourf(triang, Zp, levels=levels, cmap="jet", extend="both")
    ax.tricontour(triang, Zp, levels=line_levels,
                  colors="k", linewidths=0.4, alpha=0.4)
    fig.colorbar(cf, ax=ax, shrink=0.92, label="Stress [MPa]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}  [MPa]  range=({vmin:.1f}, {vmax:.1f})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-elem", type=int, default=500)
    ap.add_argument("--n-r",     type=int, default=60,
                    help="Number of radial rings (excludes the centre point).")
    ap.add_argument("--n-theta", type=int, default=240,
                    help="Number of angular samples per ring.")
    ap.add_argument("--graded",  action="store_true", default=True,
                    help="Use circle_graded boundary mesh (default).")
    ap.add_argument("--uniform", action="store_true",
                    help="Use uniform-density boundary mesh instead.")
    ap.add_argument("--concentration", type=float, default=8.0)
    ap.add_argument("--taper-exponent", type=float, default=4.0)
    ap.add_argument("--out-dir", type=str,
                    default=str(REPO / "VCEM_4_0" / "output"
                                / "bench_disk_compression_2_cpp_polar"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    graded = (not args.uniform)

    print(f"\nProblem: R={R*1e3:.2f} mm, P={P_TOTAL:.0f} N, "
          f"E={E:.3e} Pa, nu={NU}, plane_strain={PLANE_STRAIN}")
    print(f"  Boundary: N={args.n_elem} ({'graded C=%g p=%g' % (args.concentration, args.taper_exponent) if graded else 'uniform'})")
    print(f"  Interior polar grid: n_r={args.n_r} x n_theta={args.n_theta} "
          f"= {args.n_r*args.n_theta + 1} points (incl. center)")
    print(f"  C++ OpenMP threads: {bem_cpp.openmp_max_threads()}\n")

    # ── Boundary + solve ────────────────────────────────────────────────
    mesh, is_traction, bc_x, bc_y = build_problem(
        args.n_elem, graded, args.concentration, args.taper_exponent)
    np.save(out_dir / "boundary_x1.npy", np.asarray(mesh.x1))
    np.save(out_dir / "boundary_y1.npy", np.asarray(mesh.y1))
    np.save(out_dir / "boundary_x2.npy", np.asarray(mesh.x2))
    np.save(out_dir / "boundary_y2.npy", np.asarray(mesh.y2))

    solver = make_solver(mesh, is_traction, bc_x, bc_y)
    t0 = time.perf_counter()
    solver.solve(gauss_n=GAUSS_N_SOLVE)
    t_solve = time.perf_counter() - t0
    print(f"  solve():            {t_solve*1000:8.2f} ms")

    # ── Polar interior grid + stress ────────────────────────────────────
    xs_pts, ys_pts = polar_disk_grid(args.n_r, args.n_theta)
    np.save(out_dir / "polar_xs.npy", xs_pts)
    np.save(out_dir / "polar_ys.npy", ys_pts)

    t0 = time.perf_counter()
    Sxx, Syy, Sxy = solver.stress_at_points(list(xs_pts), list(ys_pts),
                                            GAUSS_N_STRESS)
    Sxx = np.asarray(Sxx); Syy = np.asarray(Syy); Sxy = np.asarray(Sxy)
    t_stress = time.perf_counter() - t0
    print(f"  stress_at_points:   {t_stress*1000:8.2f} ms  "
          f"({len(xs_pts)} points)")

    np.save(out_dir / "Sxx_cpp.npy", Sxx)
    np.save(out_dir / "Syy_cpp.npy", Syy)
    np.save(out_dir / "Sxy_cpp.npy", Sxy)

    with open(out_dir / "timing.json", "w") as f:
        json.dump({
            "n_elem": args.n_elem, "graded": bool(graded),
            "concentration": args.concentration,
            "taper_exponent": args.taper_exponent,
            "n_r": args.n_r, "n_theta": args.n_theta,
            "n_points": len(xs_pts),
            "openmp_threads": bem_cpp.openmp_max_threads(),
            "solve_s": t_solve, "stress_s": t_stress,
        }, f, indent=2)

    # ── Plots ───────────────────────────────────────────────────────────
    print("\nWriting tricontourf plots (no rectangular raster, no clip path):")
    for name, Z in [("Sxx", Sxx), ("Syy", Syy), ("Sxy", Sxy)]:
        out = out_dir / f"{name}_cpp_polar.png"
        plot_polar_field(xs_pts, ys_pts, Z, title=f"{name} (cpp, polar)",
                         out_path=out)
        print(f"  wrote {out}")

    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
