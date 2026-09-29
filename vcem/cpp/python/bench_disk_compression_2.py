"""Head-to-head BEM solve+stress benchmark for the disk_compression_2 case.

Builds the Brazilian disk problem (R=12.7 mm, P=3.8 kN, E=231.52 GPa, nu=0.3,
N=60 boundary elements) once, then runs both the Python BEMSolver2D and the
C++ bem_cpp.BEMSolver2D on it. Reports timing, numerical agreement
(absolute, relative, RMS), and writes paired .npy arrays.

Usage:
    python bench_disk_compression_2.py [--n-grid 120] [--engines both|cpp|python]

Output:
    vcem/output/bench_disk_compression_2/
        {Sxx,Syy,Sxy}_py.npy   (if Python engine ran)
        {Sxx,Syy,Sxy}_cpp.npy  (if C++ engine ran)
        xs.npy, ys.npy
        timing.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Ubem import bem_solver as py_bem
from fracture_utils.Ubem.boundary_conditions import (
    build_boundary, BCSpec, assemble_segment_bcs,
)


# ── Problem definition (matches VCM_3_3/notebooks/disk_compression_2.ipynb) ──
R = 25.4e-3 / 2.0          # disk radius (m), 12.7 mm
P_TOTAL = 3.8e3            # total compressive force per platen (N)
E = 231.52e9               # Young's modulus (Pa)
NU = 0.3
PLANE_STRAIN = True
H_THICK = 6.35e-3          # disk thickness (m)
N_ELEM = 60                # boundary elements (uniform default)
GAUSS_N_SOLVE = 4          # quadrature order in the matrix assembly
GAUSS_N_STRESS = 12        # quadrature order in interior stress eval
ARC_HALF_ANGLE = 15.0      # loaded arc half-angle (deg)
PAD_FRAC = 0.03            # interior pad as fraction of R


def build_problem(n_grid: int,
                  *,
                  n_elem: int = N_ELEM,
                  graded: bool = False,
                  concentration: float = 8.0,
                  taper_exponent: float = 4.0):
    """Build boundary, BCs, and the interior grid for both solvers."""
    if graded:
        mesh = build_boundary({
            "type": "circle_graded",
            "R": R,
            "n_boundary": n_elem,
            "center": (0.0, 0.0),
            "focal_angles_deg": (90.0, -90.0),
            "concentration":  concentration,
            "taper_exponent": taper_exponent,
        })
    else:
        mesh = build_boundary({
            "type": "circle",
            "R": R,
            "n_boundary": n_elem,
            "center": (0.0, 0.0),
        })

    theta_deg = np.asarray(mesh.theta_deg, dtype=float)
    L         = np.asarray(mesh.length, dtype=float)

    arc = ARC_HALF_ANGLE
    # Match the 1e-9 deg tolerance used inside _select_indices() so the
    # L_top / L_bot pressure normalization here lines up with the segments
    # assemble_segment_bcs actually selects.
    eps = 1.0e-9
    top = (theta_deg >=  90 - arc - eps) & (theta_deg <=  90 + arc + eps)
    bot = (theta_deg >= -90 - arc - eps) & (theta_deg <= -90 + arc + eps)
    L_top = float(np.sum(L[top]))
    L_bot = float(np.sum(L[bot]))
    pressure_top = P_TOTAL / (L_top * H_THICK)
    pressure_bot = P_TOTAL / (L_bot * H_THICK)

    bc_specs = [
        BCSpec("pressure_normal", pressure_top, "theta_deg_range", ( 90 - arc,  90 + arc)),
        BCSpec("pressure_normal", pressure_bot, "theta_deg_range", (-90 - arc, -90 + arc)),
    ]
    is_traction, bc_x, bc_y = assemble_segment_bcs(
        mesh, bc_specs=bc_specs, default=("traction", 0.0, 0.0))

    pad = PAD_FRAC * R
    xs = np.linspace(-R + pad, R - pad, n_grid)
    ys = np.linspace(-R + pad, R - pad, n_grid)

    return mesh, is_traction, bc_x, bc_y, xs, ys


def make_solver(mesh, is_traction, bc_x, bc_y, *, engine: str):
    if engine == "python":
        s = py_bem.BEMSolver2D(E=E, nu=NU, h=H_THICK, plane_strain=PLANE_STRAIN)
    else:
        s = bem_cpp.BEMSolver2D(E=E, nu=NU, h=H_THICK, plane_strain=PLANE_STRAIN)

    for k in range(mesh.n_seg):
        s.add_element(float(mesh.x1[k]), float(mesh.y1[k]),
                      float(mesh.x2[k]), float(mesh.y2[k]),
                      bool(is_traction[k]),
                      float(bc_x[k]), float(bc_y[k]))
    return s


def run_python(mesh, is_traction, bc_x, bc_y, xs, ys, mask):
    s = make_solver(mesh, is_traction, bc_x, bc_y, engine="python")

    t0 = time.perf_counter()
    s.solve(gauss_n=GAUSS_N_SOLVE)
    t_solve = time.perf_counter() - t0
    print(f"  Python solve:           {t_solve:8.2f} s")

    nx, ny = len(xs), len(ys)
    Sxx = np.full((ny, nx), np.nan)
    Syy = np.full((ny, nx), np.nan)
    Sxy = np.full((ny, nx), np.nan)

    n_evals = 0
    t0 = time.perf_counter()
    last_report = t0
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if not mask[i, j]:
                continue
            sxx, syy, sxy = s.compute_stress_at_point(x, y, gauss_n=GAUSS_N_STRESS)
            Sxx[i, j] = sxx; Syy[i, j] = syy; Sxy[i, j] = sxy
            n_evals += 1
        now = time.perf_counter()
        if now - last_report > 10.0:
            done_frac = (i + 1) / ny
            elapsed = now - t0
            eta = elapsed / done_frac - elapsed
            print(f"  Python stress: row {i+1}/{ny} done ({100*done_frac:.0f}%, "
                  f"{n_evals} evals, elapsed {elapsed:.0f} s, ETA {eta:.0f} s)")
            last_report = now
    t_stress = time.perf_counter() - t0
    print(f"  Python stress on grid:  {t_stress:8.2f} s  ({n_evals} evals)")

    return t_solve, t_stress, n_evals, Sxx, Syy, Sxy


def run_cpp(mesh, is_traction, bc_x, bc_y, xs, ys, mask):
    s = make_solver(mesh, is_traction, bc_x, bc_y, engine="cpp")
    print(f"  C++ OpenMP threads:     {bem_cpp.openmp_max_threads()}")

    t0 = time.perf_counter()
    s.solve(gauss_n=GAUSS_N_SOLVE)
    t_solve = time.perf_counter() - t0
    print(f"  C++ solve:              {t_solve*1000:8.2f} ms")

    t0 = time.perf_counter()
    Sxx, Syy, Sxy = s.stress_on_grid(list(xs), list(ys), GAUSS_N_STRESS)
    t_stress = time.perf_counter() - t0
    print(f"  C++ stress on grid:     {t_stress:8.2f} s  ({mask.size} evals)")

    # Apply mask: zero out / NaN outside-of-disk points
    Sxx = np.where(mask, Sxx, np.nan)
    Syy = np.where(mask, Syy, np.nan)
    Sxy = np.where(mask, Sxy, np.nan)

    return t_solve, t_stress, int(mask.sum()), Sxx, Syy, Sxy


def compare_arrays(name: str, A_py, A_cpp):
    diff = A_cpp - A_py
    valid = np.isfinite(diff)
    scale = float(np.nanmax(np.abs(A_py)))
    max_abs = float(np.nanmax(np.abs(diff[valid]))) if valid.any() else 0.0
    rms     = float(np.sqrt(np.nanmean(diff[valid] ** 2))) if valid.any() else 0.0
    rel_max = max_abs / max(scale, 1e-30)
    print(f"  {name:>4s}: scale={scale:.3e}  max|cpp-py|={max_abs:.3e}  "
          f"rms={rms:.3e}  rel_max={rel_max:.3e}")
    return {"scale": scale, "max_abs": max_abs, "rms": rms, "rel_max": rel_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=120,
                    help="Interior grid resolution (default 120 matches the notebook).")
    ap.add_argument("--n-elem", type=int, default=N_ELEM,
                    help=f"Boundary element count (default {N_ELEM}).")
    ap.add_argument("--graded", action="store_true",
                    help="Use circle_graded mesh concentrated under the platens.")
    ap.add_argument("--concentration", type=float, default=8.0,
                    help="Graded mesh: density ratio at platen vs equator (default 8).")
    ap.add_argument("--taper-exponent", type=float, default=4.0,
                    help="Graded mesh: cos^p kernel exponent (default 4 -> ~+/-45 deg width).")
    ap.add_argument("--engines", choices=["both", "cpp", "python"], default="both")
    ap.add_argument("--out-dir", type=str,
                    default=str(REPO / "vcem" / "output" / "bench_disk_compression_2"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_kind = (f"graded(C={args.concentration:g}, p={args.taper_exponent:g})"
                 if args.graded else "uniform")
    print(f"\nProblem: R={R*1e3:.2f} mm, P={P_TOTAL:.1f} N, E={E:.3e} Pa, nu={NU}")
    print(f"         N_elem={args.n_elem} ({mesh_kind}), gauss_n(solve)={GAUSS_N_SOLVE}, "
          f"gauss_n(stress)={GAUSS_N_STRESS}, n_grid={args.n_grid}")
    print(f"Engines: {args.engines}\n")

    mesh, is_traction, bc_x, bc_y, xs, ys = build_problem(
        args.n_grid,
        n_elem=args.n_elem,
        graded=args.graded,
        concentration=args.concentration,
        taper_exponent=args.taper_exponent,
    )
    # Save the mesh segment endpoints for post-hoc visualization.
    np.save(out_dir / "boundary_x1.npy", np.asarray(mesh.x1))
    np.save(out_dir / "boundary_y1.npy", np.asarray(mesh.y1))
    np.save(out_dir / "boundary_x2.npy", np.asarray(mesh.x2))
    np.save(out_dir / "boundary_y2.npy", np.asarray(mesh.y2))
    np.save(out_dir / "xs.npy", xs)
    np.save(out_dir / "ys.npy", ys)

    # Disk mask: keep points strictly inside R - pad to avoid singularity issues
    Xg, Yg = np.meshgrid(xs, ys)
    pad = PAD_FRAC * R
    mask = (Xg**2 + Yg**2) <= (R - pad) ** 2
    print(f"Mask: {int(mask.sum())} of {mask.size} grid points inside the disk\n")

    timing = {"n_grid": args.n_grid, "n_elem": N_ELEM,
              "openmp_threads": bem_cpp.openmp_max_threads()}

    if args.engines in ("both", "cpp"):
        print("=== C++ engine ===")
        t_solve_c, t_stress_c, n_evals_c, Sxx_c, Syy_c, Sxy_c = \
            run_cpp(mesh, is_traction, bc_x, bc_y, xs, ys, mask)
        np.save(out_dir / "Sxx_cpp.npy", Sxx_c)
        np.save(out_dir / "Syy_cpp.npy", Syy_c)
        np.save(out_dir / "Sxy_cpp.npy", Sxy_c)
        timing["cpp"] = {"solve_s": t_solve_c, "stress_s": t_stress_c,
                         "n_evals": n_evals_c}
        print()

    if args.engines in ("both", "python"):
        print("=== Python engine ===")
        t_solve_p, t_stress_p, n_evals_p, Sxx_p, Syy_p, Sxy_p = \
            run_python(mesh, is_traction, bc_x, bc_y, xs, ys, mask)
        np.save(out_dir / "Sxx_py.npy", Sxx_p)
        np.save(out_dir / "Syy_py.npy", Syy_p)
        np.save(out_dir / "Sxy_py.npy", Sxy_p)
        timing["python"] = {"solve_s": t_solve_p, "stress_s": t_stress_p,
                            "n_evals": n_evals_p}
        print()

    if args.engines == "both":
        print("=== Numerical agreement (cpp vs python) ===")
        agreement = {
            "Sxx": compare_arrays("Sxx", Sxx_p, Sxx_c),
            "Syy": compare_arrays("Syy", Syy_p, Syy_c),
            "Sxy": compare_arrays("Sxy", Sxy_p, Sxy_c),
        }
        timing["agreement"] = agreement

        print("\n=== Speedups ===")
        timing["speedup_solve"]  = t_solve_p  / t_solve_c
        timing["speedup_stress"] = t_stress_p / t_stress_c
        total_p = t_solve_p + t_stress_p
        total_c = t_solve_c + t_stress_c
        timing["speedup_total"]  = total_p / total_c
        print(f"  solve():       {timing['speedup_solve']:6.1f}x")
        print(f"  stress_grid(): {timing['speedup_stress']:6.1f}x")
        print(f"  total:         {timing['speedup_total']:6.1f}x  "
              f"(Python {total_p:.1f}s -> C++ {total_c:.1f}s)")

    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\nResults written to: {out_dir}")


if __name__ == "__main__":
    main()
