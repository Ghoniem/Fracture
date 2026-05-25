"""Unit-test BEMSolver2D.solve() and interior stress vs the Python implementation.

Small disk under pressurized-arc loading (pure-Neumann case to exercise the
augmented-lstsq path). Compares boundary u/t vectors and interior stress at
a few points.
"""

import sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))
sys.path.insert(0, str(REPO / "VCEM_4_0" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Ubem import bem_solver as py_bem

# ── Small Brazilian-disk-like geometry: 12-segment circle, top+bottom arcs loaded ──
N = 12
R = 1.0
E, nu = 70e9, 0.3
PS = True
P_arc_pressure = -1.0e6   # Pa, compressive normal

# Generate CCW boundary vertices
theta = np.linspace(0.0, 2.0 * np.pi, N + 1)
xs_v = R * np.cos(theta)
ys_v = R * np.sin(theta)

# Identify which segments are within ±15 deg of top (+y) or bottom (-y) for "loaded" arc
def is_loaded(xm, ym, half_angle_deg=20.0):
    angle = np.degrees(np.arctan2(ym, xm))
    # top: 90°, bottom: -90°
    return (abs(angle - 90.0) < half_angle_deg) or (abs(angle - (-90.0)) < half_angle_deg)

# Build both solvers identically
solver_py  = py_bem.BEMSolver2D(E=E, nu=nu, h=1.0, plane_strain=PS)
solver_cpp = bem_cpp.BEMSolver2D(E=E, nu=nu, h=1.0, plane_strain=PS)

for k in range(N):
    x1, y1 = xs_v[k],     ys_v[k]
    x2, y2 = xs_v[k + 1], ys_v[k + 1]
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    if is_loaded(xm, ym):
        # Pressure normal -> traction t = P * n. For a CCW disk normal is (ty, -tx),
        # pointing outward. Compressive pressure pulls inward, so t = P_arc * n.
        dx, dy = x2 - x1, y2 - y1
        L = (dx * dx + dy * dy) ** 0.5
        nx, ny = dy / L, -dx / L
        tx, ty = P_arc_pressure * nx, P_arc_pressure * ny
        for s in (solver_py, solver_cpp):
            s.add_element(x1, y1, x2, y2, True, tx, ty)
    else:
        # Traction-free
        for s in (solver_py, solver_cpp):
            s.add_element(x1, y1, x2, y2, True, 0.0, 0.0)

# ── Solve both ─────────────────────────────────────────────────────────────
t0 = time.perf_counter()
solver_py.solve(gauss_n=8)
t_py = time.perf_counter() - t0

t0 = time.perf_counter()
solver_cpp.solve(gauss_n=8)
t_cpp = time.perf_counter() - t0

print(f"solve() timing: Python = {t_py*1000:.2f} ms   C++ = {t_cpp*1000:.2f} ms   speedup = {t_py/t_cpp:.1f}x")

# ── Compare boundary u and t ──────────────────────────────────────────────
ux_py, uy_py = np.asarray(solver_py.u_x), np.asarray(solver_py.u_y)
tx_py, ty_py = np.asarray(solver_py.t_x), np.asarray(solver_py.t_y)
ux_c,  uy_c  = np.asarray(solver_cpp.u_x), np.asarray(solver_cpp.u_y)
tx_c,  ty_c  = np.asarray(solver_cpp.t_x), np.asarray(solver_cpp.t_y)

def rel_err(a, b, ref_scale):
    return float(np.max(np.abs(a - b)) / max(ref_scale, 1e-30))

# Scale for u: typical magnitude is P*R/E ~ 1e6*1/7e10 ~ 1.4e-5
u_scale = max(np.max(np.abs(ux_py)), np.max(np.abs(uy_py)), 1e-30)
t_scale = max(np.max(np.abs(tx_py)), np.max(np.abs(ty_py)), 1e-30)

print(f"max|u_x diff| = {np.max(np.abs(ux_py - ux_c)):.3e}   rel = {rel_err(ux_py, ux_c, u_scale):.3e}")
print(f"max|u_y diff| = {np.max(np.abs(uy_py - uy_c)):.3e}   rel = {rel_err(uy_py, uy_c, u_scale):.3e}")
print(f"max|t_x diff| = {np.max(np.abs(tx_py - tx_c)):.3e}   rel = {rel_err(tx_py, tx_c, t_scale):.3e}")
print(f"max|t_y diff| = {np.max(np.abs(ty_py - ty_c)):.3e}   rel = {rel_err(ty_py, ty_c, t_scale):.3e}")

# Pure-Neumann SVD may differ by tiny rigid-body-mode terms across backends.
# Acceptable: relative <1e-8 on u, near-machine on t (since t is mostly known here).
assert rel_err(ux_py, ux_c, u_scale) < 1e-8, "u_x mismatch beyond rigid-body tolerance"
assert rel_err(uy_py, uy_c, u_scale) < 1e-8, "u_y mismatch beyond rigid-body tolerance"
assert rel_err(tx_py, tx_c, t_scale) < 1e-10, "t_x mismatch (known BCs should be identical)"
assert rel_err(ty_py, ty_c, t_scale) < 1e-10, "t_y mismatch"

# ── Compare interior stress at a few interior points ──────────────────────
probes = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.5, 0.5)]
print("\nInterior stress comparison (Pa):")
print(f"{'(x, y)':16s}  {'sxx_py':>14s} {'sxx_c':>14s}   {'syy_py':>14s} {'syy_c':>14s}   {'sxy_py':>14s} {'sxy_c':>14s}")
max_stress_diff = 0.0
for (x, y) in probes:
    sxx_py, syy_py, sxy_py = solver_py.compute_stress_at_point(x, y, gauss_n=12)
    sxx_c,  syy_c,  sxy_c  = solver_cpp.compute_stress_at_point(x, y, gauss_n=12)
    print(f"({x:+.2f}, {y:+.2f})    {sxx_py:14.5e} {sxx_c:14.5e}   "
          f"{syy_py:14.5e} {syy_c:14.5e}   {sxy_py:14.5e} {sxy_c:14.5e}")
    max_stress_diff = max(max_stress_diff,
                          abs(sxx_py - sxx_c), abs(syy_py - syy_c), abs(sxy_py - sxy_c))

print(f"\nmax |stress diff| = {max_stress_diff:.3e} Pa  (typical magnitude ~{abs(P_arc_pressure):.0e})")
assert max_stress_diff / abs(P_arc_pressure) < 1e-8, "interior stress mismatch"

print("\nSOLVE + INTERIOR STRESS TESTS PASSED")
