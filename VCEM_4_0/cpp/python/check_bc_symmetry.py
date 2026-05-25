"""Inspect the boundary mesh + assembled BC vector for the disk_compression_2
setup and report whether they are symmetric under x->-x and y->-y.

If the mesh/BCs are already asymmetric, that explains an asymmetric stress
field (and is a BC-spec problem, not a solver problem).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))

from fracture_utils.Ubem.boundary_conditions import (
    build_boundary, BCSpec, assemble_segment_bcs,
)

R = 25.4e-3 / 2.0
N_ELEM = 60
P_TOTAL = 3.8e3
H_THICK = 6.35e-3
ARC = 15.0

mesh = build_boundary({"type": "circle", "R": R, "n_boundary": N_ELEM, "center": (0.0, 0.0)})

theta = np.asarray(mesh.theta_deg, dtype=float)
L     = np.asarray(mesh.length,    dtype=float)
top   = (theta >=  90 - ARC) & (theta <=  90 + ARC)
bot   = (theta >= -90 - ARC) & (theta <= -90 + ARC)
L_top = float(np.sum(L[top]))
L_bot = float(np.sum(L[bot]))
pressure_top = P_TOTAL / (L_top * H_THICK)
pressure_bot = P_TOTAL / (L_bot * H_THICK)

bc_specs = [
    BCSpec("pressure_normal", pressure_top, "theta_deg_range", ( 90 - ARC,  90 + ARC)),
    BCSpec("pressure_normal", pressure_bot, "theta_deg_range", (-90 - ARC, -90 + ARC)),
]
is_traction, bc_x, bc_y = assemble_segment_bcs(
    mesh, bc_specs=bc_specs, default=("traction", 0.0, 0.0))

xm = np.asarray(mesh.xm, dtype=float)
ym = np.asarray(mesh.ym, dtype=float)

print(f"N = {mesh.n_seg}")
print(f"loaded: top={int(top.sum())}, bot={int(bot.sum())}")
print(f"L_top = {L_top:.6e}, L_bot = {L_bot:.6e}, equal? {L_top == L_bot}")
print(f"pressure_top = {pressure_top:.6e}, pressure_bot = {pressure_bot:.6e}, equal? {pressure_top == pressure_bot}")

# For each segment, find its mirror across x-axis (y -> -y) by matching xm and -ym
# and across y-axis (x -> -x) by matching -xm and ym.
def find_mirror(target_xm, target_ym, xm, ym, tol):
    d = np.hypot(xm - target_xm, ym - target_ym)
    j = int(np.argmin(d))
    if d[j] > tol:
        return -1, float(d[j])
    return j, float(d[j])

tol = 1e-12
y_axis_pairs = []
x_axis_pairs = []
for i in range(mesh.n_seg):
    jy, dy = find_mirror(-xm[i],  ym[i], xm, ym, tol)
    jx, dx = find_mirror( xm[i], -ym[i], xm, ym, tol)
    y_axis_pairs.append((i, jy, dy))
    x_axis_pairs.append((i, jx, dx))

bad_y = sum(1 for _, j, _ in y_axis_pairs if j < 0)
bad_x = sum(1 for _, j, _ in x_axis_pairs if j < 0)
print(f"\nmidpoint mirror coverage: y-axis flip missing={bad_y}, x-axis flip missing={bad_x}")

# Mesh symmetry: midpoints should pair perfectly. If they do, the geometry is symmetric.
print(f"max midpoint mirror distance: y-axis flip = {max(d for _,_,d in y_axis_pairs):.3e}, "
      f"x-axis flip = {max(d for _,_,d in x_axis_pairs):.3e}")

# BC symmetry under reflection. For mirror across y-axis: (bc_x, bc_y) at segment i
# should match (-bc_x, bc_y) at the mirrored segment (because normal x-component flips).
# For mirror across x-axis: (bc_x, bc_y) at segment i should match (bc_x, -bc_y) at the
# mirrored segment.
print("\nBC vector symmetry checks (Pa):")
print(f"{'i':>3s} {'xm':>10s} {'ym':>10s} {'bc_x':>14s} {'bc_y':>14s}    "
      f"{'i_y_mirror':>10s} {'expect_bc':>14s} {'actual_bc':>14s} {'delta':>10s}")
errs_y_x = []
errs_y_y = []
errs_x_x = []
errs_x_y = []
for i in range(mesh.n_seg):
    jy = y_axis_pairs[i][1]; jx = x_axis_pairs[i][1]
    if jy >= 0:
        expect_bx = -bc_x[i]; expect_by =  bc_y[i]
        errs_y_x.append(bc_x[jy] - expect_bx)
        errs_y_y.append(bc_y[jy] - expect_by)
    if jx >= 0:
        expect_bx =  bc_x[i]; expect_by = -bc_y[i]
        errs_x_x.append(bc_x[jx] - expect_bx)
        errs_x_y.append(bc_y[jx] - expect_by)

def stat(name, arr):
    a = np.array(arr)
    if a.size == 0:
        print(f"  {name}: (empty)"); return
    print(f"  {name}: max|err|={np.max(np.abs(a)):.3e}  rms={np.sqrt(np.mean(a**2)):.3e}")

stat("y-axis flip   bc_x delta", errs_y_x)
stat("y-axis flip   bc_y delta", errs_y_y)
stat("x-axis flip   bc_x delta", errs_x_x)
stat("x-axis flip   bc_y delta", errs_x_y)

# Also: print the loaded segments verbatim to eyeball them.
print("\nLoaded segments (top + bot):")
print(f"{'i':>3s} {'theta':>8s} {'xm':>10s} {'ym':>10s} {'nx':>10s} {'ny':>10s} "
      f"{'bc_x':>14s} {'bc_y':>14s} {'is_trac':>7s}")
for i in range(mesh.n_seg):
    if not (top[i] or bot[i]):
        continue
    print(f"{i:>3d} {theta[i]:>8.2f} {xm[i]:>10.3e} {ym[i]:>10.3e} "
          f"{mesh.nx[i]:>10.4f} {mesh.ny[i]:>10.4f} "
          f"{bc_x[i]:>14.4e} {bc_y[i]:>14.4e} {bool(is_traction[i])!r:>7s}")
