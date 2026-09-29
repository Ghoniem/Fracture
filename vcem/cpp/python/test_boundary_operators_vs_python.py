"""Compare C++ Mt, Mu, Nbc against Python on a small straight-crack +
Brazilian-disk-boundary setup. Same fixture pattern as the assemble_operator
test, extended with a 12-segment disk boundary.
"""

import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))

import bem_cpp
from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.build_discretize import (
    discretize_polylines, allocate_unknowns,
    assemble_boundary_traction_operator      as py_Mt,
    assemble_boundary_displacement_operator  as py_Mu,
    assemble_bem_boundary_to_crack_traction_operator as py_Nbc,
)
from fracture_utils.Usolver.build_dipolar_polyline import build_polylines_full
from fracture_utils.Usolver.material import Material
from fracture_utils.Ubem.boundary_conditions import build_boundary


def make_fixture():
    # 1 mm crack along x axis, inside a 12-segment unit disk boundary.
    L = 1.0e-3
    net = CrackNetworkV4(
        vertices=[VertexV4(id=0, x=-L/2, y=0.0), VertexV4(id=1, x=L/2, y=0.0)],
        edges=[EdgeV4(id=0, v0=0, v1=1)],
    )
    polylines, _ = build_polylines_full(net)
    panels = discretize_polylines(
        net, polylines,
        n_crack_elements=10,
        node_distribution="uniform",
        collocation_mode="midpoint",
        representation="regular",
        nq_stress=4,
        tip_cluster="power", tip_cluster_power=2.0,
        endpoint_min_nodes=4,
        min_panel_length_ratio=1e-3,
    )
    deg = {0: 1, 1: 1}
    offsets, _, _, nunk = allocate_unknowns(
        panels, deg, crack_mode="full", junction_model="strict")
    material = Material(E=70e9, nu=0.3, plane_stress=False)

    mesh = build_boundary({"type": "circle", "R": 5.0e-3, "n_boundary": 12,
                           "center": (0.0, 0.0)})
    boundary_xy = np.column_stack([mesh.xm, mesh.ym])
    boundary_n  = np.column_stack([mesh.nx, mesh.ny])
    boundary_x1 = np.asarray(mesh.x1)
    boundary_y1 = np.asarray(mesh.y1)
    boundary_x2 = np.asarray(mesh.x2)
    boundary_y2 = np.asarray(mesh.y2)

    return (panels, list(offsets), int(nunk), material,
            boundary_xy, boundary_n, boundary_x1, boundary_y1, boundary_x2, boundary_y2)


def check_array(name, A_py, A_cpp, rel_tol):
    scale = max(float(np.max(np.abs(A_py))), 1e-30)
    err   = float(np.max(np.abs(A_py - A_cpp)))
    rel   = err / scale
    print(f"  {name:>4s}: shape {A_py.shape}  scale={scale:.3e}  "
          f"max|err|={err:.3e}  rel={rel:.3e}")
    assert rel < rel_tol, f"{name}: rel error {rel:.3e} > {rel_tol:.0e}"


def main():
    (panels, offsets, nunk, material,
     bxy, bn, bx1, by1, bx2, by2) = make_fixture()
    print(f"fixture: nunk={nunk}, Nb={bxy.shape[0]}, "
          f"ncol_tot={sum(len(p['x_col']) for p in panels)}")

    # ── Mt ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Mt_py = py_Mt(material=material, poly_panels=panels,
                  offsets=offsets, nunk=nunk,
                  boundary_xy=bxy, boundary_n=bn)
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    Mt_cpp = bem_cpp.assemble_boundary_traction_operator(
        list(panels), offsets, nunk,
        material.E, material.nu, bool(material.plane_stress),
        bxy, bn)
    t_cpp = time.perf_counter() - t0
    print(f"Mt: py={t_py*1000:.2f} ms  cpp={t_cpp*1000:.2f} ms  "
          f"speedup={t_py/t_cpp:.1f}x")
    check_array("Mt", Mt_py, Mt_cpp, rel_tol=1e-10)

    # ── Mu ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Mu_py = py_Mu(material=material, poly_panels=panels,
                  offsets=offsets, nunk=nunk, boundary_xy=bxy)
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    Mu_cpp = bem_cpp.assemble_boundary_displacement_operator(
        list(panels), offsets, nunk,
        material.nu, bool(material.plane_stress),
        bxy)
    t_cpp = time.perf_counter() - t0
    print(f"Mu: py={t_py*1000:.2f} ms  cpp={t_cpp*1000:.2f} ms  "
          f"speedup={t_py/t_cpp:.1f}x")
    check_array("Mu", Mu_py, Mu_cpp, rel_tol=1e-10)

    # ── Nbc ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Nbc_py = py_Nbc(material=material, poly_panels=panels,
                    boundary_x1=bx1, boundary_y1=by1,
                    boundary_x2=bx2, boundary_y2=by2,
                    gauss_n=4)
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    Nbc_cpp = bem_cpp.assemble_bem_boundary_to_crack_traction_operator(
        list(panels),
        material.E, material.nu, bool(material.plane_stress),
        bx1, by1, bx2, by2, gauss_n=4)
    t_cpp = time.perf_counter() - t0
    print(f"Nbc: py={t_py*1000:.2f} ms  cpp={t_cpp*1000:.2f} ms  "
          f"speedup={t_py/t_cpp:.1f}x")
    check_array("Nbc", Nbc_py, Nbc_cpp, rel_tol=1e-10)

    print("\nALL BOUNDARY OPERATOR TESTS PASSED")


if __name__ == "__main__":
    main()
