"""End-to-end test of the engine='cpp' dispatch in parametrization.py.

Solves the same crack problem (single straight crack under remote tension)
twice -- once with the default Python engine and once with the cpp engine
-- and checks the recovered crack-unknown vector q matches to machine
precision.

This is the integration test that proves the full Python->C++ dispatch
path works for the non-augmented case. The augmented (BEM-coupled) case
goes through additional code paths and is covered separately.
"""

import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))
sys.path.insert(0, str(REPO / "VCEM_4_0" / "cpp" / "python"))

import bem_cpp  # ensure the C++ ext is importable before we start

from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.material import Material, AppliedStress
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4


def run(engine: str, n_crack_elements: int = 20):
    L = 1.0e-3
    net = CrackNetworkV4(
        vertices=[VertexV4(id=0, x=-L/2, y=0.0), VertexV4(id=1, x=L/2, y=0.0)],
        edges=[EdgeV4(id=0, v0=0, v1=1)],
    )
    material = Material(E=70e9, nu=0.3, plane_stress=False)
    applied  = AppliedStress(sigma_xx=0.0, sigma_yy=-1.0e6, sigma_xy=0.0)
    solver = DCENetworkStaticV4(material=material, network=net, applied=applied)

    t0 = time.perf_counter()
    sol = solver.solve(
        n_crack_elements=n_crack_elements,
        crack_mode="full",
        junction_model="strict",
        representation="panel",
        node_distribution="uniform",
        collocation_mode="midpoint",
        nq_stress=4,
        nq_col=4,
        ridge=0.0,
        # Engine flag goes through _ignored / **kwargs into solve()
        engine=engine,
    )
    t = time.perf_counter() - t0
    return sol, t


def extract_q(sol):
    """Concatenate (bI_hat, bII_hat) per polyline into a single vector."""
    pieces = []
    for ps in sol["polyline_solutions"]:
        pieces.append(ps["bI_hat"])
        pieces.append(ps["bII_hat"])
    return np.concatenate(pieces)


print("=== engine='python' ===")
sol_py, t_py = run(engine="python", n_crack_elements=20)
q_py = extract_q(sol_py)
print(f"  wall = {t_py*1000:.2f} ms   ||q|| = {np.linalg.norm(q_py):.6e}   "
      f"shape = {q_py.shape}")

print("\n=== engine='cpp' ===")
sol_cpp, t_cpp = run(engine="cpp", n_crack_elements=20)
q_cpp = extract_q(sol_cpp)
print(f"  wall = {t_cpp*1000:.2f} ms   ||q|| = {np.linalg.norm(q_cpp):.6e}   "
      f"shape = {q_cpp.shape}   (full-solve speedup = {t_py/t_cpp:.1f}x)")

assert q_py.shape == q_cpp.shape, f"shape mismatch: {q_py.shape} vs {q_cpp.shape}"
scale = max(float(np.max(np.abs(q_py))), 1e-30)
err   = float(np.max(np.abs(q_py - q_cpp)))
rel   = err / scale
print(f"\nmax|q_py - q_cpp| = {err:.3e}   rel = {rel:.3e}")
assert rel < 1e-9, f"end-to-end q mismatch beyond tolerance: rel={rel:.3e}"
print("\nEND-TO-END TEST PASSED (engine='python' == engine='cpp')")
