"""KKT-engine scaling sweep: full-solve wall-clock vs crack-network size,
Python vs C++ engines. Plots speedup curve so we can see whether the
complex-network speedup grows the way the scaling theory predicts.

Network topology: N_cracks parallel straight cracks at varying vertical
offsets, each 1 mm long. Same n_crack_elements per crack. Same applied
remote uniaxial stress for every run. Auto-merge guard active.
"""

import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))

import bem_cpp                                              # noqa: F401
from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.material import Material, AppliedStress
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_parallel_cracks(N: int, L: float = 1.0e-3, gap: float = 2.0e-3):
    """N parallel straight cracks at y = -((N-1)/2)*gap + k*gap."""
    vertices = []
    edges = []
    for k in range(N):
        y = (k - (N - 1) / 2.0) * gap
        v0 = VertexV4(id=2 * k,     x=-L / 2, y=y)
        v1 = VertexV4(id=2 * k + 1, x= L / 2, y=y)
        vertices += [v0, v1]
        edges.append(EdgeV4(id=k, v0=v0.id, v1=v1.id))
    return CrackNetworkV4(vertices=vertices, edges=edges)


def run_one(N_cracks: int, n_crack_elements: int, engine: str):
    net = make_parallel_cracks(N=N_cracks)
    mat = Material(E=70e9, nu=0.3, plane_stress=False)
    app = AppliedStress(sigma_xx=0.0, sigma_yy=-1.0e6, sigma_xy=0.0)
    solver = DCENetworkStaticV4(material=mat, network=net, applied=app)

    t0 = time.perf_counter()
    sol = solver.solve(
        n_crack_elements=n_crack_elements,
        crack_mode="full",
        junction_model="strict",
        representation="panel",
        node_distribution="uniform",
        collocation_mode="midpoint",
        nq_stress=4, nq_col=4,
        ridge=0.0,
        engine=engine,
    )
    t = time.perf_counter() - t0
    n_panels = sum(len(ps["s_mid"]) for ps in sol["polyline_solutions"])
    return t, n_panels


def main():
    N_list = [1, 3, 5, 10, 20, 40]
    n_crack_elements = 20

    print(f"{'N_cracks':>8s} {'Npanels':>8s} {'tpy[ms]':>10s} {'tcpp[ms]':>10s} {'speedup':>9s}")
    results = []
    for N in N_list:
        # Warm-up the C++ once to avoid first-call JIT/cache effects
        run_one(N, n_crack_elements, engine="cpp")
        t_py,  npan = run_one(N, n_crack_elements, engine="python")
        t_cpp, _    = run_one(N, n_crack_elements, engine="cpp")
        speedup = t_py / t_cpp
        print(f"{N:>8d} {npan:>8d} {t_py*1000:>10.1f} {t_cpp*1000:>10.1f} {speedup:>8.1f}x")
        results.append((N, npan, t_py, t_cpp, speedup))

    # ── Plot ────────────────────────────────────────────────────────────
    Ns      = [r[0] for r in results]
    npans   = [r[1] for r in results]
    t_pys   = [r[2] * 1000 for r in results]
    t_cpps  = [r[3] * 1000 for r in results]
    speedups= [r[4] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(npans, t_pys,  "o-", label="Python engine")
    ax1.loglog(npans, t_cpps, "s-", label="C++ engine")
    ax1.set_xlabel("Total crack panels")
    ax1.set_ylabel("Full-solve wall-clock [ms]")
    ax1.set_title("KKT pipeline wall-clock (1mm parallel cracks, n_elem=20 per crack)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.semilogx(npans, speedups, "o-", color="C2")
    ax2.set_xlabel("Total crack panels")
    ax2.set_ylabel("Speedup (Python / C++)")
    ax2.set_title("Full-solve speedup vs network size")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.axhline(1.0, color="k", linewidth=0.5, alpha=0.5)

    out_dir = REPO / "vcem" / "output" / "bench_kkt_scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = out_dir / "kkt_scaling.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Save raw numbers
    import json
    with open(out_dir / "kkt_scaling.json", "w") as f:
        json.dump({"records": [
            {"N_cracks": r[0], "n_panels": r[1],
             "py_s": r[2], "cpp_s": r[3], "speedup": r[4]} for r in results]},
                  f, indent=2)


if __name__ == "__main__":
    main()
