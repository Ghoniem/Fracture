"""Verify the spatial-hash _merge_close_vertices produces the same final
graph as the brute-force reference, and report speedup at growing V.

We test by deep-copying a randomly generated network, running each
implementation independently, and checking the resulting node sets and
edge sets are identical.
"""

import sys
import time
import copy
from pathlib import Path
import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))

from fracture_utils.Ugenerator.crack_network_simplifier import (
    CrackNetworkSimplifier, SimplificationConfig,
)


def make_random_network(V: int, *, p_close: float = 0.10, p_edge: float = 0.04,
                        domain: float = 1.0, tol: float = 0.01, seed: int = 0):
    """Build a random graph with `V` vertices, ~p_close fraction within
    `tol` of an earlier vertex (forces merges to happen)."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    for i in range(V):
        if i > 0 and rng.random() < p_close:
            # Place near an earlier vertex
            j = int(rng.integers(0, i))
            base = np.asarray(G.nodes[j]['pos'])
            jitter = (rng.standard_normal(2) * tol * 0.4)
            pos = base + jitter
        else:
            pos = rng.uniform(0.0, domain, size=2)
        G.add_node(i, pos=(float(pos[0]), float(pos[1])))

    # Sparse edges
    for i in range(V):
        for j in range(i + 1, V):
            if rng.random() < p_edge:
                G.add_edge(i, j)
    return G


def run_one(V: int, tol: float, seed: int):
    G0 = make_random_network(V, tol=tol, seed=seed)
    cfg = SimplificationConfig(merge_vertex_tolerance=tol)

    # Brute force
    sim_bf = CrackNetworkSimplifier(cfg)
    sim_bf.G = copy.deepcopy(G0)
    t0 = time.perf_counter()
    nodes = list(sim_bf.G.nodes())
    merged_bf = sim_bf._merge_close_vertices_bruteforce(nodes, tol)
    t_bf = time.perf_counter() - t0
    nodes_bf = set(sim_bf.G.nodes())
    edges_bf = set(frozenset(e) for e in sim_bf.G.edges())

    # Spatial hash
    sim_sh = CrackNetworkSimplifier(cfg)
    sim_sh.G = copy.deepcopy(G0)
    t0 = time.perf_counter()
    merged_sh = sim_sh._merge_close_vertices()
    t_sh = time.perf_counter() - t0
    nodes_sh = set(sim_sh.G.nodes())
    edges_sh = set(frozenset(e) for e in sim_sh.G.edges())

    # Equivalence
    assert merged_bf == merged_sh, f"V={V}: merge counts differ {merged_bf} vs {merged_sh}"
    assert nodes_bf == nodes_sh, f"V={V}: surviving node sets differ"
    assert edges_bf == edges_sh, f"V={V}: edge sets differ"

    return merged_bf, t_bf, t_sh


def main():
    tol = 0.01
    print(f"tol = {tol}")
    print(f"{'V':>6s} {'merged':>7s} {'tbf[ms]':>10s} {'tsh[ms]':>10s} {'speedup':>10s}")
    for V in (100, 500, 1000, 2000, 5000):
        merged, t_bf, t_sh = run_one(V, tol=tol, seed=42)
        speedup = t_bf / max(t_sh, 1e-9)
        print(f"{V:>6d} {merged:>7d} {t_bf*1000:>10.1f} {t_sh*1000:>10.1f} {speedup:>9.1f}x")
    print("\nSPATIAL-HASH MERGE TEST PASSED (equivalent results, much faster)")


if __name__ == "__main__":
    main()
