"""Compare C++ detect_segment_intersections vs the Python brute-force
reference in generator.py. Tests semantic equivalence on a random
network and reports speedup at growing edge count.
"""

import sys
import time
from pathlib import Path
import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))

import bem_cpp


def make_random_network(E: int, *, domain: float = 1.0, edge_frac: float = 0.4, seed: int = 0):
    """Random network with ~E edges. Vertices placed uniformly in the
    domain; each candidate edge accepted with probability `edge_frac`
    until we have E edges.
    """
    rng = np.random.default_rng(seed)
    # Aim for V such that E edges are likely from V*(V-1)*edge_frac/2
    V = max(int(np.ceil(np.sqrt(2.0 * E / edge_frac))), 4)
    positions = rng.uniform(0.0, domain, size=(V, 2))
    edges = []
    for i in range(V):
        for j in range(i + 1, V):
            if rng.random() < edge_frac:
                edges.append((i, j))
                if len(edges) >= E:
                    break
        if len(edges) >= E:
            break
    return positions, np.asarray(edges, dtype=int)


def py_brute_force(positions, edges, endpoint_tol=0.05, parallel_tol=1e-10):
    """Reference: matches generator.py._find_edge_intersection +
    _detect_single_pass inner math."""
    out = []
    for ii in range(len(edges)):
        a1, a2 = edges[ii]
        p1 = positions[a1]; p2 = positions[a2]
        for jj in range(ii + 1, len(edges)):
            b1, b2 = edges[jj]
            if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
                continue
            p3 = positions[b1]; p4 = positions[b2]
            d1 = p2 - p1
            d2 = p4 - p3
            d3 = p3 - p1
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < parallel_tol:
                continue
            t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
            s = (d3[0] * d1[1] - d3[1] * d1[0]) / cross
            tol = endpoint_tol
            if tol < t < 1.0 - tol and tol < s < 1.0 - tol:
                pt = p1 + t * d1
                out.append((ii, jj, pt[0], pt[1]))
    return out


def run_one(E_target: int, seed: int = 0):
    pos, edges = make_random_network(E_target, seed=seed)
    edges_p1 = pos[edges[:, 0]]
    edges_p2 = pos[edges[:, 1]]

    # Python brute force
    t0 = time.perf_counter()
    py_recs = py_brute_force(pos, edges)
    t_py = time.perf_counter() - t0

    # C++
    t0 = time.perf_counter()
    cpp_out = bem_cpp.detect_segment_intersections(
        edges_p1, edges_p2, edges.astype(np.int32), 0.05, 1e-10)
    t_cpp = time.perf_counter() - t0

    # Compare
    cpp_recs = [(int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in cpp_out]
    py_sorted = sorted(py_recs, key=lambda r: (r[0], r[1]))
    assert len(py_sorted) == len(cpp_recs), \
        f"E={len(edges)}: counts differ {len(py_sorted)} (py) vs {len(cpp_recs)} (cpp)"
    for (rp, rc) in zip(py_sorted, cpp_recs):
        assert rp[0] == rc[0] and rp[1] == rc[1], f"index mismatch: {rp} vs {rc}"
        assert abs(rp[2] - rc[2]) < 1e-12 and abs(rp[3] - rc[3]) < 1e-12, \
            f"point mismatch: {rp} vs {rc}"

    return len(edges), len(py_sorted), t_py, t_cpp


def main():
    print(f"{'E':>6s} {'found':>6s} {'tpy[ms]':>10s} {'tcpp[ms]':>10s} {'speedup':>10s}")
    for E in (50, 200, 500, 1000, 2000):
        E_actual, n_found, t_py, t_cpp = run_one(E_target=E, seed=42)
        speedup = t_py / max(t_cpp, 1e-9)
        print(f"{E_actual:>6d} {n_found:>6d} {t_py*1000:>10.1f} {t_cpp*1000:>10.1f} {speedup:>9.1f}x")
    print("\nINTERSECTION DETECTION TEST PASSED (cpp == py, much faster)")


if __name__ == "__main__":
    main()
