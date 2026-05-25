"""
VCM / DCE diagnostics utilities.

Save as: fracture_utils/Uprocessor/diagnostics.py

Usage (Jupyter):
    from fracture_utils.Uprocessor.diagnostics import (
        junction_constraint_diagnostics,
        tip_jump_diagnostics,
        tip_jump_from_reconstruction,
        tip_jump_from_reconstruction_multi,
        cod_csd_at_junction,
    )

Design notes
------------
- Incidence lists are built from polyline endpoints (v_start/v_end) and are
  de-duplicated. This is essential for "branch as chain of segments".
- Burgers-content (B) is NOT expected to sum to zero at junctions in half/monopolar
  mode unless you explicitly enforce such constraints. We still report it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np

# vertex_degrees_from_network used to live here as a third copy of the same
# routine. Use the canonical Usolver implementation, aliased to keep the old
# name for any out-of-tree consumers.
from fracture_utils.Usolver.build_geometry import vertex_degrees as vertex_degrees_from_network  # noqa: F401


def build_vertex_incidence_from_solution(sol: dict) -> Dict[int, List[Tuple[int, str]]]:
    """Build vertex -> incident (pid, endflag) list from polyline endpoints, de-duplicated.

    IMPORTANT:
    Do NOT iterate edge->pid mapping to build incidence for chained branches.
    Use polyline endpoints so each branch is counted once at a vertex.
    """
    polys = sol.get("parametrized_polylines", None)
    if polys is None:
        raise KeyError("Solution dict missing 'parametrized_polylines'.")

    v_inc: Dict[int, List[Tuple[int, str]]] = {}
    seen: Dict[int, set[Tuple[int, str]]] = {}

    for pid, pmeta in enumerate(polys):
        pid = int(pid)

        vids = pmeta.get("path_vertex_ids", None)
        if vids is None:
            raise KeyError("Polyline metadata missing 'path_vertex_ids'.")

        vs = int(pmeta.get("v_start", vids[0]))
        ve = int(pmeta.get("v_end", vids[-1]))

        ks = (pid, "start")
        ke = (pid, "end")

        if vs not in seen:
            seen[vs] = set()
        if ks not in seen[vs]:
            seen[vs].add(ks)
            v_inc.setdefault(vs, []).append(ks)

        if ve not in seen:
            seen[ve] = set()
        if ke not in seen[ve]:
            seen[ve].add(ke)
            v_inc.setdefault(ve, []).append(ke)

    return v_inc


def _poly_solution_arrays(solps: list, pid: int):
    """Extract arrays from sol['polyline_solutions'][pid] with robust keys."""
    s = solps[int(pid)]

    bI  = np.asarray(s["bI"],  float).reshape(-1)
    bII = np.asarray(s["bII"], float).reshape(-1)
    ds  = np.asarray(s["ds"],  float).reshape(-1)

    t = s.get("t_mid", s.get("t_col", None))
    n = s.get("n_mid", s.get("n_col", None))
    if t is None or n is None:
        raise KeyError("Missing t_mid/t_col or n_mid/n_col in polyline_solutions.")

    t = np.asarray(t, float)
    n = np.asarray(n, float)

    return bI, bII, ds, t, n, s


def outgoing_tangent_from_solution(sol: dict, pid: int, which: str) -> np.ndarray:
    """Outgoing unit tangent for a branch at its endpoint."""
    solps = sol["polyline_solutions"]
    _, _, _, t, _, _ = _poly_solution_arrays(solps, pid)

    which = str(which).lower()
    v = (+t[0]) if which == "start" else (-t[-1])

    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 0 else v


def B_of_branch(sol: dict, pid: int) -> np.ndarray:
    """Integrated Burgers-content vector for a branch/polyline."""
    solps = sol["polyline_solutions"]
    bI, bII, ds, t, n, _ = _poly_solution_arrays(solps, pid)

    B = np.zeros(2, float)
    for k in range(ds.size):
        B += (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])
    return B


def J_at_vertex(sol: dict, pid: int, which: str) -> np.ndarray:
    """Geometric jump J at the polyline endpoint (start/end).

    Convention:
      - solps[pid]["J0"] is taken as the jump at the 'start' endpoint.
      - Jump at 'end' is J0 minus integrated dB along the polyline.
    """
    solps = sol["polyline_solutions"]
    bI, bII, ds, t, n, s = _poly_solution_arrays(solps, pid)

    J0 = np.asarray(s.get("J0", [0.0, 0.0]), float).reshape(2,)
    J = J0.copy()

    which = str(which).lower()
    if which == "end":
        for k in range(ds.size):
            J -= (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])
    return J


# -------------------------
# High-level diagnostics
# -------------------------

@dataclass
class JunctionDiagResult:
    vertex: int
    degree: int
    incident: List[Tuple[int, str]]
    J_values: List[np.ndarray]
    B_values: List[np.ndarray]
    B_sum: np.ndarray


def junction_constraint_diagnostics(
    *,
    calc,
    sol: dict,
    only_vertices: Optional[Iterable[int]] = None,
    min_degree: int = 2,
    report_projection: bool = True,
    print_output: bool = True,
) -> List[JunctionDiagResult]:
    """Print and return junction diagnostics.

    Parameters
    ----------
    calc : object with .network
    sol : solution dict
    only_vertices : optional subset of vertex ids
    min_degree : 2 prints kinked nodes and junctions; 3 restricts to true junctions
    report_projection : prints p·sum(B) where p is perpendicular to sum outgoing tangents
    print_output : prints diagnostics

    Returns
    -------
    list[JunctionDiagResult]
    """
    if print_output:
        print("\n=== JUNCTION CONSTRAINT DIAGNOSTICS (HALF CRACK) ===")

    deg = vertex_degrees_from_network(calc.network)
    v_inc = build_vertex_incidence_from_solution(sol)

    if only_vertices is not None:
        only_vertices = set(int(v) for v in only_vertices)

    results: List[JunctionDiagResult] = []

    for v, items in v_inc.items():
        v = int(v)
        dv = int(deg.get(v, 0))
        if dv < int(min_degree):
            continue
        if only_vertices is not None and v not in only_vertices:
            continue

        # defensive de-dup at reporting stage too
        seen = set()
        items_u: List[Tuple[int, str]] = []
        for it in items:
            it = (int(it[0]), str(it[1]))
            if it in seen:
                continue
            seen.add(it)
            items_u.append(it)

        Js = [J_at_vertex(sol, pid, which) for (pid, which) in items_u]
        Bs = [B_of_branch(sol, pid) for (pid, _) in items_u]
        Bsum = np.sum(Bs, axis=0) if Bs else np.zeros(2, float)

        if print_output:
            print(f"\n--- Vertex v={v} (degree={dv}) ---")
            print("Incident branches (pid, end):", items_u)

            for (pid, which), Jv in zip(items_u, Js):
                print(f"  pid {pid:2d}  J(v) [{which}] = {Jv}")
            if Js:
                Jref = Js[0]
                for j, Jv in enumerate(Js[1:], start=1):
                    print(f"  |J[{j}] - J[0]| = {np.linalg.norm(Jv - Jref):.3e}")

            for (pid, _), Bj in zip(items_u, Bs):
                print(f"  pid {pid:2d}  B = {Bj}")
            print("  Sum B =", Bsum, " |norm| =", float(np.linalg.norm(Bsum)))

            if report_projection and dv >= 3:
                g = np.zeros(2, float)
                for (pid, which) in items_u:
                    g += outgoing_tangent_from_solution(sol, pid, which)
                ng = float(np.linalg.norm(g))
                if ng > 0:
                    g /= ng
                    p = np.array([-g[1], g[0]], float)
                    r = float(p @ Bsum)
                    print("  Projection direction p =", p)
                    print("  p · Sum B =", r)
                else:
                    print("  WARNING: could not form projection direction (||g||=0)")

        results.append(JunctionDiagResult(
            vertex=v,
            degree=dv,
            incident=items_u,
            J_values=Js,
            B_values=Bs,
            B_sum=Bsum,
        ))

    if print_output:
        print("\n=== END JUNCTION DIAGNOSTICS ===")

    return results


def tip_jump_diagnostics(*, calc, sol: dict, print_output: bool = True) -> None:
    deg = vertex_degrees_from_network(calc.network)
    polys = sol["parametrized_polylines"]

    if print_output:
        print("\n--- Tip jump check (polyline endpoints) ---")

    for pid, meta in enumerate(polys):
        vids = meta["path_vertex_ids"]
        v_start, v_end = int(vids[0]), int(vids[-1])

        # J vectors
        J_start = J_at_vertex(sol, pid, "start")
        J_end   = J_at_vertex(sol, pid, "end")

        # COD / CSD
        CODs, CSDs, CODe, CSDe = cod_csd_at_polyline_endpoints(sol, pid)

        if print_output:
            print(f"pid {pid}: v_start={v_start} (deg={deg.get(v_start,0)}), "
                  f"v_end={v_end} (deg={deg.get(v_end,0)})")
            print(f"   J(start)={J_start},  J(end)={J_end}")
            print(f"   COD_start={CODs:+.3e}  CSD_start={CSDs:+.3e}")
            print(f"   COD_end  ={CODe:+.3e}  CSD_end  ={CSDe:+.3e}")



def cod_csd_at_junction(
    *,
    sol: dict,
    vertex_jump: np.ndarray,
    pid_end_list: List[Tuple[int, str]],
    print_output: bool = True,
):
    """Compute COD/CSD at a junction for each incident branch given a common J(v)."""
    Jv = np.asarray(vertex_jump, float).reshape(2,)
    out = []
    for (pid, which) in pid_end_list:
        t = outgoing_tangent_from_solution(sol, pid, which)
        n = np.array([-t[1], t[0]], float)
        COD = float(Jv @ n)
        CSD = float(Jv @ t)
        out.append((int(pid), COD, CSD))
        if print_output:
            print(f"pid {pid}: COD(v)={COD:.3e}  CSD(v)={CSD:.3e}")
    return out


def tip_jump_from_reconstruction(res, edge_index: int):
    """Compute tip COD/CSD/J at the tip from reconstruction arrays for a given edge_index."""
    x, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(
        edge_index=edge_index,
        enforce_global_tip_zero=True,
    )
    ex = np.asarray(extra["ex"], float).reshape(2,)
    ey = np.asarray(extra["ey"], float).reshape(2,)
    COD_tip = float(COD[-1])
    CSD_tip = float(CSD[-1])
    J_tip = ex * CSD_tip + ey * COD_tip
    return COD_tip, CSD_tip, J_tip


def tip_jump_from_reconstruction_multi(res, edge_indices: Iterable[int], print_output: bool = True):
    """Convenience wrapper to print tip COD/CSD/J for multiple edges."""
    for ei in edge_indices:
        COD_tip, CSD_tip, J_tip = tip_jump_from_reconstruction(res, int(ei))
        if print_output:
            print(f"edge {int(ei)}: COD_tip={COD_tip:+.3e}  CSD_tip={CSD_tip:+.3e}  J_tip={J_tip}")

def cod_csd_at_polyline_endpoints(sol: dict, pid: int):
    """Return COD/CSD at start and end of a polyline using outgoing endpoint frames.

    COD = J · n,  CSD = J · t
    where:
      t_start = +t[0]
      t_end   = -t[-1]   (outgoing at end)
      n is CCW rotation of t: n = (-t_y, t_x)
    """
    solps = sol["polyline_solutions"]
    bI, bII, ds, t, n, s = _poly_solution_arrays(solps, pid)

    # Endpoint jumps
    J_start = J_at_vertex(sol, pid, "start")
    J_end   = J_at_vertex(sol, pid, "end")

    # Outgoing tangents
    t_start = np.asarray(t[0], float).reshape(2,)
    t_end   = -np.asarray(t[-1], float).reshape(2,)

    # Unit normalize (defensive)
    ns = float(np.linalg.norm(t_start))
    ne = float(np.linalg.norm(t_end))
    if ns > 0: t_start = t_start / ns
    if ne > 0: t_end   = t_end   / ne

    # CCW normals
    n_start = np.array([-t_start[1], t_start[0]], float)
    n_end   = np.array([-t_end[1],   t_end[0]],   float)

    COD_start = float(J_start @ n_start)
    CSD_start = float(J_start @ t_start)
    COD_end   = float(J_end   @ n_end)
    CSD_end   = float(J_end   @ t_end)

    return COD_start, CSD_start, COD_end, CSD_end

def jump_vector_diagnostics(
    *,
    calc,
    sol: dict,
    min_degree: int = 1,
    only_vertices: Optional[Iterable[int]] = None,
    print_output: bool = True,
):
    """
    Print jump vector, COD, and CSD diagnostics at vertices.

    For each vertex with degree >= min_degree:
      - iterate over incident branches (pid, start/end)
      - print J(v), COD(v), CSD(v) using outgoing endpoint frame
    """

    deg = vertex_degrees_from_network(calc.network)
    v_inc = build_vertex_incidence_from_solution(sol)

    if only_vertices is not None:
        only_vertices = set(int(v) for v in only_vertices)

    if print_output:
        print("\n===== JUMP VECTOR DIAGNOSTICS =====")
    
    keys = sorted(v_inc.keys())
    if print_output:
        print(f"\n[JumpDiag] vertices in incidence map: {len(keys)}")
        print(f"[JumpDiag] printing vertices with degree >= {min_degree}"
            + ("" if only_vertices is None else f" and restricted to {sorted(list(only_vertices))}"))

    for v, items in v_inc.items():
        v = int(v)
        dv = int(deg.get(v, 0))
        if dv < int(min_degree):
            continue
        if only_vertices is not None and v not in only_vertices:
            continue

        # de-duplicate defensively
        seen = set()
        items_u = []
        for (pid, which) in items:
            key = (int(pid), str(which))
            if key in seen:
                continue
            seen.add(key)
            items_u.append(key)

        if print_output:
            print(f"\n--- Vertex v={v} (degree={dv}) ---")

        for (pid, which) in items_u:
            # jump vector
            Jv = J_at_vertex(sol, pid, which)

            # outgoing frame
            t = outgoing_tangent_from_solution(sol, pid, which)
            n = np.array([-t[1], t[0]], float)

            COD = float(Jv @ n)
            CSD = float(Jv @ t)

            if print_output:
                print(
                    f"  pid {pid:2d} [{which:5s}] : "
                    f"J = [{Jv[0]:+.3e}, {Jv[1]:+.3e}]   "
                    f"COD = {COD:+.3e}   CSD = {CSD:+.3e}"
                )

    if print_output:
        print("\n===== END OF JUMP VECTOR DIAGNOSTICS =====")

