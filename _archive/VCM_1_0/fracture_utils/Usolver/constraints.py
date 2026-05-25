"""
Helper module: KKT constraint construction for solver_parametrized_half.py
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import math

from .network import CrackNetworkV4


def build_constraints_full(
    *,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    use_tip_singular: bool,
) -> np.ndarray:
    Crows: List[np.ndarray] = []
    for pid, pp in enumerate(poly_panels):
        off = offsets[pid]
        Np = int(pp["Np"])
        ds = np.asarray(pp["ds"], float).reshape(-1)

        rI = np.zeros((nunk,), float)
        rII = np.zeros((nunk,), float)
        for k in range(Np):
            wC = float(pp["panel_wsing"][k]) if use_tip_singular else float(ds[k])
            rI[off + 2 * k + 0] = wC
            rII[off + 2 * k + 1] = wC
        Crows.append(rI)
        Crows.append(rII)

    return np.vstack(Crows) if Crows else np.zeros((0, nunk), float)


def build_constraints_half_option_a(
    *,
    network: CrackNetworkV4,
    deg: dict[int, int],
    poly_panels: List[dict],
    offsets: List[int],
    junction_dof: Dict[int, int],
    nunk: int,
    theta_min_flat: float,
) -> np.ndarray:
    if not (0.0 < float(theta_min_flat) < math.pi):
        theta_min_flat = 5.0 * math.pi / 180.0

    Crows: List[np.ndarray] = []

    v_inc: dict[int, List[Tuple[int, str]]] = {}
    for pid, pp in enumerate(poly_panels):
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        v_inc.setdefault(vs, []).append((pid, "start"))
        v_inc.setdefault(ve, []).append((pid, "end"))

    def _junction_row(vertex: int, comp: int) -> np.ndarray:
        r = np.zeros((nunk,), float)
        if int(vertex) in junction_dof:
            joff = int(junction_dof[int(vertex)])
            r[joff + 0] = 1.0 if comp == 0 else 0.0
            r[joff + 1] = 1.0 if comp == 1 else 0.0
        return r

    def row_for_integral(pid: int, comp: int) -> np.ndarray:
        pp = poly_panels[pid]
        off = offsets[pid]
        Np = int(pp["Np"])
        ds = np.asarray(pp["ds"], float).reshape(-1)
        tmid = np.asarray(pp["t_mid"], float)
        nmid = np.asarray(pp["n_mid"], float)
        r = np.zeros((nunk,), float)
        for k in range(Np):
            w = float(ds[k])
            tx, ty = float(tmid[k, 0]), float(tmid[k, 1])
            nx, ny = float(nmid[k, 0]), float(nmid[k, 1])
            r[off + 2 * k + 0] += (nx if comp == 0 else ny) * w
            r[off + 2 * k + 1] += (tx if comp == 0 else ty) * w
        return r

    def row_for_J_at(pid: int, which: str, comp: int) -> np.ndarray:
        pp = poly_panels[pid]
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        if which == "start":
            return _junction_row(vs, comp)
        if int(deg.get(ve, 0)) >= 2 and ve in junction_dof:
            return _junction_row(ve, comp)
        return _junction_row(vs, comp) - row_for_integral(pid, comp)

    # Endpoint compatibility when end is a junction: J_end - J_start + integral = 0
    for pid, pp in enumerate(poly_panels):
        ve = int(pp["v_end"])
        if int(deg.get(ve, 0)) >= 2 and ve in junction_dof:
            for comp in (0, 1):
                rr = row_for_J_at(pid, "end", comp) - (_junction_row(int(pp["v_start"]), comp) - row_for_integral(pid, comp))
                Crows.append(rr)

    def outgoing_tangent(pid: int, which: str) -> np.ndarray:
        pp = poly_panels[pid]
        tmid = np.asarray(pp["t_mid"], float)
        if tmid.size == 0:
            return np.array([1.0, 0.0], float)
        t = tmid[0].copy() if which == "start" else (-tmid[-1].copy())
        nrm = float(np.hypot(t[0], t[1]))
        return t / nrm if nrm > 0 else np.array([1.0, 0.0], float)

    def add_vector_closure(items: List[Tuple[int, str]]):
        for comp in (0, 1):
            r = np.zeros((nunk,), float)
            for (pid_j, which_j) in items:
                sgn = +1.0 if which_j == "start" else -1.0
                r += sgn * row_for_integral(pid_j, comp)
            Crows.append(r)

    def add_projected_closure(items: List[Tuple[int, str]], proj: np.ndarray):
        proj = np.asarray(proj, float).reshape(2,)
        r = np.zeros((nunk,), float)
        for (pid_j, which_j) in items:
            sgn = +1.0 if which_j == "start" else -1.0
            r += sgn * (proj[0] * row_for_integral(pid_j, 0) + proj[1] * row_for_integral(pid_j, 1))
        Crows.append(r)

    for v, items in v_inc.items():
        dv = int(deg.get(int(v), 0))
        if dv < 2 or len(items) < 2:
            continue
        if dv >= 3:
            add_vector_closure(items)
        elif dv == 2 and len(items) == 2:
            (pid0, w0), (pid1, w1) = items[0], items[1]
            t0 = outgoing_tangent(pid0, w0)
            t1 = outgoing_tangent(pid1, w1)
            c = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
            theta = float(np.arccos(c))
            if theta < theta_min_flat or abs(theta - math.pi) < theta_min_flat:
                continue
            b = t0 + t1
            nb = float(np.hypot(b[0], b[1]))
            if nb <= 0:
                continue
            b = b / nb
            proj = np.array([-b[1], b[0]], float)
            add_projected_closure(items, proj)

    # Global gauge: fix J at ONE degree-1 tip (first found)
    ref_tip = None
    for v_tip, d in deg.items():
        if int(d) == 1:
            ref_tip = int(v_tip)
            break
    if ref_tip is not None and ref_tip in v_inc:
        pid_t, which_t = v_inc[ref_tip][0]
        for comp in (0, 1):
            Crows.append(row_for_J_at(pid_t, which_t, comp))

    return np.vstack(Crows) if Crows else np.zeros((0, nunk), float)
