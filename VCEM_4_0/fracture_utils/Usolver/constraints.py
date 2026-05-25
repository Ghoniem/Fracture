"""
Helper module: KKT constraint construction for solver_parametrized_half.py
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import math

from .network import CrackNetworkV4


def _row_for_integral(
    pid: int,
    comp: int,
    *,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
) -> np.ndarray:
    """Linear functional row implementing the integral of the jump density along polyline pid.

    Used identically by build_constraints_half_option_a and build_constraints_half;
    previously copy-pasted in two nested closures.
    """
    pp = poly_panels[int(pid)]
    off = int(offsets[int(pid)])
    Np = int(pp["Np"])
    ds = np.asarray(pp["ds"], float).reshape(-1)
    tmid = np.asarray(pp["t_mid"], float)
    nmid = np.asarray(pp["n_mid"], float)
    r = np.zeros((nunk,), float)
    for k in range(Np):
        w = float(ds[k])
        tx, ty = float(tmid[k, 0]), float(tmid[k, 1])
        nx, ny = float(nmid[k, 0]), float(nmid[k, 1])
        r[off + 2 * k + 0] += (nx if comp == 0 else ny) * w  # bI
        r[off + 2 * k + 1] += (tx if comp == 0 else ty) * w  # bII
    return r


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
    """Legacy/strict junction constraints for half-crack mode (Option A).

    Hard constraints implemented:

    1) Along-branch endpoint compatibility (when end is a junction):
         J_end - (J_start - ∫(bII t + bI n) ds) = 0     (x and y components)

       With shared junction DOFs J_v at degree>=2 vertices.

    2) Dipole neutrality / closure at junction vertices:
       - degree >= 3: full vector closure (2 constraints)
       - degree == 2: projected closure along bisector normal (1 constraint),
                      skipped for near-collinear tangents.

    3) Gauge: fix J at one degree-1 tip (first encountered), 2 constraints.

    Notes
    -----
    This function assumes piecewise-constant densities per panel and uses the
    panel-length weights ds for line integrals (consistent with the operator
    assembly in build_v4.py).
    """
    import math

    if not (0.0 < float(theta_min_flat) < math.pi):
        theta_min_flat = 5.0 * math.pi / 180.0

    Crows: List[np.ndarray] = []

    # ------------------------------------------------------------
    # Incident branch list at each vertex (de-duplicated)
    # ------------------------------------------------------------
    v_inc: dict[int, List[Tuple[int, str]]] = {}
    _seen: dict[int, set[Tuple[int, str]]] = {}
    for pid, pp in enumerate(poly_panels):
        vs = int(pp["v_start"])
        ve = int(pp["v_end"])

        ks = (int(pid), "start")
        ke = (int(pid), "end")

        _seen.setdefault(vs, set())
        if ks not in _seen[vs]:
            _seen[vs].add(ks)
            v_inc.setdefault(vs, []).append(ks)

        _seen.setdefault(ve, set())
        if ke not in _seen[ve]:
            _seen[ve].add(ke)
            v_inc.setdefault(ve, []).append(ke)

    # ------------------------------------------------------------
    # Helper rows
    # ------------------------------------------------------------
    def _junction_row(vertex: int, comp: int) -> np.ndarray:
        r = np.zeros((nunk,), float)
        if int(vertex) in junction_dof:
            joff = int(junction_dof[int(vertex)])
            r[joff + comp] = 1.0
        return r

    def row_for_integral(pid: int, comp: int) -> np.ndarray:
        return _row_for_integral(pid, comp, poly_panels=poly_panels, offsets=offsets, nunk=nunk)

    def row_for_J_at(pid: int, which: str, comp: int) -> np.ndarray:
        pp = poly_panels[int(pid)]
        vs = int(pp["v_start"])
        ve = int(pp["v_end"])

        if which == "start":
            return _junction_row(vs, comp)

        # which == 'end'
        if int(deg.get(ve, 0)) >= 2 and ve in junction_dof:
            return _junction_row(ve, comp)

        # end is a tip: J_end = J_start - integral
        return _junction_row(vs, comp) - row_for_integral(pid, comp)

    # ------------------------------------------------------------
    # (1) Endpoint compatibility when end is a junction
    # ------------------------------------------------------------
    for pid, pp in enumerate(poly_panels):
        ve = int(pp["v_end"])
        if int(deg.get(ve, 0)) >= 2 and ve in junction_dof:
            for comp in (0, 1):
                # J_end - (J_start - integral) = 0
                rr = row_for_J_at(pid, "end", comp) - (_junction_row(int(pp["v_start"]), comp) - row_for_integral(pid, comp))
                Crows.append(rr)

    # ------------------------------------------------------------
    # Closure constraints at junction vertices
    # ------------------------------------------------------------
    def outgoing_tangent(pid: int, which: str) -> np.ndarray:
        pp = poly_panels[int(pid)]
        tmid = np.asarray(pp["t_mid"], float)
        if tmid.size == 0:
            return np.array([1.0, 0.0], float)
        # For incidence bookkeeping, define the tangent "outgoing" from the vertex:
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
            # Skip near-collinear "kinks" (flat turns)
            if theta < theta_min_flat or abs(theta - math.pi) < theta_min_flat:
                continue
            b = t0 + t1
            nb = float(np.hypot(b[0], b[1]))
            if nb <= 0:
                continue
            b = b / nb
            proj = np.array([-b[1], b[0]], float)  # normal to bisector
            add_projected_closure(items, proj)

    # ------------------------------------------------------------
    # (3) Global gauge: fix J at one degree-1 tip (first found)
    # ------------------------------------------------------------
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


def build_constraints_half(
    *,
    network: CrackNetworkV4,
    deg: dict[int, int],
    poly_panels: List[dict],
    offsets: List[int],
    junction_dof: Dict[int, int],
    branch_end_dof: Dict[Tuple[int, str], int],
    nunk: int,
    junction_model: str = "strict",
    theta_min_flat: float = 5.0 * math.pi / 180.0,
    soft_eta: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Unified half-mode constraints supporting junction models: strict/core/soft.

    Returns
    -------
    C : hard constraints for KKT solve
    P : penalty operator for soft model (to be added as extra LS equations), else None
    """
    jm = str(junction_model).lower().strip()
    if jm not in ("strict", "core", "soft"):
        jm = "strict"

    if jm == "strict":
        C = build_constraints_half_option_a(
            network=network,
            deg=deg,
            poly_panels=poly_panels,
            offsets=offsets,
            junction_dof=junction_dof,
            nunk=nunk,
            theta_min_flat=theta_min_flat,
        )
        return C, None

    # ---------------- core/soft models ----------------
    # Hard constraints:
    # (1) If both endpoints are junctions: J_end - (J_start - integral)=0
    # (2) Tip closure: if endpoint is degree-1 tip, enforce J_tip=0
    #     With your orientation policy (junction at start, tip at end):
    #        J_end = J_start - integral = 0  -> J_start - integral = 0

    Crows: List[np.ndarray] = []

    def row_for_integral(pid: int, comp: int) -> np.ndarray:
        return _row_for_integral(pid, comp, poly_panels=poly_panels, offsets=offsets, nunk=nunk)

    def row_for_branch_end_J(pid: int, which: str, comp: int) -> np.ndarray:
        r = np.zeros((nunk,), float)
        key = (int(pid), str(which).lower())
        if key in branch_end_dof:
            off = int(branch_end_dof[key])
            r[off + comp] = 1.0
        return r

    # (1) junction-to-junction: compatibility
    for pid, pp in enumerate(poly_panels):
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        if int(deg.get(vs, 0)) >= 2 and int(deg.get(ve, 0)) >= 2:
            for comp in (0, 1):
                Crows.append(
                    row_for_branch_end_J(pid, "end", comp)
                    - (row_for_branch_end_J(pid, "start", comp) - row_for_integral(pid, comp))
                )

    # (2) tip closure at degree-1 endpoints
    for pid, pp in enumerate(poly_panels):
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        ds_vs = int(deg.get(vs, 0))
        ds_ve = int(deg.get(ve, 0))

        # end is free tip: J_end = J_start - integral = 0  -> J_start - integral = 0
        if ds_ve == 1:
            for comp in (0, 1):
                Crows.append(row_for_branch_end_J(pid, "start", comp) - row_for_integral(pid, comp))

        # start is free tip (rare with your orientation flip): enforce if DOF exists
        if ds_vs == 1:
            for comp in (0, 1):
                rr = row_for_branch_end_J(pid, "start", comp)
                if np.any(rr):
                    Crows.append(rr)

    # Gauge for networks with no tips: pin one branch-end DOF
    if not any(int(d) == 1 for d in deg.values()):
        if branch_end_dof:
            (k0, off0) = next(iter(branch_end_dof.items()))
            for comp in (0, 1):
                r = np.zeros((nunk,), float)
                r[int(off0) + comp] = 1.0
                Crows.append(r)

    C = np.vstack(Crows) if Crows else np.zeros((0, nunk), float)

    # Soft coupling operator P: pairwise differences of branch-end J at each junction vertex
    P = None
    if jm == "soft":
        v_inc: Dict[int, List[Tuple[int, str]]] = {}
        for pid, pp in enumerate(poly_panels):
            vs = int(pp["v_start"]); ve = int(pp["v_end"])
            if int(deg.get(vs, 0)) >= 2:
                v_inc.setdefault(vs, []).append((int(pid), "start"))
            if int(deg.get(ve, 0)) >= 2:
                v_inc.setdefault(ve, []).append((int(pid), "end"))

        ProWs: List[np.ndarray] = []
        for v, items in v_inc.items():
            items = list(dict.fromkeys(items))  # de-dup preserve order
            if len(items) < 2:
                continue
            pid0, w0 = items[0]
            for pidj, wj in items[1:]:
                for comp in (0, 1):
                    r = np.zeros((nunk,), float)
                    r += row_for_branch_end_J(pidj, wj, comp)
                    r -= row_for_branch_end_J(pid0, w0, comp)
                    ProWs.append(r)
        P = np.vstack(ProWs) if ProWs else np.zeros((0, nunk), float)

    return C, P

# ============================================================
# Inequality support: COD(s) >= 0 via active-set (converted to hard equalities)
# ============================================================

def cod_row_midpoint(
    *,
    pid: int,
    k_mid: int,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    crack_mode: str,
    junction_model: str,
    junction_dof: Dict[int, int],
    branch_end_dof: Dict[Tuple[int, str], int],
) -> np.ndarray:
    """Return a linear row r such that r @ q = COD at panel-midpoint k_mid.

    Conventions (consistent with half-mode jump reconstruction used in this module):
      J(s) = J0 - ∫_0^s ( bII(s') t(s') + bI(s') n(s') ) ds'
      COD(s) = n(s) · J(s)

    Approximation:
      - bI, bII are piecewise-constant per panel (midpoint value)
      - integral to midpoint of panel k uses:
            full panels j < k with weight ds[j]
            half of panel k with weight 0.5*ds[k]
      - n(s) is taken as the panel midpoint normal n_mid[k]
    """
    pid = int(pid)
    k_mid = int(k_mid)

    pp = poly_panels[pid]
    off = int(offsets[pid])
    Np = int(pp["Np"])
    if not (0 <= k_mid < Np):
        raise IndexError(f"k_mid out of range: {k_mid} not in [0,{Np})")

    ds = np.asarray(pp["ds"], float).reshape(-1)
    tmid = np.asarray(pp["t_mid"], float)
    nmid = np.asarray(pp["n_mid"], float)

    nk = np.array([float(nmid[k_mid, 0]), float(nmid[k_mid, 1])], float)

    r = np.zeros((int(nunk),), float)

    # --- J0 contribution
    jm = str(junction_model).lower().strip()
    if str(crack_mode).lower().strip() == "half":
        if jm == "strict":
            v0 = int(pp["v_start"])
            if v0 in junction_dof:
                joff = int(junction_dof[v0])
                r[joff + 0] += nk[0]
                r[joff + 1] += nk[1]
        else:
            key = (int(pid), "start")
            if key in branch_end_dof:
                joff = int(branch_end_dof[key])
                r[joff + 0] += nk[0]
                r[joff + 1] += nk[1]

    # --- integral contribution up to s_mid[k]
    for j in range(0, k_mid + 1):
        w = float(ds[j]) * (0.5 if j == k_mid else 1.0)

        tj = np.array([float(tmid[j, 0]), float(tmid[j, 1])], float)
        nj = np.array([float(nmid[j, 0]), float(nmid[j, 1])], float)

        # COD = nk · (J0 - Σ (bII*tj + bI*nj) * w)
        r[off + 2 * j + 0] += -w * float(np.dot(nk, nj))  # bI
        r[off + 2 * j + 1] += -w * float(np.dot(nk, tj))  # bII

    return r


def build_cod_rows(
    *,
    active_pairs: Iterable[Tuple[int, int]],
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    crack_mode: str,
    junction_model: str,
    junction_dof: Dict[int, int],
    branch_end_dof: Dict[Tuple[int, str], int],
) -> np.ndarray:
    """Stack COD midpoint rows for a set of (pid, k_mid) pairs."""
    rows: List[np.ndarray] = []
    for (pid, k) in active_pairs:
        rows.append(
            cod_row_midpoint(
                pid=int(pid),
                k_mid=int(k),
                poly_panels=poly_panels,
                offsets=offsets,
                nunk=int(nunk),
                crack_mode=str(crack_mode),
                junction_model=str(junction_model),
                junction_dof=junction_dof,
                branch_end_dof=branch_end_dof,
            )
        )
    return np.vstack(rows) if rows else np.zeros((0, int(nunk)), float)
