"""
constraints_v1.py

Constraint construction for solver_parametrized_half.py (v1).

Fixes:
- Corrected indentation/flow bugs in build_constraints_half_option_a (the previous
  version accidentally nested helper functions inside a loop, and returned early).
- Adds optional construction of linear inequality constraints for COD >= eps,
  to be enforced via an active-set method in KKT_v1.py.

Notes on COD>=0 inequalities:
- COD is approximated as the cumulative integral of the opening density bI along
  each polyline.
- The anchoring of COD depends on a tip (degree-1 endpoint). We therefore only
  build COD>=eps inequalities for polylines that have at least one degree-1
  endpoint. For junction-to-junction polylines, COD has an arbitrary additive
  constant (jump at the start junction), so COD>=0 is not well-posed without
  introducing extra reference DOFs; those are skipped here.

API compatibility:
- Original functions build_constraints_full, build_constraints_half_option_a,
  build_constraints_half are preserved (with bug fixes).
- New function build_cod_inequalities is added.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
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
    """
    Full-crack global gauge / neutrality constraints:
      ∑ w_k bI_k = 0
      ∑ w_k bII_k = 0
    """
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
    """
    Legacy "Option A" half-mode constraints (strict junction model).

    - For endpoints that are junctions (deg>=2): shared junction DOFs J_v.
    - For each polyline with a junction endpoint, impose:
        J_end - (J_start - ∫(bII t + bI n) ds) = 0     (vector, comp=0,1)
      whenever end is a junction with DOF.
    - Dipole neutrality / closure at multi-branch vertices:
        * deg>=3: full vector closure (2 constraints)
        * deg==2: projected closure (1 constraint) unless near-collinear
    - Tip closure:
        Enforce J_tip = 0 for *every* degree-1 endpoint that appears as a polyline end.
        This generalizes the prior "single-tip gauge" and eliminates singular KKT
        systems for isolated components (tip-to-tip branches).
    """
    if not (0.0 < float(theta_min_flat) < math.pi):
        theta_min_flat = 5.0 * math.pi / 180.0

    Crows: List[np.ndarray] = []

    # Incident branch list at each vertex (de-duplicated).
    v_inc: dict[int, List[Tuple[int, str]]] = {}
    _seen: dict[int, set[Tuple[int, str]]] = {}

    for pid, pp in enumerate(poly_panels):
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        ks = (int(pid), "start")
        ke = (int(pid), "end")

        if vs not in _seen:
            _seen[vs] = set()
        if ks not in _seen[vs]:
            _seen[vs].add(ks)
            v_inc.setdefault(vs, []).append(ks)

        if ve not in _seen:
            _seen[ve] = set()
        if ke not in _seen[ve]:
            _seen[ve].add(ke)
            v_inc.setdefault(ve, []).append(ke)

    def _junction_row(vertex: int, comp: int) -> np.ndarray:
        r = np.zeros((nunk,), float)
        vj = int(vertex)
        if vj in junction_dof:
            joff = int(junction_dof[vj])
            r[joff + 0] = 1.0 if comp == 0 else 0.0
            r[joff + 1] = 1.0 if comp == 1 else 0.0
        return r

    def row_for_integral(pid: int, comp: int) -> np.ndarray:
        pp = poly_panels[int(pid)]
        off = offsets[int(pid)]
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
        """
        Return a row representing J(which) for polyline pid, component comp.

        If the endpoint vertex has a junction DOF, it's that DOF row.
        Otherwise, we represent J_end = J_start - integral, and J_start = junction_row(v_start).
        """
        pp = poly_panels[int(pid)]
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        which = str(which).lower().strip()
        if which == "start":
            return _junction_row(vs, comp)
        if which == "end":
            # If end is a junction with DOF: use it.
            if int(deg.get(ve, 0)) >= 2 and ve in junction_dof:
                return _junction_row(ve, comp)
            # Else: J_end = J_start - integral
            return _junction_row(vs, comp) - row_for_integral(pid, comp)
        raise ValueError("which must be 'start' or 'end'")

    # Endpoint compatibility when end is a junction: J_end - (J_start - integral) = 0
    for pid, pp in enumerate(poly_panels):
        ve = int(pp["v_end"])
        if int(deg.get(ve, 0)) >= 2 and ve in junction_dof:
            for comp in (0, 1):
                rr = row_for_J_at(pid, "end", comp) - (row_for_J_at(pid, "start", comp) - row_for_integral(pid, comp))
                Crows.append(rr)

    # Closure constraints at vertices
    def outgoing_tangent(pid: int, which: str) -> np.ndarray:
        pp = poly_panels[int(pid)]
        tmid = np.asarray(pp["t_mid"], float)
        if tmid.size == 0:
            return np.array([1.0, 0.0], float)
        which = str(which).lower().strip()
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
        items = list(dict.fromkeys(items))  # de-dup preserve order
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

    # Tip closure: enforce J_tip = 0 for every degree-1 endpoint encountered
    for pid, pp in enumerate(poly_panels):
        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        if int(deg.get(vs, 0)) == 1:
            for comp in (0, 1):
                rr = row_for_J_at(pid, "start", comp)
                if np.any(rr):
                    Crows.append(rr)
        if int(deg.get(ve, 0)) == 1:
            for comp in (0, 1):
                rr = row_for_J_at(pid, "end", comp)
                # For end tips, rr will generally be nonzero (junction - integral) or (-integral)
                if np.any(rr):
                    Crows.append(rr)
                else:
                    # In the rare case this is all zeros (no junction DOF and zero integral row),
                    # skip; it would be redundant anyway.
                    pass

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
        pp = poly_panels[int(pid)]
        off = offsets[int(pid)]
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
            (_, off0) = next(iter(branch_end_dof.items()))
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


def build_cod_inequalities(
    *,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    deg: dict[int, int],
    crack_mode: str,
    representation: str,
    eps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Build linear inequalities enforcing COD >= eps at a set of sample locations.

    We approximate COD(s) as a cumulative integral of the opening density bI:
      COD(s_j) ≈ sum_k bI_k * ds_k   (forward or backward cumulative)

    Anchoring:
    - If a polyline has an end tip (deg(v_end)==1): we set COD=0 at the end tip and
      enforce COD>=eps using a *backward* cumulative from panel k to the tip.
    - Else if it has a start tip (deg(v_start)==1): we set COD=0 at the start tip and
      enforce COD>=eps using a *forward* cumulative from the start.
    - Else: skipped (junction-to-junction has an unknown additive constant).

    representation:
    - If 'singular', bI = bI_hat * sing_mid (consistent with parametrization.py post-processing),
      so the cumulative weights include sing_mid.

    Returns
    -------
    G, h, meta for G q >= h
    """
    crack_mode = str(crack_mode).lower().strip()
    representation = str(representation).lower().strip()
    use_tip_singular = (representation == "singular")
    eps = float(eps)

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    meta_rows = []

    for pid, pp in enumerate(poly_panels):
        off = offsets[int(pid)]
        Np = int(pp["Np"])
        ds = np.asarray(pp["ds"], float).reshape(-1)

        vs = int(pp["v_start"]); ve = int(pp["v_end"])
        d_vs = int(deg.get(vs, 0))
        d_ve = int(deg.get(ve, 0))

        # singular factor at midpoints (matches parametrization.py logic)
        if use_tip_singular:
            s_mid = np.asarray(pp["s_mid"], float).reshape(-1)
            Lp = float(pp["L"])
            eps_s = 1e-14 * Lp if Lp > 0 else 1e-14
            if crack_mode == "full":
                sing_mid = 1.0 / np.sqrt(np.maximum(s_mid, eps_s) * np.maximum(Lp - s_mid, eps_s))
            else:
                sing_mid = 1.0 / np.sqrt(np.maximum(Lp - s_mid, eps_s))
        else:
            sing_mid = np.ones((Np,), float)

        w = sing_mid * ds  # weights for bI_hat -> COD increment

        if d_ve == 1:
            # Backward cumulative from panel k to end tip (COD(L)=0)
            cum = np.cumsum(w[::-1])[::-1]  # cum[k] = sum_{j=k..Np-1} w[j]
            for k in range(Np):
                r = np.zeros((nunk,), float)
                # COD at midpoint k uses panels k..end
                for j in range(k, Np):
                    r[off + 2 * j + 0] += float(w[j])
                rows.append(r)
                rhs.append(eps)
                meta_rows.append(("pid", int(pid), "anchor", "end_tip", "k", int(k), "cum", float(cum[k])))
        elif d_vs == 1:
            # Forward cumulative from start tip (COD(0)=0)
            cum = np.cumsum(w)
            for k in range(Np):
                r = np.zeros((nunk,), float)
                for j in range(0, k + 1):
                    r[off + 2 * j + 0] += float(w[j])
                rows.append(r)
                rhs.append(eps)
                meta_rows.append(("pid", int(pid), "anchor", "start_tip", "k", int(k), "cum", float(cum[k])))
        else:
            # No tip anchor: skip
            continue

    G = np.vstack(rows) if rows else np.zeros((0, nunk), float)
    h = np.asarray(rhs, float) if rhs else np.zeros((0,), float)
    meta = {
        "eps": eps,
        "n_ineq": int(G.shape[0]),
        "note": "COD inequalities built only for polylines with a degree-1 endpoint.",
    }
    return G, h, meta
