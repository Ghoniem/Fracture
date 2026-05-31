"""Sub-critical isolated-crack deactivation for the KKT solve.

A polyline is *isolated* when both of its endpoint vertices are degree-1 free
tips (the polyline is a free-standing crack, not connected to any network).
For such polylines we estimate the tip stress intensity from the closed-form
isolated-infinite-medium formula

    K_I  ~ sigma_n * sqrt(pi * a),
    K_II ~ |sigma_t| * sqrt(pi * a),    a = L / 2,

evaluated using the local applied stress sampled at the polyline center
and both tip vertices, taking the location of maximum
K_eff = hypot(K_I, K_II). If K_eff falls below a multiple of the local
fracture toughness, the polyline is *deactivated*: its panel DOFs and
collocation rows are removed from the KKT system, and its COD is filled
in post-solve from the analytical Westergaard solution for an isolated
crack in a uniform remote field. The crack retains its geometry and its
stress-field perturbation (via the analytical COD), but it does not
participate in the global KKT coupling.

Deactivation auto-reverses without any explicit state machine: when the
polyline links into another crack (one endpoint becomes degree >= 2),
the isolation predicate fails and the polyline is back in the system on
the next solve.

Hysteresis is supported when ``safety`` is given as a ``(lo, hi)`` tuple
with ``lo < hi``: an active polyline stays active until K_eff drops below
``lo * K_Ic``; an inactive polyline reactivates only when K_eff rises
above ``hi * K_Ic``. Caller threads the previous-cycle state via
``prev_state``, a dict keyed by the unordered endpoint pair
``frozenset({v_start, v_end})`` (stable across cycles for polylines whose
topology does not change).

Connected polylines (at least one endpoint of degree >= 2) are always
active; the existing per-panel ``min_panel_length_ratio`` merge guard in
``discretize_polylines`` handles their conditioning.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np

from .material import AppliedStress, Material


KIcLike = Union[float, Callable[[float, float], float]]
SafetyLike = Union[float, Tuple[float, float]]


def _resolve_kic(K_Ic: KIcLike) -> Callable[[float, float], float]:
    """Coerce K_Ic to a callable f(x, y) -> float.

    Accepts a scalar, a plain callable f(x, y), or an object exposing
    ``Kc(x, y)`` (the ``ToughnessField`` protocol from Upropagation).
    """
    if hasattr(K_Ic, "Kc"):
        Kc_method = getattr(K_Ic, "Kc")
        return lambda x, y: float(Kc_method(float(x), float(y)))
    if callable(K_Ic):
        return lambda x, y: float(K_Ic(float(x), float(y)))
    K_value = float(K_Ic)
    return lambda x, y: K_value


def _resolve_safety(safety: SafetyLike) -> Tuple[float, float]:
    """Coerce safety to (lo, hi) with 0 < lo <= hi.

    Scalar -> (s, s) (no hysteresis gap).
    """
    if isinstance(safety, (tuple, list)) and len(safety) == 2:
        lo, hi = float(safety[0]), float(safety[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= 0.0:
            return (1.0, 1.0)
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)
    s = float(safety)
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    return (s, s)


def _polyline_endpoints_xy(pp: dict, network: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_start, x_end) tip positions.

    Uses network vertex coordinates when available (exact tip position);
    falls back to first/last panel-midpoint when not.
    """
    x_mid = np.asarray(pp["x_mid"], float)
    if network is not None:
        v_start = int(pp["v_start"])
        v_end = int(pp["v_end"])
        x0 = x1 = None
        for v in getattr(network, "vertices", []):
            if int(v.id) == v_start:
                x0 = np.array([float(v.x), float(v.y)], float)
            elif int(v.id) == v_end:
                x1 = np.array([float(v.x), float(v.y)], float)
            if x0 is not None and x1 is not None:
                break
        if x0 is not None and x1 is not None:
            return x0, x1
    return x_mid[0].astype(float), x_mid[-1].astype(float)


def classify_active_polylines(
    *,
    poly_panels: List[dict],
    deg: Dict[int, int],
    applied: AppliedStress,
    K_Ic: Optional[KIcLike],
    safety: SafetyLike = 1.0,
    network: Any = None,
    prev_state: Optional[Dict[FrozenSet[int], bool]] = None,
) -> Tuple[List[bool], Dict[FrozenSet[int], bool]]:
    """Per-polyline activation classification.

    A polyline is *isolated* when both endpoint vertices have degree 1.
    Only isolated polylines are candidates for deactivation.

    Decision per isolated polyline:
      - K_eff = max over {center, v_start tip, v_end tip} of
                hypot(sigma_n_open * sqrt(pi*a), |sigma_t| * sqrt(pi*a))
      - K_Ic_local sampled at the location of max K_eff
      - With (safety_lo, safety_hi) = _resolve_safety(safety) and
        prev_active = prev_state.get(frozenset({v_start, v_end}), True):
          threshold = (safety_lo if prev_active else safety_hi) * K_Ic_local
          active = (K_eff >= threshold)

    Returns
    -------
    active : list of bool, one per polyline (True = participates in KKT solve)
    state  : dict[frozenset({v_start, v_end}) -> bool] for caller to thread
             back as prev_state on the next solve (enables hysteresis)
    """
    n = len(poly_panels)
    if K_Ic is None or n == 0:
        return [True] * n, {}

    Kc_at = _resolve_kic(K_Ic)
    safety_lo, safety_hi = _resolve_safety(safety)
    prev = prev_state or {}

    active: List[bool] = []
    state: Dict[FrozenSet[int], bool] = {}
    for pp in poly_panels:
        v_start = int(pp["v_start"])
        v_end = int(pp["v_end"])
        key = frozenset({v_start, v_end})
        isolated = (int(deg.get(v_start, 0)) == 1
                    and int(deg.get(v_end, 0)) == 1)
        if not isolated:
            active.append(True)
            state[key] = True
            continue

        L = float(pp.get("L", 0.0))
        x_mid = np.asarray(pp["x_mid"], float)
        n_mid = np.asarray(pp["n_mid"], float)
        if L <= 0.0 or x_mid.size == 0 or n_mid.size == 0:
            active.append(True)
            state[key] = True
            continue

        n_avg = n_mid.mean(axis=0)
        n_norm = float(np.hypot(n_avg[0], n_avg[1]))
        if n_norm <= 0.0:
            active.append(True)
            state[key] = True
            continue
        n_hat = n_avg / n_norm
        t_hat = np.array([-n_hat[1], n_hat[0]], float)

        x_tip0, x_tip1 = _polyline_endpoints_xy(pp, network)
        x_center = x_mid.mean(axis=0)
        X_sample = np.vstack([x_center[None, :], x_tip0[None, :], x_tip1[None, :]])
        sigmas = applied.tensor_at(X_sample)
        sigma_n_arr = np.einsum("i,kij,j->k", n_hat, sigmas, n_hat)
        sigma_t_arr = np.einsum("i,kij,j->k", t_hat, sigmas, n_hat)
        sigma_n_open = np.maximum(sigma_n_arr, 0.0)

        a = 0.5 * L
        root_pi_a = float(np.sqrt(np.pi * a))
        KI_arr = sigma_n_open * root_pi_a
        KII_arr = np.abs(sigma_t_arr) * root_pi_a
        K_eff_arr = np.hypot(KI_arr, KII_arr)
        i_max = int(np.argmax(K_eff_arr))
        K_eff_max = float(K_eff_arr[i_max])
        x_eval = X_sample[i_max]
        K_Ic_local = float(Kc_at(float(x_eval[0]), float(x_eval[1])))

        prev_active = bool(prev.get(key, True))
        s_factor = safety_lo if prev_active else safety_hi
        threshold = s_factor * K_Ic_local

        is_active = bool(K_eff_max >= threshold)
        active.append(is_active)
        state[key] = is_active

    return active, state


def build_dof_masks(
    *,
    poly_panels: List[dict],
    offsets: List[int],
    nunk: int,
    active_polylines: List[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (row_mask, col_mask) for filtering the assembled crack system.

    Returns
    -------
    row_mask : (2 * ncol_tot,) bool
        Collocation-row selector. Inactive polylines' tn AND ts rows are
        dropped. Row layout is [tn_block, ts_block] per assemble_operator.
    col_mask : (nunk,) bool
        DOF selector. Inactive polylines' 2*Np panel DOFs are dropped.
        Junction / branch-end DOFs are always kept (they only exist at
        vertices of degree >= 2, which by definition are not on isolated
        polylines).
    """
    col_mask = np.ones(int(nunk), bool)
    row_pieces: List[np.ndarray] = []
    for pid, pp in enumerate(poly_panels):
        Np = int(pp["Np"])
        if active_polylines[pid]:
            row_pieces.append(np.ones(Np, bool))
        else:
            off = int(offsets[pid])
            col_mask[off : off + 2 * Np] = False
            row_pieces.append(np.zeros(Np, bool))
    if row_pieces:
        row_one_side = np.concatenate(row_pieces)
    else:
        row_one_side = np.zeros(0, bool)
    row_mask = np.concatenate([row_one_side, row_one_side])
    return row_mask, col_mask


def compute_isolated_analytical_densities(
    *,
    poly_panels: List[dict],
    active_polylines: List[bool],
    applied: AppliedStress,
    material: Material,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Westergaard isolated-crack densities (b_I, b_II) per inactive polyline.

    For each polyline marked inactive, returns per-panel constant
    dislocation densities consistent with the closed-form COD profile of
    an isolated crack of half-length a = L/2 under the local remote
    stress sampled at the crack center:

        COD_n(s) = (4 / E')  * sigma_n_open * sqrt(a^2 - (s - L/2)^2)
        COD_t(s) = (4 / E')  * sigma_t      * sqrt(a^2 - (s - L/2)^2)

    with E' = E / (1 - nu^2) (plane strain) or E (plane stress). The
    per-panel density used by the dislocation-density assembler is the
    derivative of COD, integrated over the panel:

        b_I[k]  = (COD_n(s_{k+1}) - COD_n(s_k)) / ds[k]
        b_II[k] = (COD_t(s_{k+1}) - COD_t(s_k)) / ds[k]

    COD vanishes at both tips so the tip-closure constraint
    ``int b ds = 0`` is satisfied identically. A closed crack
    (sigma_n <= 0) contributes zero mode I; mode II is retained
    irrespective of the normal sign (no friction model).

    Returns
    -------
    dict[int, (np.ndarray, np.ndarray)]
        Mapping pid -> (b_I, b_II), each of length pp["Np"], for
        inactive polylines only.
    """
    E = float(material.E)
    nu = float(material.nu)
    plane_stress = bool(getattr(material, "plane_stress", False))
    E_prime = E if plane_stress else E / max(1.0 - nu * nu, 1.0e-30)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for pid, pp in enumerate(poly_panels):
        if active_polylines[pid]:
            continue
        Np = int(pp["Np"])
        L = float(pp.get("L", 0.0))
        s_nodes = np.asarray(pp["s_nodes"], float)
        ds = np.asarray(pp["ds"], float)
        x_mid = np.asarray(pp["x_mid"], float)
        n_mid = np.asarray(pp["n_mid"], float)
        if L <= 0.0 or s_nodes.size < 2 or x_mid.size == 0 or n_mid.size == 0:
            out[pid] = (np.zeros(Np, float), np.zeros(Np, float))
            continue

        n_avg = n_mid.mean(axis=0)
        n_norm = float(np.hypot(n_avg[0], n_avg[1]))
        if n_norm <= 0.0:
            out[pid] = (np.zeros(Np, float), np.zeros(Np, float))
            continue
        n_hat = n_avg / n_norm
        t_hat = np.array([-n_hat[1], n_hat[0]], float)

        x_c = x_mid.mean(axis=0).reshape(1, 2)
        sig = applied.tensor_at(x_c)[0]
        sigma_n = float(n_hat @ sig @ n_hat)
        sigma_t = float(t_hat @ sig @ n_hat)
        sigma_n_open = max(sigma_n, 0.0)

        a = 0.5 * L
        xi = s_nodes - 0.5 * L
        sqrt_term = np.sqrt(np.maximum(a * a - xi * xi, 0.0))
        cod_n = (4.0 / E_prime) * sigma_n_open * sqrt_term
        cod_t = (4.0 / E_prime) * sigma_t * sqrt_term

        ds_safe = np.where(ds > 0.0, ds, 1.0)
        b_I = np.diff(cod_n) / ds_safe
        b_II = np.diff(cod_t) / ds_safe
        out[pid] = (b_I.astype(float), b_II.astype(float))

    return out


def scatter_q(
    q_active: np.ndarray,
    col_mask: Optional[np.ndarray],
    nunk: int,
    *,
    poly_panels: Optional[List[dict]] = None,
    offsets: Optional[List[int]] = None,
    analytical_densities: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> np.ndarray:
    """Scatter reduced solution to full DOF vector, optionally filling
    inactive polylines with analytical Westergaard densities.

    When ``col_mask`` is None and no analytical fill is provided, this
    is the identity (returns q_active as a fresh float array).
    """
    q_active_arr = np.asarray(q_active, float).reshape(-1)
    if col_mask is None and not analytical_densities:
        return q_active_arr.astype(float, copy=True)

    q_full = np.zeros(int(nunk), float)
    if col_mask is not None:
        q_full[col_mask] = q_active_arr
    else:
        q_full[: q_active_arr.size] = q_active_arr

    if analytical_densities and poly_panels is not None and offsets is not None:
        for pid, (b_I, b_II) in analytical_densities.items():
            off = int(offsets[pid])
            Np = int(poly_panels[pid]["Np"])
            n_take = min(Np, len(b_I), len(b_II))
            for k in range(n_take):
                q_full[off + 2 * k + 0] = float(b_I[k])
                q_full[off + 2 * k + 1] = float(b_II[k])
    return q_full


def drop_zero_rows(
    C: np.ndarray, d: Optional[np.ndarray] = None, tol: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop all-zero rows of a (possibly column-masked) constraint matrix.

    After applying ``col_mask`` to a tip-closure-style constraint, rows
    referencing only inactive polylines collapse to identically zero. Those
    rows enforce 0 = d_i which is either trivial (d_i = 0) or
    contradictory; we keep only rows with at least one nonzero coefficient.
    """
    if C is None or C.size == 0:
        return C, d
    row_norms = np.linalg.norm(C, axis=1)
    keep = row_norms > float(tol)
    Cf = C[keep]
    if d is None:
        return Cf, None
    df = np.asarray(d, float).reshape(-1)[keep]
    return Cf, df
