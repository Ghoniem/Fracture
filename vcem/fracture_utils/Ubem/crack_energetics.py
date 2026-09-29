"""Crack energetics post-processing for VCEM growth histories.

Implements the force-control relations used in the phase-transition framing:

- C(a) = delta / P
- U(a) = 0.5 * P^2 * C(a)
- Pi(a) = U(a) - W_ext(a) = -0.5 * P^2 * C(a)
- G(a) = -dPi/dA = (KI^2 + KII^2) / E'
- dC/dA = 2*G/P^2

Notes
-----
- `P` is the effective scalar load conjugate to the chosen generalized
  displacement `delta` in your model definition.
- For plane strain: E' = E/(1-nu^2). For plane stress: E' = E.
- For a through crack in a plate/disk of thickness B, use A(a)=2*B*a.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EnergeticsConfig:
    E: float
    nu: float
    plane_stress: bool
    P: float
    delta: float | None = None
    thickness: float = 1.0
    fracture_surfaces: float = 2.0
    compliance_C0: float = 0.0
    fracture_energy_Gamma: float | None = None


def effective_modulus(E: float, nu: float, plane_stress: bool) -> float:
    """Return E' used in mixed-mode energy-release computations."""
    E = float(E)
    nu = float(nu)
    if bool(plane_stress):
        return E
    den = 1.0 - nu * nu
    if abs(den) < 1e-14:
        raise ValueError("Invalid Poisson ratio for plane strain (1-nu^2 ~ 0).")
    return E / den


def G_from_sif(KI_MPa_sqrt_m: np.ndarray, KII_MPa_sqrt_m: np.ndarray, *, E: float, nu: float, plane_stress: bool) -> np.ndarray:
    """Compute G [J/m^2] from KI/KII given in MPa*sqrt(m)."""
    KI = np.asarray(KI_MPa_sqrt_m, float) * 1e6
    KII = np.asarray(KII_MPa_sqrt_m, float) * 1e6
    Eeff = effective_modulus(E=E, nu=nu, plane_stress=plane_stress)
    return (KI * KI + KII * KII) / float(Eeff)


def _validate_path_arrays(path: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(path["a_m"], float)
    KI = np.asarray(path["KI_MPa_sqrt_m"], float)
    KII = np.asarray(path["KII_MPa_sqrt_m"], float)

    if a.ndim != 1 or KI.ndim != 1 or KII.ndim != 1 or not (len(a) == len(KI) == len(KII)):
        raise ValueError("path arrays a/KI/KII must be 1D with equal lengths")
    if len(a) == 0:
        raise ValueError("empty path")
    return a, KI, KII


def _incremental_drop(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) > 1:
        out[1:] = x[:-1] - x[1:]
    return out


def _disk_loaded_arc_masks(theta_deg: np.ndarray, arc_half_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    th = np.asarray(theta_deg, float).reshape(-1)
    arc = float(arc_half_angle_deg)
    top = (th >= 90.0 - arc) & (th <= 90.0 + arc)
    bot = (th >= -90.0 - arc) & (th <= -90.0 + arc)
    return top, bot


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, float).reshape(-1)
    w = np.asarray(weights, float).reshape(-1)
    ws = float(np.nansum(w))
    if ws <= 0.0:
        m = np.nanmean(v)
        return float(m) if np.isfinite(m) else 0.0
    return float(np.nansum(w * v) / ws)


def compute_disk_boundary_compliance_from_fields(
    *,
    mesh,
    ux: np.ndarray,
    uy: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    h: float,
    arc_half_angle_deg: float = 15.0,
) -> Dict[str, float]:
    """Compute bounded-disk compliance directly from boundary fields.

    Uses loaded top/bottom arcs and projects boundary displacement in the
    direction of resultant boundary force.

    Returned primary metric:
      C_rel_m_per_N = delta_rel_m / P_N
    where delta_rel_m is top-minus-bottom average projected displacement along
    the load axis, and P_N is average resultant load magnitude per platen.
    """
    n = int(mesh.n_seg)
    ux = np.asarray(ux, float).reshape(-1)
    uy = np.asarray(uy, float).reshape(-1)
    tx = np.asarray(tx, float).reshape(-1)
    ty = np.asarray(ty, float).reshape(-1)
    if not (ux.size == uy.size == tx.size == ty.size == n):
        raise ValueError("ux, uy, tx, ty must all have length mesh.n_seg.")
    if getattr(mesh, "theta_deg", None) is None:
        raise ValueError("mesh.theta_deg is required to identify top/bottom loaded arcs.")

    L = np.asarray(mesh.length, float).reshape(-1)
    top, bot = _disk_loaded_arc_masks(np.asarray(mesh.theta_deg, float), arc_half_angle_deg=float(arc_half_angle_deg))
    if not np.any(top) or not np.any(bot):
        raise ValueError("Top/bottom loaded arcs are empty. Check arc_half_angle_deg and boundary discretization.")

    # Segment force vectors [N] = traction [Pa] * length [m] * thickness [m]
    fx = tx * L * float(h)
    fy = ty * L * float(h)

    F_top = np.array([np.sum(fx[top]), np.sum(fy[top])], float)
    F_bot = np.array([np.sum(fx[bot]), np.sum(fy[bot])], float)
    P_top = float(np.linalg.norm(F_top))
    P_bot = float(np.linalg.norm(F_bot))
    P = 0.5 * (P_top + P_bot)

    # Load axis from top resultant (fallback to opposite bottom resultant).
    if P_top > 0.0:
        e_load = F_top / P_top
    elif P_bot > 0.0:
        e_load = -F_bot / P_bot
    else:
        raise ValueError("Zero resultant force on both loaded arcs; cannot compute compliance.")

    u_axis = ux * float(e_load[0]) + uy * float(e_load[1])

    # Use force-component weighting on each loaded arc to define generalized displacement.
    wf = np.abs(fx * float(e_load[0]) + fy * float(e_load[1]))
    w_top = np.where(top, wf, 0.0)
    w_bot = np.where(bot, wf, 0.0)
    if float(np.sum(w_top)) <= 0.0:
        w_top = np.where(top, L, 0.0)
    if float(np.sum(w_bot)) <= 0.0:
        w_bot = np.where(bot, L, 0.0)

    delta_top = _weighted_mean(u_axis[top], w_top[top])
    delta_bot = _weighted_mean(u_axis[bot], w_bot[bot])
    delta_rel = delta_top - delta_bot

    C_top = delta_top / P_top if P_top > 0.0 else np.nan
    C_bot = delta_bot / P_bot if P_bot > 0.0 else np.nan
    C_rel = delta_rel / P if P > 0.0 else np.nan

    return {
        "delta_top_m": float(delta_top),
        "delta_bot_m": float(delta_bot),
        "delta_rel_m": float(delta_rel),
        "P_top_N": float(P_top),
        "P_bot_N": float(P_bot),
        "P_N": float(P),
        "C_top_m_per_N": float(C_top) if np.isfinite(C_top) else np.nan,
        "C_bot_m_per_N": float(C_bot) if np.isfinite(C_bot) else np.nan,
        "C_rel_m_per_N": float(C_rel) if np.isfinite(C_rel) else np.nan,
        "load_axis_x": float(e_load[0]),
        "load_axis_y": float(e_load[1]),
    }


def compute_disk_boundary_compliance_from_solver(
    *,
    solver,
    mesh,
    h: float,
    arc_half_angle_deg: float = 15.0,
) -> Dict[str, float]:
    """Convenience wrapper using solved BEMSolver2D boundary arrays."""
    if getattr(solver, "u_x", None) is None or getattr(solver, "t_x", None) is None:
        raise RuntimeError("Solver has no boundary solution. Call solver.solve() first.")
    return compute_disk_boundary_compliance_from_fields(
        mesh=mesh,
        ux=np.asarray(solver.u_x, float),
        uy=np.asarray(solver.u_y, float),
        tx=np.asarray(solver.t_x, float),
        ty=np.asarray(solver.t_y, float),
        h=float(h),
        arc_half_angle_deg=float(arc_half_angle_deg),
    )


def aggregate_inner_cycle_series(
    tip_history: Any,
    *,
    include_nongrowing_steps: bool = True,
) -> Dict[str, np.ndarray]:
    """Aggregate per-tip history into one row per INNER cycle.

    Returns arrays aligned by inner-cycle order:
      cycle_index, outer_cycle, global_step, inner_step,
      L_total_m, delta_a_m, KI_MPa_sqrt_m, KII_MPa_sqrt_m
    """
    d = _to_arrays(tip_history)
    if not d:
        raise ValueError("tip_history is empty")

    phase = np.asarray(d.get("phase", []), dtype=str)
    if phase.size == 0:
        raise ValueError("tip_history missing 'phase'")
    n = int(phase.size)

    KI = np.asarray(d.get("KI_MPa_sqrt_m", np.full(n, np.nan)), float)
    KII = np.asarray(d.get("KII_MPa_sqrt_m", np.full(n, np.nan)), float)
    grew = np.asarray(d.get("grew", np.zeros(n, bool)), bool)
    delta_a = np.asarray(d.get("delta_a_m", np.where(grew, np.nan, 0.0)), float)
    Ltot = np.asarray(d.get("total_length_m", np.full(n, np.nan)), float)
    outer = np.asarray(d.get("outer_cycle", d.get("outer_coupling_cycle", np.full(n, -1))), int)
    global_step = np.asarray(d.get("global_step", np.arange(n)), int)
    inner_step = np.asarray(d.get("inner_step", np.full(n, -1)), int)

    m = phase == "inner_step"
    idx = np.where(m)[0]
    if idx.size == 0:
        raise ValueError("tip_history has no 'inner_step' rows")

    order = np.lexsort((inner_step[idx], global_step[idx], outer[idx]))
    idx = idx[order]
    keys = np.column_stack([outer[idx], global_step[idx], inner_step[idx]])

    rows = []
    s = 0
    cyc = 0
    while s < idx.size:
        e = s + 1
        while e < idx.size and np.all(keys[e] == keys[s]):
            e += 1
        gidx = idx[s:e]

        da = np.asarray(delta_a[gidx], float)
        da_pos = np.where(np.isfinite(da) & (da > 0.0), da, 0.0)
        da_sum = float(np.sum(da_pos))
        if da_sum <= 0.0:
            # non-growing inner step
            if not include_nongrowing_steps:
                s = e
                continue
            w = np.full((gidx.size,), 1.0 / max(gidx.size, 1), float)
            da_sum = 0.0
        else:
            w = da_pos / da_sum

        L_after = float(np.nanmax(Ltot[gidx])) if np.any(np.isfinite(Ltot[gidx])) else np.nan
        rows.append(
            dict(
                cycle_index=int(cyc),
                outer_cycle=int(outer[gidx[0]]),
                global_step=int(global_step[gidx[0]]),
                inner_step=int(inner_step[gidx[0]]),
                L_total_m=L_after,
                delta_a_m=da_sum,
                KI_MPa_sqrt_m=float(np.sum(w * KI[gidx])),
                KII_MPa_sqrt_m=float(np.sum(w * KII[gidx])),
            )
        )
        cyc += 1
        s = e

    if not rows:
        raise ValueError("No inner-cycle rows assembled from tip_history.")
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0].keys()}


def topology_metrics_for_network(
    network: Any,
    *,
    domain_area: float,
    region_1: Any,
    region_2: Any,
    loading_axis: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """Compute alpha, P_infty, Q, L_total for a single network snapshot."""
    from fracture_utils.Ugenerator.analysis import NetworkAnalyzer
    import networkx as nx

    if isinstance(network, nx.Graph):
        G = network
    else:
        # Support CrackNetworkV4-like objects with vertices/edges attributes.
        G = nx.Graph()
        for v in getattr(network, "vertices", []):
            G.add_node(int(v.id), pos=(float(v.x), float(v.y)))
        for e in getattr(network, "edges", []):
            v0 = int(e.v0)
            v1 = int(e.v1)
            if v0 in G.nodes and v1 in G.nodes:
                p0 = np.asarray(G.nodes[v0]["pos"], float)
                p1 = np.asarray(G.nodes[v1]["pos"], float)
                L = float(np.linalg.norm(p1 - p0))
                G.add_edge(v0, v1, length=L)

    an = NetworkAnalyzer(G)
    topo = an.get_topology_metrics(
        domain_area=float(domain_area),
        region_1=region_1,
        region_2=region_2,
        loading_axis=loading_axis,
        include_alignment_Q=True,
    )
    return {
        "alpha": float(topo.get("alpha", np.nan)),
        "P_infty": float(topo.get("P_infty", np.nan)),
        "Q": float(topo.get("Q", np.nan)),
        "L_total_m": float(topo.get("L_tot_m", np.nan)),
    }


def build_inner_cycle_metrics(
    *,
    tip_history: Any,
    domain_area: float,
    thickness: float,
    control_mode: str,
    P_force_N: Optional[float] = None,
    delta_disp_m: Optional[float] = None,
    per_cycle_topology: Optional[Mapping[int, Mapping[str, float]]] = None,
    per_cycle_compliance: Optional[Mapping[int, float]] = None,
) -> Dict[str, np.ndarray]:
    """Build all requested per-inner-cycle arrays for notebook plotting.

    Inputs
    ------
    tip_history:
      tip history rows/arrays containing inner-step info.
    per_cycle_topology:
      optional mapping cycle_index -> {'alpha','P_infty','Q','L_total_m'}.
    per_cycle_compliance:
      optional mapping cycle_index -> C_rel_m_per_N.
    """
    base = aggregate_inner_cycle_series(tip_history, include_nongrowing_steps=True)
    cyc = np.asarray(base["cycle_index"], int)
    n = int(cyc.size)

    L_total = np.asarray(base["L_total_m"], float).copy()
    alpha = np.full((n,), np.nan, float)
    Pinf = np.full((n,), np.nan, float)
    Q = np.full((n,), np.nan, float)
    Crel = np.full((n,), np.nan, float)

    # Defaults from L_total + area.
    if float(domain_area) > 0.0:
        alpha = L_total / float(domain_area)

    if per_cycle_topology is not None:
        for i, c in enumerate(cyc):
            if int(c) not in per_cycle_topology:
                continue
            d = per_cycle_topology[int(c)]
            alpha[i] = float(d.get("alpha", alpha[i]))
            Pinf[i] = float(d.get("P_infty", Pinf[i]))
            Q[i] = float(d.get("Q", Q[i]))
            L_total[i] = float(d.get("L_total_m", L_total[i]))

    if per_cycle_compliance is not None:
        for i, c in enumerate(cyc):
            if int(c) in per_cycle_compliance:
                Crel[i] = float(per_cycle_compliance[int(c)])

    # Area based on total crack length in bounded disk:
    # A = (fracture surfaces) * h * L_total, with fracture_surfaces=2.
    A = 2.0 * float(thickness) * L_total
    dC_dA = np.full((n,), np.nan, float)
    for i in range(1, n):
        dA = float(A[i] - A[i - 1])
        if np.isfinite(Crel[i]) and np.isfinite(Crel[i - 1]) and abs(dA) > 0.0:
            dC_dA[i] = (Crel[i] - Crel[i - 1]) / dA

    mode = str(control_mode).strip().lower()
    U = np.full((n,), np.nan, float)
    W = np.full((n,), np.nan, float)
    Phi = np.full((n,), np.nan, float)
    G = np.full((n,), np.nan, float)

    if mode == "force":
        if P_force_N is None:
            raise ValueError("P_force_N is required for force-control metrics.")
        P0 = float(P_force_N)
        U = 0.5 * (P0 ** 2) * Crel
        W = (P0 ** 2) * Crel
        Phi = U - W
        G = 0.5 * (P0 ** 2) * dC_dA
    elif mode == "displacement":
        if delta_disp_m is None:
            raise ValueError("delta_disp_m is required for displacement-control metrics.")
        d0 = float(delta_disp_m)
        # Per model definition in README:
        U = 0.5 * (d0 ** 2) / Crel
        Phi = U
        G = 0.5 * (d0 ** 2) * dC_dA / (Crel * Crel)
        W[:] = np.nan
    else:
        raise ValueError("control_mode must be 'force' or 'displacement'.")

    out = {
        "cycle_index": cyc.astype(int),
        "outer_cycle": np.asarray(base["outer_cycle"], int),
        "global_step": np.asarray(base["global_step"], int),
        "inner_step": np.asarray(base["inner_step"], int),
        "L_total_m": L_total,
        "alpha": alpha,
        "P_infty": Pinf,
        "Q": Q,
        "C_rel_m_per_N": Crel,
        "A_m2": A,
        "dC_dA_1_per_Pa": dC_dA,
        "U_J": U,
        "W_J": W,
        "Phi_J": Phi,
        "G_J_per_m2": G,
        "KI_MPa_sqrt_m": np.asarray(base["KI_MPa_sqrt_m"], float),
        "KII_MPa_sqrt_m": np.asarray(base["KII_MPa_sqrt_m"], float),
        "delta_a_m": np.asarray(base["delta_a_m"], float),
    }
    return out


def save_inner_cycle_metrics_npz(path: str | Path, metrics: Mapping[str, np.ndarray]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{k: np.asarray(v) for k, v in metrics.items()})
    return p


def _to_arrays(tip_history: Any) -> Dict[str, np.ndarray]:
    """Normalize tip-history input (dict-like or list[dict]) to array dict."""
    if isinstance(tip_history, Mapping):
        out: Dict[str, np.ndarray] = {}
        for k, v in tip_history.items():
            out[str(k)] = np.asarray(v)
        return out

    if isinstance(tip_history, Sequence):
        rows = list(tip_history)
        if not rows:
            return {}
        keys = set()
        for r in rows:
            if isinstance(r, Mapping):
                keys.update(r.keys())
        out = {}
        for k in sorted(keys):
            out[str(k)] = np.asarray([r.get(k, np.nan) for r in rows], object)
        return out

    raise TypeError("tip_history must be mapping-like or sequence of dict rows")


def aggregate_growth_path(
    tip_history: Any,
    *,
    include_initial: bool = True,
    initial_total_length_m: float | None = None,
) -> Dict[str, np.ndarray]:
    """Aggregate per-tip growth records into stepwise global crack-length history.

    Parameters
    ----------
    tip_history:
        Mapping/npz-like arrays or list of row dicts (e.g., `tip_history_all`).
    include_initial:
        Include an initial state (delta_a=0) if `phase=='initial'` records exist.
    initial_total_length_m:
        Optional fallback initial length used if history does not carry `total_length_m`.

    Returns
    -------
    dict of arrays containing:
      - step_index
      - outer_coupling_cycle
      - global_step
      - inner_step
      - a_m
      - delta_a_m
      - KI_MPa_sqrt_m
      - KII_MPa_sqrt_m
    """
    d = _to_arrays(tip_history)
    if not d:
        raise ValueError("tip_history is empty")

    phase = np.asarray(d.get("phase", []), dtype=str)
    if phase.size == 0:
        raise ValueError("tip_history missing 'phase'")
    n = int(phase.size)

    KI = np.asarray(d.get("KI_MPa_sqrt_m", np.full(n, np.nan)), float)
    KII = np.asarray(d.get("KII_MPa_sqrt_m", np.full(n, np.nan)), float)
    grew = np.asarray(d.get("grew", np.zeros(n, bool)), bool)
    delta_a = np.asarray(d.get("delta_a_m", np.where(grew, np.nan, 0.0)), float)
    total_length = np.asarray(d.get("total_length_m", np.full(n, np.nan)), float)

    outer = np.asarray(d.get("outer_coupling_cycle", d.get("outer_cycle", np.full(n, -1))), int)
    source = np.asarray(d.get("coupling_source", np.full(n, "growth")), dtype=str)
    global_step = np.asarray(d.get("global_step", np.arange(n)), int)
    inner_step = np.asarray(d.get("inner_step", np.full(n, -1)), int)

    rows = []
    if include_initial:
        m0 = phase == "initial"
        if np.any(m0):
            w0 = np.ones(int(np.count_nonzero(m0)), float)
            if np.any(np.isfinite(delta_a[m0])):
                w0 = np.where(np.isfinite(delta_a[m0]) & (delta_a[m0] > 0.0), delta_a[m0], 1.0)
            L0 = float(np.nanmax(total_length[m0])) if np.any(np.isfinite(total_length[m0])) else np.nan
            rows.append(
                dict(
                    step_index=0,
                    outer_coupling_cycle=int(np.nanmax(outer[m0])) if np.any(m0) else -1,
                    global_step=0,
                    inner_step=0,
                    a_m=L0,
                    delta_a_m=0.0,
                    KI_MPa_sqrt_m=float(np.average(KI[m0], weights=w0)),
                    KII_MPa_sqrt_m=float(np.average(KII[m0], weights=w0)),
                )
            )

    mg = (phase == "inner_step") & grew
    idx = np.where(mg)[0]
    if idx.size == 0:
        if rows:
            return {k: np.asarray([r[k] for r in rows]) for k in rows[0].keys()}
        raise ValueError("No growing inner-step records found in tip history")

    order = np.lexsort((inner_step[idx], global_step[idx], source[idx], outer[idx]))
    idx = idx[order]

    group_keys = np.column_stack([outer[idx], global_step[idx], inner_step[idx]])

    start = 0
    step_counter = len(rows)
    while start < idx.size:
        stop = start + 1
        while stop < idx.size and np.all(group_keys[stop] == group_keys[start]):
            stop += 1

        gidx = idx[start:stop]
        da = np.asarray(delta_a[gidx], float)
        da_pos = np.where(np.isfinite(da) & (da > 0.0), da, 0.0)
        da_total = float(np.sum(da_pos))
        if da_total <= 0.0:
            # Fallback for histories without delta_a: use equal weighting and unit step.
            da_pos = np.ones((gidx.size,), float)
            da_total = float(gidx.size)

        w = da_pos / da_total
        KI_eq = float(np.sum(w * KI[gidx]))
        KII_eq = float(np.sum(w * KII[gidx]))

        L_after = float(np.nanmax(total_length[gidx])) if np.any(np.isfinite(total_length[gidx])) else np.nan

        rows.append(
            dict(
                step_index=step_counter,
                outer_coupling_cycle=int(outer[gidx[0]]),
                global_step=int(global_step[gidx[0]]),
                inner_step=int(inner_step[gidx[0]]),
                a_m=L_after,
                delta_a_m=da_total,
                KI_MPa_sqrt_m=KI_eq,
                KII_MPa_sqrt_m=KII_eq,
            )
        )
        step_counter += 1
        start = stop

    if not rows:
        raise ValueError("Could not assemble any growth-path rows from tip history")

    out = {k: np.asarray([r[k] for r in rows]) for k in rows[0].keys()}

    a = np.asarray(out["a_m"], float)
    da = np.asarray(out["delta_a_m"], float)
    if not np.all(np.isfinite(a)):
        a0 = float(initial_total_length_m) if initial_total_length_m is not None else 0.0
        if not np.isfinite(a0):
            a0 = 0.0
        a_fallback = a0 + np.cumsum(np.where(np.isfinite(da), da, 0.0))
        if include_initial and rows and np.isfinite(a[0]):
            a_fallback[0] = a[0]
        mask = ~np.isfinite(a)
        a[mask] = a_fallback[mask]
        out["a_m"] = a

    return out


def compute_force_control_energetics(path: Mapping[str, np.ndarray], cfg: EnergeticsConfig) -> Dict[str, np.ndarray]:
    """Compute U, Pi, delta_U, compliance C and G versus crack length a.

    Uses:
      G = (KI^2 + KII^2)/E'
      dC/dA = 2G/P^2
      U = 0.5 P^2 C
      Pi = -U
      delta_U[k] = U[k-1] - U[k]
    """
    a, KI, KII = _validate_path_arrays(path)

    if float(cfg.P) == 0.0:
        raise ValueError("cfg.P must be non-zero for force-control energetics")

    G = G_from_sif(KI, KII, E=cfg.E, nu=cfg.nu, plane_stress=cfg.plane_stress)

    area_factor = float(cfg.fracture_surfaces) * float(cfg.thickness)
    A = area_factor * a

    dC_dA = 2.0 * G / (float(cfg.P) ** 2)

    C = np.full_like(a, np.nan, dtype=float)
    C[0] = float(cfg.compliance_C0)
    for i in range(1, len(a)):
        dA = float(A[i] - A[i - 1])
        C[i] = C[i - 1] + 0.5 * float(dC_dA[i] + dC_dA[i - 1]) * dA

    U = 0.5 * (float(cfg.P) ** 2) * C
    Wext = (float(cfg.P) ** 2) * C
    Pi = U - Wext
    delta_U = _incremental_drop(U)
    delta_Pi = _incremental_drop(Pi)

    if len(A) > 1:
        dPi_dA = np.gradient(Pi, A, edge_order=1)
        G_from_Pi = -dPi_dA
    else:
        G_from_Pi = np.full_like(Pi, np.nan)

    Gamma = cfg.fracture_energy_Gamma
    F = None
    if Gamma is not None:
        F = Pi + float(Gamma) * A

    return {
        "a_m": a,
        "A_m2": A,
        "KI_MPa_sqrt_m": KI,
        "KII_MPa_sqrt_m": KII,
        "G_J_per_m2": G,
        "dC_dA_1_per_Pa": dC_dA,
        "C_m_per_N": C,
        "U_J": U,
        "Wext_J": Wext,
        "Pi_J": Pi,
        "Phi_force_J": Pi,
        "delta_U_J": delta_U,
        "delta_Pi_J": delta_Pi,
        "G_from_Pi_J_per_m2": G_from_Pi,
        **({"F_J": F} if F is not None else {}),
    }


def compute_displacement_control_energetics(
    path: Mapping[str, np.ndarray],
    cfg: EnergeticsConfig,
    *,
    delta: float | None = None,
) -> Dict[str, np.ndarray]:
    """Compute displacement-control energetics versus crack length a.

    Uses:
      G = (KI^2 + KII^2)/E'
      dC/dA = 2 G C^2 / delta^2
      U = Phi_disp = 0.5 * delta^2 / C
      P = delta / C
      delta_Phi[k] = Phi[k-1] - Phi[k]
    """
    a, KI, KII = _validate_path_arrays(path)
    G = G_from_sif(KI, KII, E=cfg.E, nu=cfg.nu, plane_stress=cfg.plane_stress)

    delta_eff = float(cfg.delta if delta is None else delta)
    if not np.isfinite(delta_eff) or abs(delta_eff) == 0.0:
        raise ValueError("Non-zero displacement amplitude is required for displacement-control energetics.")

    area_factor = float(cfg.fracture_surfaces) * float(cfg.thickness)
    A = area_factor * a

    C = np.full_like(a, np.nan, dtype=float)
    C[0] = float(cfg.compliance_C0)
    if not np.isfinite(C[0]) or C[0] <= 0.0:
        raise ValueError("cfg.compliance_C0 must be finite and > 0 for displacement-control energetics.")

    # Integrate dC/dA = k(A)*C^2 with piecewise-constant k over each step:
    #   1/C_i = 1/C_{i-1} - k_i * dA
    for i in range(1, len(a)):
        dA = float(A[i] - A[i - 1])
        kbar = 2.0 * 0.5 * float(G[i] + G[i - 1]) / (delta_eff * delta_eff)
        denom = (1.0 / C[i - 1]) - kbar * dA
        C[i] = np.nan if denom <= 0.0 else (1.0 / denom)

    P = delta_eff / C
    U = 0.5 * (delta_eff * delta_eff) / C
    Pi = U
    delta_U = _incremental_drop(U)
    delta_Pi = _incremental_drop(Pi)

    dC_dA = np.full_like(C, np.nan, dtype=float)
    if len(A) > 1:
        dC_dA = np.gradient(C, A, edge_order=1)
    G_from_C = 0.5 * (delta_eff * delta_eff) * dC_dA / (C * C)

    Gamma = cfg.fracture_energy_Gamma
    F = None
    if Gamma is not None:
        F = Pi + float(Gamma) * A

    return {
        "a_m": a,
        "A_m2": A,
        "KI_MPa_sqrt_m": KI,
        "KII_MPa_sqrt_m": KII,
        "G_J_per_m2": G,
        "dC_dA_1_per_Pa": dC_dA,
        "C_m_per_N": C,
        "delta_m": np.full_like(a, delta_eff, dtype=float),
        "P_N": P,
        "U_J": U,
        "Pi_J": Pi,
        "Phi_disp_J": Pi,
        "delta_U_J": delta_U,
        "delta_Pi_J": delta_Pi,
        "G_from_C_J_per_m2": G_from_C,
        **({"F_J": F} if F is not None else {}),
    }


def save_energetics_npz(path: str | Path, energetics: Mapping[str, np.ndarray]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{k: np.asarray(v) for k, v in energetics.items()})
    return p


def plot_energetics_vs_length(
    energetics: Mapping[str, np.ndarray],
    *,
    out_dir: str | Path | None = None,
    show: bool = True,
    prefix: str = "energetics",
) -> Dict[str, Path]:
    """Create separate plots of U, Pi, delta_U, C, G versus crack length a."""
    a_mm = np.asarray(energetics["a_m"], float) * 1e3
    curves = [
        ("U_J", "U [J]", "Stored Elastic Energy vs Total Crack Length"),
        ("Wext_J", r"$W_{ext}$ [J]", "External Work vs Total Crack Length"),
        ("Pi_J", r"$\Pi$ [J]", "Total Potential Energy vs Total Crack Length"),
        ("delta_U_J", r"$\Delta U_k = U_{k-1}-U_k$ [J]", "Incremental Stored-Energy Drop vs Total Crack Length"),
        ("delta_Pi_J", r"$\Delta \Pi_k = \Pi_{k-1}-\Pi_k$ [J]", "Incremental Potential-Energy Drop vs Total Crack Length"),
        ("F_J", "F [J]", "Fracture Functional vs Total Crack Length"),
        ("C_m_per_N", "Compliance C [m/N]", "Compliance vs Total Crack Length"),
        ("P_N", "Load P [N]", "Load vs Total Crack Length"),
        ("G_J_per_m2", "Energy Release Rate G [J/m$^2$]", "Energy Release Rate vs Total Crack Length"),
        ("G_from_Pi_J_per_m2", r"$-d\Pi/dA$ [J/m$^2$]", "Energy Release from Potential Derivative"),
        ("G_from_C_J_per_m2", r"$\frac{1}{2}\delta^2 C^{-2} dC/dA$ [J/m$^2$]", "Energy Release from Compliance Relation"),
    ]

    out_paths: Dict[str, Path] = {}
    out_root = Path(out_dir) if out_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for key, ylabel, title in curves:
        if key not in energetics:
            continue
        y = np.asarray(energetics[key], float)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.plot(a_mm, y, "-o", lw=1.4, ms=3.8)
        ax.set_xlabel("Total crack length a [mm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if out_root is not None:
            p = out_root / f"{prefix}_{key}_vs_a.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            out_paths[key] = p

        if show:
            plt.show()
        else:
            plt.close(fig)

    return out_paths
