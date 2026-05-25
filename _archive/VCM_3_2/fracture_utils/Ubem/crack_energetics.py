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
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EnergeticsConfig:
    E: float
    nu: float
    plane_stress: bool
    P: float
    thickness: float = 1.0
    fracture_surfaces: float = 2.0
    compliance_C0: float = 0.0


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
    a = np.asarray(path["a_m"], float)
    KI = np.asarray(path["KI_MPa_sqrt_m"], float)
    KII = np.asarray(path["KII_MPa_sqrt_m"], float)

    if a.ndim != 1 or KI.ndim != 1 or KII.ndim != 1 or not (len(a) == len(KI) == len(KII)):
        raise ValueError("path arrays a/KI/KII must be 1D with equal lengths")

    if len(a) == 0:
        raise ValueError("empty path")

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
    Pi = -U

    delta_U = np.full_like(U, np.nan)
    if len(U) > 1:
        delta_U[1:] = U[:-1] - U[1:]

    return {
        "a_m": a,
        "A_m2": A,
        "KI_MPa_sqrt_m": KI,
        "KII_MPa_sqrt_m": KII,
        "G_J_per_m2": G,
        "dC_dA_1_per_Pa": dC_dA,
        "C_m_per_N": C,
        "U_J": U,
        "Pi_J": Pi,
        "delta_U_J": delta_U,
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
        ("Pi_J", r"$\Pi$ [J]", "Total Potential Energy vs Total Crack Length"),
        ("delta_U_J", r"$\Delta U_k = U_{k-1}-U_k$ [J]", "Incremental Energy Drop vs Total Crack Length"),
        ("C_m_per_N", "Compliance C [m/N]", "Compliance vs Total Crack Length"),
        ("G_J_per_m2", "Energy Release Rate G [J/m$^2$]", "Energy Release Rate vs Total Crack Length"),
    ]

    out_paths: Dict[str, Path] = {}
    out_root = Path(out_dir) if out_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for key, ylabel, title in curves:
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
