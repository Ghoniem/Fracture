# validation_sif.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SIFMeasures:
    """
    Plane-strain SIF measures for Mode I/II.

    You can set KII=0 for pure Mode I cases if preferred.
    """
    KI_tip: float
    KI_jump: float
    KI_pk: float
    KII_tip: float = 0.0
    KII_jump: float = 0.0
    KII_pk: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SIFCaseResult:
    knobs: Dict[str, Any]
    numerical: SIFMeasures
    analytical_KI: float
    analytical_KII: float = 0.0

    # errors (percent)
    err_KI_tip_pct: float = np.nan
    err_KI_jump_pct: float = np.nan
    err_KI_pk_pct: float = np.nan
    err_KII_tip_pct: float = np.nan
    err_KII_jump_pct: float = np.nan
    err_KII_pk_pct: float = np.nan


@dataclass
class SIFSweepResult:
    crack_meta: Dict[str, Any]
    results: List[SIFCaseResult]


def _err_pct(num: float, ana: float, eps: float = 1e-14) -> float:
    """Relative percentage error, NaN when the analytical value cannot anchor one.

    The old implementation fabricated a tiny denominator (sign * eps + eps) when
    |ana| <= eps, which silently produced enormous percentage errors for cases
    like pure Mode I where KII_ana == 0 — making the K_II validation columns
    useless. Returning NaN is honest: downstream consumers use nanmean/nanstd
    and will simply skip these rows.
    """
    if abs(ana) <= eps:
        return float("nan")
    return float(100.0 * (num - ana) / ana)


def run_sif_sweep(
    *,
    crack_meta: Dict[str, Any],
    knob_sweep: Sequence[Dict[str, Any]],
    run_solver_sif: Callable[[Dict[str, Any], Dict[str, Any]], SIFMeasures],
    analytical_sif: Callable[[Dict[str, Any]], Tuple[float, float]],
) -> SIFSweepResult:
    """
    Parameters
    ----------
    crack_meta:
        Must include what analytical_sif needs (e.g., a, sigma, geometry flags, etc.).
    knob_sweep:
        List of knob dictionaries.
    run_solver_sif(knobs, crack_meta) -> SIFMeasures:
        Must compute KI/KII using your three estimators: tip, jump, PK.
    analytical_sif(crack_meta) -> (KI, KII):
        Analytical plane-strain SIF reference.

    Returns
    -------
    SIFSweepResult
    """
    out: List[SIFCaseResult] = []
    KI_ana, KII_ana = analytical_sif(crack_meta)

    for knobs in knob_sweep:
        meas = run_solver_sif(knobs, crack_meta)

        r = SIFCaseResult(
            knobs=dict(knobs),
            numerical=meas,
            analytical_KI=float(KI_ana),
            analytical_KII=float(KII_ana),
            err_KI_tip_pct=_err_pct(meas.KI_tip, KI_ana),
            err_KI_jump_pct=_err_pct(meas.KI_jump, KI_ana),
            err_KI_pk_pct=_err_pct(meas.KI_pk, KI_ana),
            err_KII_tip_pct=_err_pct(meas.KII_tip, KII_ana),
            err_KII_jump_pct=_err_pct(meas.KII_jump, KII_ana),
            err_KII_pk_pct=_err_pct(meas.KII_pk, KII_ana),
        )
        out.append(r)

    return SIFSweepResult(crack_meta=dict(crack_meta), results=out)


def plot_sif_error_vs_knob(
    sweep: SIFSweepResult,
    *,
    knob: str,
    mode: str = "I",  # "I" or "II"
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Error (%) vs knob for three estimators (tip, jump, pk).
    """
    xs = []
    tip = []
    jmp = []
    pk = []

    for r in sweep.results:
        if knob not in r.knobs:
            continue
        xs.append(r.knobs[knob])
        if mode.upper() == "I":
            tip.append(r.err_KI_tip_pct)
            jmp.append(r.err_KI_jump_pct)
            pk.append(r.err_KI_pk_pct)
        else:
            tip.append(r.err_KII_tip_pct)
            jmp.append(r.err_KII_jump_pct)
            pk.append(r.err_KII_pk_pct)

    if not xs:
        raise ValueError(f"No results contained knob '{knob}'")

    # sort by x if numeric
    try:
        order = np.argsort(np.asarray(xs, float))
        xs = [xs[i] for i in order]
        tip = [tip[i] for i in order]
        jmp = [jmp[i] for i in order]
        pk  = [pk[i]  for i in order]
    except Exception:
        pass

    if title is None:
        title = f"Mode {mode.upper()} SIF error vs {knob}"

    fig, ax = plt.subplots()
    ax.plot(xs, tip, marker="o", label="K_from_tip")
    ax.plot(xs, jmp, marker="o", label="K_from_jump")
    ax.plot(xs, pk,  marker="o", label="K_from_PK")
    ax.set_xlabel(knob)
    ax.set_ylabel("Error (%)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, ax


def plot_sif_estimators_overlay(
    sweep: SIFSweepResult,
    *,
    knob: str,
    mode: str = "I",
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Plot the raw estimator values vs knob, with the analytical line.
    Useful to see estimator convergence to the correct value.
    """
    xs = []
    tip = []
    jmp = []
    pk = []

    if mode.upper() == "I":
        ana = sweep.results[0].analytical_KI if sweep.results else np.nan
    else:
        ana = sweep.results[0].analytical_KII if sweep.results else np.nan

    for r in sweep.results:
        if knob not in r.knobs:
            continue
        xs.append(r.knobs[knob])
        if mode.upper() == "I":
            tip.append(r.numerical.KI_tip)
            jmp.append(r.numerical.KI_jump)
            pk.append(r.numerical.KI_pk)
        else:
            tip.append(r.numerical.KII_tip)
            jmp.append(r.numerical.KII_jump)
            pk.append(r.numerical.KII_pk)

    if not xs:
        raise ValueError(f"No results contained knob '{knob}'")

    try:
        order = np.argsort(np.asarray(xs, float))
        xs = [xs[i] for i in order]
        tip = [tip[i] for i in order]
        jmp = [jmp[i] for i in order]
        pk  = [pk[i]  for i in order]
    except Exception:
        pass

    if title is None:
        title = f"Mode {mode.upper()} SIF estimators vs {knob}"

    fig, ax = plt.subplots()
    ax.plot(xs, tip, marker="o", label="K_from_tip")
    ax.plot(xs, jmp, marker="o", label="K_from_jump")
    ax.plot(xs, pk,  marker="o", label="K_from_PK")
    ax.axhline(ana, linewidth=2, label="Analytical")
    ax.set_xlabel(knob)
    ax.set_ylabel(f"K_{mode.upper()}")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, ax
