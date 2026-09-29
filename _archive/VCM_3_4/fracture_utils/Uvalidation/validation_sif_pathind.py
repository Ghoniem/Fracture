# validation_sif_pathind.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class SIFMeasures4:
    KI_tip: float
    KI_jump: float
    KI_pk: float
    KI_path: float
    KII_tip: float = 0.0
    KII_jump: float = 0.0
    KII_pk: float = 0.0
    KII_path: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SIFCaseResult4:
    knobs: Dict[str, Any]
    numerical: SIFMeasures4
    analytical_KI: float
    analytical_KII: float = 0.0
    err_KI_tip_pct: float = np.nan
    err_KI_jump_pct: float = np.nan
    err_KI_pk_pct: float = np.nan
    err_KI_path_pct: float = np.nan

@dataclass
class SIFSweepResult4:
    crack_meta: Dict[str, Any]
    results: List[SIFCaseResult4]

from ._common import err_pct as _err_pct  # noqa: F401  (re-export for backwards-compat)

def run_sif_sweep4(
    *,
    crack_meta: Dict[str, Any],
    knob_sweep: Sequence[Dict[str, Any]],
    run_solver_sif: Callable[[Dict[str, Any], Dict[str, Any]], SIFMeasures4],
    analytical_sif: Callable[[Dict[str, Any]], Tuple[float, float]],
) -> SIFSweepResult4:
    KI_ana, KII_ana = analytical_sif(crack_meta)
    out: List[SIFCaseResult4] = []
    for knobs in knob_sweep:
        meas = run_solver_sif(knobs, crack_meta)
        out.append(SIFCaseResult4(
            knobs=dict(knobs),
            numerical=meas,
            analytical_KI=float(KI_ana),
            analytical_KII=float(KII_ana),
            err_KI_tip_pct=_err_pct(meas.KI_tip, KI_ana),
            err_KI_jump_pct=_err_pct(meas.KI_jump, KI_ana),
            err_KI_pk_pct=_err_pct(meas.KI_pk, KI_ana),
            err_KI_path_pct=_err_pct(meas.KI_path, KI_ana),
        ))
    return SIFSweepResult4(crack_meta=dict(crack_meta), results=out)

def plot_sif_error_vs_knob4(
    sweep: SIFSweepResult4,
    *,
    knob: str,
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    xs=[]; e_tip=[]; e_jump=[]; e_pk=[]; e_path=[]
    for r in sweep.results:
        if knob not in r.knobs:
            continue
        xs.append(r.knobs[knob])
        e_tip.append(r.err_KI_tip_pct)
        e_jump.append(r.err_KI_jump_pct)
        e_pk.append(r.err_KI_pk_pct)
        e_path.append(r.err_KI_path_pct)

    if not xs:
        raise ValueError(f"No results contained knob '{knob}'")

    try:
        order = np.argsort(np.asarray(xs, float))
        xs = [xs[i] for i in order]
        e_tip = [e_tip[i] for i in order]
        e_jump = [e_jump[i] for i in order]
        e_pk = [e_pk[i] for i in order]
        e_path = [e_path[i] for i in order]
    except Exception:
        pass

    if title is None:
        title = f"Mode I SIF error vs {knob}"

    fig, ax = plt.subplots()
    ax.plot(xs, e_tip, marker="o", label="K_from_tip")
    ax.plot(xs, e_jump, marker="o", label="K_from_jump")
    ax.plot(xs, e_pk, marker="o", label="K_from_PK_window")
    #ax.plot(xs, e_path, marker="o", label="K_from_J_contour")
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
