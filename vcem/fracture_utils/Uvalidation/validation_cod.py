# validation_cod.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray


# ----------------------------
# Data containers
# ----------------------------

@dataclass(frozen=True)
class CODProfile:
    """COD vs parametric coordinate s in [0, 1] or arclength x in [-a, a]."""
    s: Array                      # shape (n,)
    cod: Array                    # shape (n,)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CODCaseResult:
    """One solver run + comparison to analytical profile."""
    knobs: Dict[str, Any]
    numerical: CODProfile
    analytical: CODProfile
    cod_abs_err: Array            # |num - ana|
    cod_rel_err_pct: Array        # 100*(num-ana)/ana with safe handling
    # scalar metrics
    l2_rel_pct: float
    linf_rel_pct: float
    rms_abs: float


@dataclass
class CODSweepResult:
    """Results for a sweep over knob values (potentially multiple knobs)."""
    crack_meta: Dict[str, Any]
    results: List[CODCaseResult]


# ----------------------------
# Utilities
# ----------------------------

def _safe_rel_err_pct(num: Array, ana: Array, eps: float = 1e-14) -> Array:
    denom = np.where(np.abs(ana) > eps, ana, np.sign(ana) * eps + eps)
    return 100.0 * (num - ana) / denom


def _metrics(num: Array, ana: Array, eps: float = 1e-14) -> Tuple[float, float, float]:
    """
    Returns: (L2_rel_pct, Linf_rel_pct, RMS_abs)
    """
    diff = num - ana
    ana_norm2 = np.linalg.norm(ana)
    l2_rel = (np.linalg.norm(diff) / (ana_norm2 + eps)) * 100.0
    linf_rel = (np.max(np.abs(diff)) / (np.max(np.abs(ana)) + eps)) * 100.0
    rms_abs = float(np.sqrt(np.mean(diff**2)))
    return float(l2_rel), float(linf_rel), rms_abs


def _pick_overlay_indices(n: int, max_curves: int = 5) -> List[int]:
    """Pick up to max_curves indices evenly spaced for overlay plots."""
    if n <= max_curves:
        return list(range(n))
    return list(np.linspace(0, n - 1, max_curves).round().astype(int))


def _format_knobs(knobs: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> str:
    if keys is None:
        keys = list(knobs.keys())
    parts = []
    for k in keys:
        if k in knobs:
            parts.append(f"{k}={knobs[k]}")
    return ", ".join(parts)


# ----------------------------
# Public API
# ----------------------------

def run_cod_sweep(
    *,
    crack_meta: Dict[str, Any],
    knob_sweep: Sequence[Dict[str, Any]],
    # user-provided callables
    run_solver: Callable[[Dict[str, Any], Dict[str, Any]], CODProfile],
    analytical_cod: Callable[[Array, Dict[str, Any]], Array],
    # sampling
    s_sample: Union[int, Array] = 201,
    align_to: str = "s",   # "s" (0..1) or "x" (physical coordinate); keeps module generic
) -> CODSweepResult:
    """
    Parameters
    ----------
    crack_meta:
        Crack description used by your solver and analytical function
        (e.g., {"a": 1.0, "sigma": 1.0, "E":..., "nu":..., ...}).
    knob_sweep:
        List of knob dictionaries; each entry becomes one solver run.
    run_solver(knobs, crack_meta) -> CODProfile:
        Must return s and cod arrays.
    analytical_cod(s, crack_meta) -> cod_analytical:
        Must accept s-array and crack_meta and return COD at those sample points.
    s_sample:
        Either an int (number of points in [0,1]) or an explicit array.
    align_to:
        Placeholder for future; currently assumes solver returns "s" coordinate.

    Returns
    -------
    CODSweepResult
    """
    if isinstance(s_sample, int):
        s = np.linspace(0.0, 1.0, int(s_sample))
    else:
        s = np.asarray(s_sample, float)

    out: List[CODCaseResult] = []

    for knobs in knob_sweep:
        num_prof = run_solver(knobs, crack_meta)

        # Re-sample numerical onto s (if needed)
        s_num = np.asarray(num_prof.s, float)
        cod_num = np.asarray(num_prof.cod, float)

        # Enforce monotonic s for interpolation
        if s_num.ndim != 1 or cod_num.ndim != 1:
            raise ValueError("CODProfile.s and CODProfile.cod must be 1D arrays.")
        if len(s_num) != len(cod_num):
            raise ValueError("CODProfile.s and CODProfile.cod must have same length.")

        # sort if required
        if np.any(np.diff(s_num) < 0):
            idx = np.argsort(s_num)
            s_num = s_num[idx]
            cod_num = cod_num[idx]

        cod_num_i = np.interp(s, s_num, cod_num)

        cod_ana = np.asarray(analytical_cod(s, crack_meta), float)
        if cod_ana.shape != cod_num_i.shape:
            raise ValueError("analytical_cod returned wrong shape.")

        abs_err = np.abs(cod_num_i - cod_ana)
        rel_err_pct = _safe_rel_err_pct(cod_num_i, cod_ana)
        l2_rel, linf_rel, rms_abs = _metrics(cod_num_i, cod_ana)

        out.append(
            CODCaseResult(
                knobs=dict(knobs),
                numerical=CODProfile(s=s, cod=cod_num_i, meta=dict(num_prof.meta)),
                analytical=CODProfile(s=s, cod=cod_ana, meta={"source": "analytical"}),
                cod_abs_err=abs_err,
                cod_rel_err_pct=rel_err_pct,
                l2_rel_pct=l2_rel,
                linf_rel_pct=linf_rel,
                rms_abs=rms_abs,
            )
        )

    return CODSweepResult(crack_meta=dict(crack_meta), results=out)


def plot_cod_overlay(
    sweep: CODSweepResult,
    *,
    title: str = "COD profile vs analytical",
    knob_label_keys: Optional[Sequence[str]] = None,
    max_curves: int = 5,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Plots analytical COD and up to max_curves numerical COD curves on the same axes.
    """
    res = sweep.results
    if not res:
        raise ValueError("Empty sweep.results")

    idxs = _pick_overlay_indices(len(res), max_curves=max_curves)

    fig, ax = plt.subplots()
    # Analytical from first result (same crack_meta, same s)
    ax.plot(res[0].analytical.s, res[0].analytical.cod, linewidth=2, label="Analytical")

    for i in idxs:
        r = res[i]
        label = _format_knobs(r.knobs, knob_label_keys)
        ax.plot(r.numerical.s, r.numerical.cod, marker=None, label=label)

    ax.set_xlabel("s (parametric coordinate)")
    ax.set_ylabel("COD")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, ax


def plot_cod_error_vs_knob(
    sweep: CODSweepResult,
    *,
    knob: str,
    metric: str = "l2_rel_pct",  # "l2_rel_pct" | "linf_rel_pct" | "rms_abs"
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Scatter/line plot of COD error metric vs a single knob value.

    Assumes sweep varies that knob meaningfully; if multiple knobs vary, you still get a plot
    but interpret carefully.
    """
    xs = []
    ys = []
    for r in sweep.results:
        if knob not in r.knobs:
            continue
        xs.append(r.knobs[knob])
        ys.append(getattr(r, metric))

    if not xs:
        raise ValueError(f"No results contained knob '{knob}'")

    # sort by x if numeric
    try:
        order = np.argsort(np.asarray(xs, float))
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
    except Exception:
        pass

    if title is None:
        title = f"COD error ({metric}) vs {knob}"

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(knob)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, ax
