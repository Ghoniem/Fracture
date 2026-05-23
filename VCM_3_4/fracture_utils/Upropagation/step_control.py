"""Step-size selection strategies for crack growth.

Two modes are supported (via PropagationConfig.step_mode):

- fixed_step: Δa = f_fixed * L_total (no recursion, no extra solves)
- adaptive_step: legacy f / 2f / (f/2) stability search (angle + optional keff)

This module is solver-agnostic; it relies on `evaluator.solve(...)` and
`evaluator.eval_tip(...)` when adaptive stepping is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .config import PropagationConfig
from .results import TrialRecord
from .tip_state import TipState


@dataclass
class StepDecision:
    accepted: bool
    f: float
    delta_a: float
    theta: float
    keff: float
    reason: str
    trials: list[TrialRecord]


class StepControllerProto(Protocol):
    def choose_step(
        self,
        *,
        evaluator,
        base_network,
        base_sol: Dict[str, Any],
        tip: TipState,
        theta0: float,
        keff0: float,
        # build_candidate(network, tip, theta, delta_a) must return
        # the tuple (new_network, new_tip_state) so the controller can evaluate
        # SIFs at the freshly-grown tip rather than the now-interior old one.
        build_candidate,
        total_length: float,
    ) -> StepDecision: ...


class FixedStepController:
    """Always accept a single step using cfg.f_fixed.

    This mode performs *no* extra solves. It is the recommended baseline while
    debugging propagation geometry and direction logic.
    """

    def __init__(self, cfg: PropagationConfig):
        self.cfg = cfg

    def choose_step(
        self,
        *,
        evaluator,          # unused
        base_network,       # unused
        base_sol: Dict[str, Any],  # unused
        tip: TipState,      # unused
        theta0: float,
        keff0: float,
        build_candidate,    # unused
        total_length: float,
    ) -> StepDecision:
        cfg = self.cfg
        f = float(cfg.f_fixed)
        da = float(f * float(total_length))
        if cfg.delta_a_min > 0.0:
            da = max(da, float(cfg.delta_a_min))
        return StepDecision(
            accepted=True,
            f=f,
            delta_a=da,
            theta=float(theta0),
            keff=float(keff0),
            reason="fixed_step",
            trials=[],
        )


class AdaptiveStepController:
    """Adaptive f selection using stability between f and 2f (shrinking by /2)."""

    def __init__(self, cfg: PropagationConfig):
        self.cfg = cfg

    def choose_step(
        self,
        *,
        evaluator,
        base_network,
        base_sol: Dict[str, Any],
        tip: TipState,
        theta0: float,
        keff0: float,
        build_candidate,   # callable(network, tip, theta, delta_a) -> new_network
        total_length: float,
    ) -> StepDecision:
        cfg = self.cfg
        f = float(min(max(cfg.f0, cfg.f_min), cfg.f_max))
        trials: list[TrialRecord] = []

        def stable(theta_a, theta_b, keff_a, keff_b) -> bool:
            # angle: use max(relative, absolute)
            dth = abs(float(theta_b) - float(theta_a))
            th_scale = max(abs(float(theta_b)), 1e-14)
            ok_th = (dth <= cfg.tol_theta_abs) or (dth / th_scale <= cfg.tol_theta_rel)

            # keff: relative
            dk = abs(float(keff_b) - float(keff_a))
            k_scale = max(abs(float(keff_b)), 1e-14)
            ok_k = (dk / k_scale) <= cfg.tol_keff_rel
            return bool(ok_th and ok_k)

        for _it in range(int(cfg.max_trials)):
            f = float(min(max(f, cfg.f_min), cfg.f_max))
            da1 = float(f * total_length)
            da2 = float(min(2.0 * f, cfg.f_max) * total_length)

            if cfg.delta_a_min > 0.0:
                da1 = max(da1, float(cfg.delta_a_min))
                da2 = max(da2, float(cfg.delta_a_min))

            # --- Trial at f
            net1, tip1 = build_candidate(base_network, tip, float(theta0), da1)
            sol1 = evaluator.solve(net1)
            ev1 = evaluator.eval_tip(sol1, tip1)
            trials.append(TrialRecord(f=f, delta_a=da1, theta=ev1.theta, keff=ev1.keff, accepted=False))

            # --- Trial at 2f (if distinct)
            if da2 > da1 * (1.0 + 1e-12):
                net2, tip2 = build_candidate(base_network, tip, float(theta0), da2)
                sol2 = evaluator.solve(net2)
                ev2 = evaluator.eval_tip(sol2, tip2)
                trials.append(TrialRecord(f=min(2.0*f, cfg.f_max), delta_a=da2, theta=ev2.theta, keff=ev2.keff, accepted=False))

                if stable(ev1.theta, ev2.theta, ev1.keff, ev2.keff):
                    # prefer larger step
                    if cfg.prefer_larger_step:
                        trials[-1].accepted = True
                        return StepDecision(True, min(2.0*f, cfg.f_max), da2, ev2.theta, ev2.keff, "accepted_2f", trials)
                    trials[-2].accepted = True
                    return StepDecision(True, f, da1, ev1.theta, ev1.keff, "accepted_f", trials)

            # Not stable: shrink f
            if f <= cfg.f_min * (1.0 + 1e-12):
                return StepDecision(False, f, da1, ev1.theta, ev1.keff, "reached_f_min", trials)
            f *= 0.5

        return StepDecision(False, f, float(f*total_length), float(theta0), float(keff0), "max_trials", trials)


def make_step_controller(cfg: PropagationConfig) -> StepControllerProto:
    """Factory: build controller from cfg.step_mode."""
    if str(getattr(cfg, "step_mode", "fixed_step")) == "adaptive_step":
        return AdaptiveStepController(cfg)
    return FixedStepController(cfg)
