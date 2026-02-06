"""Adaptive step-size selection for crack growth.

Implements the f / 2f / (f/2) search with angle (+ optional keff) stability checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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

class StepController:
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

            # --- Trial at f
            net1 = build_candidate(base_network, tip, theta0, da1)
            sol1 = evaluator.solve(net1)
            ev1 = evaluator.eval_tip(sol1, tip)
            trials.append(TrialRecord(f=f, delta_a=da1, theta=ev1.theta, keff=ev1.keff, accepted=False))

            # --- Trial at 2f (if distinct)
            if da2 > da1 * (1.0 + 1e-12):
                net2 = build_candidate(base_network, tip, theta0, da2)
                sol2 = evaluator.solve(net2)
                ev2 = evaluator.eval_tip(sol2, tip)
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
