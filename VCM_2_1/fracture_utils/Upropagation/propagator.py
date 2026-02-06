# fracture_utils/Upropagation/propagator.py
# CrackPropagator: fixed_step works for arbitrary networks (multiple open polylines, multiple deg-1 tips).
# Key property: "two-phase" update (evaluate all tips on baseline solution, then apply updates),
# which avoids order dependence when multiple tips grow in the same increment.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np


@dataclass
class TipPropagationReport:
    tip_vid: int
    which: str
    grew: bool
    delta_a: float
    theta: float
    keff: float
    KI: float
    KII: float
    reason: str = ""


@dataclass
class PropagationResult:
    network_new: Any
    reports: List[TipPropagationReport]


class CrackPropagator:
    def __init__(
        self,
        *,
        cfg: Any,
        evaluator: Any,
        toughness: Any,
        direction_law: Any,
        remesh_policy: Any = None,
        updater: Any = None,
    ):
        self.cfg = cfg
        self.evaluator = evaluator
        self.toughness = toughness
        self.direction_law = direction_law
        self.remesh_policy = remesh_policy
        self.updater = updater

        # Step controller is only used for adaptive_step (if available)
        self.step_controller = None
        try:
            from .step_control import StepController
            self.step_controller = StepController(cfg)
        except Exception:
            self.step_controller = None

    def _get_updater(self):
        if self.updater is not None:
            return self.updater
        from .geometry_update import GeometryUpdater
        return GeometryUpdater()

    def _step_mode(self) -> str:
        return str(getattr(self.cfg, "step_mode", "fixed_step")).lower()

    def _simultaneous(self) -> bool:
        return bool(getattr(self.cfg, "simultaneous_tip_growth", True))

    def grow_one_increment(self, network: Any) -> PropagationResult:
        from .network_ops import get_netops

        netops = get_netops()
        deg = netops.degree_map(network)
        polylines = netops.extract_open_polylines(network)
        tips = netops.extract_deg1_tips(network, polylines, deg) or []

        if len(tips) == 0:
            return PropagationResult(network_new=network, reports=[])

        updater = self._get_updater()
        reports: List[TipPropagationReport] = []

        # ---- Solve baseline once
        base_sol = self.evaluator.solve(network)

        # ---- Phase A: evaluate all tips and decide growth based on the SAME baseline solution
        decisions: List[Tuple[Any, float, float, float, float, float, float, float, str]] = []
        # tuple: (tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason)

        step_mode = self._step_mode()

        for tip in tips:
            which = str(getattr(getattr(tip, "tip_id", tip), "which", getattr(tip, "which", "unknown")))
            vid = int(getattr(tip, "v_tip", getattr(tip, "vid", -1)))
            Ltot = float(getattr(tip, "total_length", 0.0))

            tev0 = self.evaluator.eval_tip(base_sol, tip)
            KI0 = float(tev0.KI); KII0 = float(tev0.KII)
            keff0 = float(tev0.keff); theta0 = float(tev0.theta)

            # toughness
            try:
                Kc = float(self.toughness.Kc(float(tip.x_tip[0]), float(tip.x_tip[1])))
            except Exception:
                Kc = 0.0

            if keff0 <= Kc:
                decisions.append((tip, 0.0, theta0, keff0, KI0, KII0, Kc, Ltot, "below_toughness"))
                continue

            theta = theta0
            delta_a = 0.0
            reason = ""

            if step_mode == "adaptive_step" and self.step_controller is not None:
                # adaptive controller expects keyword-only signature (your step_control.py)
                def build_candidate(net_base, _tip, _theta, _delta_a):
                    return updater.extend_tip(net_base, _tip, theta=float(_theta), delta_a=float(_delta_a))

                try:
                    decision = self.step_controller.choose_step(
                        evaluator=self.evaluator,
                        base_network=network,
                        base_sol=base_sol,
                        tip=tip,
                        theta0=theta0,
                        keff0=keff0,
                        build_candidate=build_candidate,
                        total_length=Ltot,
                    )
                    delta_a = float(getattr(decision, "delta_a", 0.0))
                    theta = float(getattr(decision, "theta", theta0))
                    accepted = bool(getattr(decision, "accepted", True))
                    if not accepted:
                        reason = f"adaptive_not_accepted:{getattr(decision,'reason','')}"
                        # fall back to fixed
                        f = float(getattr(self.cfg, "f_fixed", getattr(self.cfg, "f0", 0.10)))
                        delta_a = float(f * Ltot)
                        theta = theta0
                except Exception as e:
                    reason = f"adaptive_failed:{e}"
                    # fall back to fixed
                    f = float(getattr(self.cfg, "f_fixed", getattr(self.cfg, "f0", 0.10)))
                    delta_a = float(f * Ltot)
                    theta = theta0

            else:
                # fixed step
                f = float(getattr(self.cfg, "f_fixed", getattr(self.cfg, "f0", 0.10)))
                delta_a = float(f * Ltot)

            # minimum delta_a
            delta_a_min = float(getattr(self.cfg, "delta_a_min", 0.0))
            if delta_a < delta_a_min:
                delta_a = delta_a_min

            if delta_a <= 0.0:
                if not reason:
                    reason = "zero_step"
                decisions.append((tip, 0.0, theta, keff0, KI0, KII0, Kc, Ltot, reason))
            else:
                decisions.append((tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason))

        # If not simultaneous growth: choose a single best tip (max keff - Kc) among those with delta_a>0
        if not self._simultaneous():
            best_i: Optional[int] = None
            best_metric = -np.inf
            for i, (tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason) in enumerate(decisions):
                if delta_a <= 0.0:
                    continue
                metric = float(keff0 - Kc)
                if metric > best_metric:
                    best_metric = metric
                    best_i = i
            # zero out all others
            if best_i is not None:
                for i in range(len(decisions)):
                    if i != best_i:
                        tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason = decisions[i]
                        decisions[i] = (tip, 0.0, theta, keff0, KI0, KII0, Kc, Ltot, "not_selected")

        # ---- Phase B: apply all accepted extensions on the evolving network (but decisions fixed from baseline)
        net_new = network
        for (tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason) in decisions:
            which = str(getattr(getattr(tip, "tip_id", tip), "which", getattr(tip, "which", "unknown")))
            vid = int(getattr(tip, "v_tip", getattr(tip, "vid", -1)))

            if delta_a <= 0.0:
                reports.append(TipPropagationReport(vid, which, False, 0.0, float(theta), float(keff0), float(KI0), float(KII0), reason))
                continue

            try:
                net_new = updater.extend_tip(net_new, tip, theta=float(theta), delta_a=float(delta_a))
                reports.append(TipPropagationReport(vid, which, True, float(delta_a), float(theta), float(keff0), float(KI0), float(KII0), reason))
            except Exception as e:
                reports.append(TipPropagationReport(vid, which, False, 0.0, float(theta), float(keff0), float(KI0), float(KII0), f"extend_failed:{e}"))

        # Optional remesh (keep conservative for now)
        if self.remesh_policy is not None:
            try:
                net_new = self.remesh_policy.remesh_network(net_new, solver_kwargs=getattr(self.evaluator, "solver_kwargs", None))
            except Exception:
                try:
                    net_new = self.remesh_policy.remesh_network(net_new)
                except Exception:
                    pass

        return PropagationResult(network_new=net_new, reports=reports)
