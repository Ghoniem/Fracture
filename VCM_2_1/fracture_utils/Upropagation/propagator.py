# fracture_utils/Upropagation/propagator.py
# CrackPropagator supporting fixed_step and adaptive_step (StepController signature in step_control.py).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
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

        # Step controller is only used for adaptive_step
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
        net_new = network

        # Baseline solve once (solver dict + calc handle)
        base_sol = self.evaluator.solve(network)

        step_mode = str(getattr(self.cfg, "step_mode", "fixed_step")).lower()

        # Helper to build a candidate network for StepController
        def build_candidate(net_base, tip, theta, delta_a):
            return updater.extend_tip(net_base, tip, theta=float(theta), delta_a=float(delta_a))

        for tip in tips:
            which = str(getattr(getattr(tip, "tip_id", tip), "which", getattr(tip, "which", "unknown")))
            vid = int(getattr(tip, "v_tip", getattr(tip, "vid", -1)))
            Ltot = float(getattr(tip, "total_length", 0.0))

            # Evaluate baseline SIF/direction (from already-solved base_sol)
            tev0 = self.evaluator.eval_tip(base_sol, tip)
            KI0 = float(tev0.KI); KII0 = float(tev0.KII)
            keff0 = float(tev0.keff); theta0 = float(tev0.theta)

            # toughness at this tip
            try:
                Kc = float(self.toughness.Kc(float(tip.x_tip[0]), float(tip.x_tip[1])))
            except Exception:
                Kc = 0.0

            if keff0 <= Kc:
                reports.append(TipPropagationReport(vid, which, False, 0.0, theta0, keff0, KI0, KII0, "below_toughness"))
                continue

            # Choose delta_a and theta
            theta = theta0
            delta_a = 0.0

            if step_mode == "adaptive_step" and self.step_controller is not None:
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
                    if not bool(getattr(decision, "accepted", True)):
                        # If adaptive failed to accept, fall back to fixed (safe)
                        step_mode = "fixed_step"
                        reports.append(TipPropagationReport(vid, which, False, 0.0, theta0, keff0, KI0, KII0, f"adaptive_not_accepted:{getattr(decision,'reason','')}"))
                except Exception as e:
                    # fall back to fixed
                    step_mode = "fixed_step"
                    reports.append(TipPropagationReport(vid, which, False, 0.0, theta0, keff0, KI0, KII0, f"adaptive_failed:{e}"))

            if step_mode == "fixed_step":
                f = float(getattr(self.cfg, "f_fixed", getattr(self.cfg, "f0", 0.10)))
                delta_a = float(f * Ltot)

            # minimum delta_a
            delta_a_min = float(getattr(self.cfg, "delta_a_min", 0.0))
            if delta_a < delta_a_min:
                delta_a = delta_a_min

            if delta_a <= 0.0:
                reports.append(TipPropagationReport(vid, which, False, 0.0, theta, keff0, KI0, KII0, "zero_step"))
                continue

            try:
                net_new = updater.extend_tip(net_new, tip, theta=theta, delta_a=delta_a)
                reports.append(TipPropagationReport(vid, which, True, delta_a, theta, keff0, KI0, KII0, ""))
            except Exception as e:
                reports.append(TipPropagationReport(vid, which, False, 0.0, theta, keff0, KI0, KII0, f"extend_failed:{e}"))

        # Optional remesh
        if self.remesh_policy is not None:
            try:
                net_new = self.remesh_policy.remesh_network(net_new, solver_kwargs=getattr(self.evaluator, "solver_kwargs", None))
            except Exception:
                try:
                    net_new = self.remesh_policy.remesh_network(net_new)
                except Exception:
                    pass

        return PropagationResult(network_new=net_new, reports=reports)
