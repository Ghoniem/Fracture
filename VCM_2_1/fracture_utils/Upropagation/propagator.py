"""High-level propagation orchestrator.

Updates in this version
-----------------------
- Optional arc-length remeshing (Option A) after accepted growth.
- Non-blocking robustness for SIF tip-fit "window too small" failures:
  * Try adaptive (rmax_frac/min_pts) ladder in CandidateEvaluator (already done there).
  * If still failing, escalate ne_half (re-solve) a limited number of times.

Notes
-----
This orchestrator remains conservative: if a tip cannot be evaluated even after
allowed retries, that tip is skipped for the increment (other tips may still grow).
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

from .config import PropagationConfig
from .toughness import ToughnessField
from .direction import DirectionLaw
from .tip_state import extract_tips_from_polylines, TipState
from .step_control import StepController
from .geometry_update import GeometryUpdater
from .results import PropagationResult, TipPropagationReport

try:
    from .remesh_arclength import ArcLengthRemeshPolicy
except Exception:  # pragma: no cover
    ArcLengthRemeshPolicy = None  # type: ignore

if TYPE_CHECKING:
    from fracture_utils.Usolver.network import CrackNetworkV4


@dataclass
class _SolveBundle:
    res: object
    sol: dict
    total_length: float
    polylines: list


class CrackPropagator:
    """Orchestrates one propagation increment for a single (or multi-)polyline crack."""

    def __init__(
        self,
        *,
        cfg: PropagationConfig,
        evaluator,
        toughness: ToughnessField,
        direction_law: DirectionLaw,
        step_controller: Optional[StepController] = None,
        updater: Optional[GeometryUpdater] = None,
        remesh_policy: Optional[object] = None,
    ):
        self.cfg = cfg
        self.evaluator = evaluator
        self.toughness = toughness
        self.direction_law = direction_law
        self.step_controller = step_controller or StepController(cfg)
        self.updater = updater or GeometryUpdater()
        self.remesh_policy = remesh_policy  # e.g., ArcLengthRemeshPolicy()

    # -------------------------
    # Main entry point
    # -------------------------
    def grow_one_increment(self, network: "CrackNetworkV4") -> PropagationResult:
        """Run a single growth increment: decide and apply tip extensions."""
        bundle = self._solve_bundle_with_escalation(network, escalations=0)

        # degrees (solver may not export; compute from network)
        from fracture_utils.Usolver.build_geometry import vertex_degrees
        deg = vertex_degrees(network)

        tips = extract_tips_from_polylines(
            network,
            bundle.polylines,
            deg,
            total_length=float(bundle.total_length),
        )

        reports: List[TipPropagationReport] = []
        net_work = network

        def build_candidate(net_base, tip: TipState, theta: float, delta_a: float):
            return self.updater.extend_tip(net_base, tip, theta=theta, delta_a=delta_a)

        for tip in tips:
            tev0 = self._safe_eval_tip(bundle, tip, reports)
            if tev0 is None:
                continue

            Kc = float(self.toughness.Kc(float(tip.x_tip[0]), float(tip.x_tip[1])))

            if tev0.keff <= Kc:
                reports.append(TipPropagationReport(
                    pid=int(tip.tip_id.pid), which=str(tip.tip_id.which),
                    grew=False, reason="below_toughness",
                    Kc=Kc, KI=tev0.KI, KII=tev0.KII, keff=tev0.keff, theta0=tev0.theta,
                ))
                continue

            # Step control (adaptive f search)
            try:
                decision = self.step_controller.choose_step(
                    evaluator=self.evaluator,
                    base_network=net_work if not self.cfg.simultaneous_tip_growth else network,
                    base_sol=bundle.sol,
                    tip=tip,
                    theta0=float(tev0.theta),
                    keff0=float(tev0.keff),
                    build_candidate=build_candidate,
                    total_length=float(bundle.total_length),
                )
            except Exception as e:
                # If step search fails due to tip-fit window issues, try a single ne_half escalation and retry
                if getattr(self.evaluator, "_is_window_too_small", None) and self.evaluator._is_window_too_small(e):
                    if self.evaluator.enable_ne_half_escalation and self.evaluator.escalate_ne_half():
                        bundle = self._solve_bundle_with_escalation(network, escalations=1)
                        # retry once
                        decision = self.step_controller.choose_step(
                            evaluator=self.evaluator,
                            base_network=net_work if not self.cfg.simultaneous_tip_growth else network,
                            base_sol=bundle.sol,
                            tip=tip,
                            theta0=float(tev0.theta),
                            keff0=float(tev0.keff),
                            build_candidate=build_candidate,
                            total_length=float(bundle.total_length),
                        )
                    else:
                        raise
                else:
                    raise

            rep = TipPropagationReport(
                pid=int(tip.tip_id.pid), which=str(tip.tip_id.which),
                grew=bool(decision.accepted),
                reason=str(decision.reason),
                Kc=Kc, KI=tev0.KI, KII=tev0.KII, keff=tev0.keff, theta0=float(tev0.theta),
                f_accepted=float(decision.f) if decision.accepted else None,
                delta_a=float(decision.delta_a) if decision.accepted else None,
                theta_accepted=float(decision.theta) if decision.accepted else None,
                trials=list(decision.trials),
            )
            reports.append(rep)

            if decision.accepted:
                net_work = build_candidate(net_work, tip, float(tev0.theta), float(decision.delta_a))

        # Optional remesh (Option A) after increment
        net_new = net_work
        if self.remesh_policy is not None:
            try:
                ne_half = int(getattr(self.evaluator, "solver_kwargs", {}).get("ne_half", 0) or 0)
                if ne_half > 0:
                    net_new = self.remesh_policy.remesh_network(net_new, ne_half=ne_half)
            except NotImplementedError:
                # silently skip for unsupported topologies
                pass

        return PropagationResult(network_new=net_new, base_solution=bundle.sol, reports=reports)

    # -------------------------
    # Helpers
    # -------------------------
    def _solve_bundle_with_escalation(self, network: "CrackNetworkV4", escalations: int) -> _SolveBundle:
        """Solve and return (res, sol, total_length, polylines)."""
        res, sol = self.evaluator.solve_results(network)
        polylines = sol.get("parametrized_polylines", []) or sol.get("polylines", []) or []
        total_length = float(sum(float(p.get("total_length", 0.0)) for p in polylines))

        return _SolveBundle(res=res, sol=sol, total_length=total_length, polylines=polylines)

    def _safe_eval_tip(self, bundle: _SolveBundle, tip: TipState, reports: List[TipPropagationReport]):
        """Evaluate tip; if window-too-small persists, try ne_half escalation; else skip tip."""
        # Try with current solve
        try:
            return self.evaluator.eval_tip(bundle.res, tip)
        except Exception as e:
            # If window-too-small, attempt limited ne_half escalation and retry
            if getattr(self.evaluator, "_is_window_too_small", None) and self.evaluator._is_window_too_small(e):
                if self.evaluator.enable_ne_half_escalation:
                    for _ in range(int(getattr(self.evaluator, "ne_half_escalations_max", 1))):
                        if not self.evaluator.escalate_ne_half():
                            break
                        bundle2 = self._solve_bundle_with_escalation(bundle.sol.get("_network", None) or bundle.sol["_network"], escalations=1)
                        try:
                            # update bundle in-place (best effort)
                            bundle.res, bundle.sol, bundle.total_length, bundle.polylines = bundle2.res, bundle2.sol, bundle2.total_length, bundle2.polylines
                            return self.evaluator.eval_tip(bundle.res, tip)
                        except Exception as e2:
                            if not self.evaluator._is_window_too_small(e2):
                                raise
                            e = e2  # continue escalation loop
            # Skip tip with report entry
            reports.append(TipPropagationReport(
                pid=int(tip.tip_id.pid), which=str(tip.tip_id.which),
                grew=False, reason=f"sif_fit_failed: {type(e).__name__}",
                Kc=float("nan"), KI=float("nan"), KII=float("nan"), keff=float("nan"), theta0=float("nan"),
            ))
            return None
