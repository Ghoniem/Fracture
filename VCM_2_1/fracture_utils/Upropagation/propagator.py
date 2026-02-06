"""High-level propagation orchestrator.

This version:
- Supports step_mode in PropagationConfig via StepController (fixed/adaptive).
- Delegates ALL network topology operations (degrees, polylines, tip extraction) to
  Upropagation.network_ops, which prefers the user's Ugenerator package if available.
- Avoids import-time failures by keeping optional imports local.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from .config import PropagationConfig
from .toughness import ToughnessField
from .direction import DirectionLaw
from .step_control import StepController
from .geometry_update import GeometryUpdater
from .results import PropagationResult, TipPropagationReport
from .network_ops import get_network_ops, PolylinePath

if TYPE_CHECKING:
    from fracture_utils.Usolver.network import CrackNetworkV4
    from .tip_state import TipState


@dataclass
class _SolveBundle:
    res: object
    sol: dict
    total_length: float
    polylines: List[PolylinePath]


class CrackPropagator:
    """Orchestrates one propagation increment for a network (current v1: open polylines)."""

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
        prefer_ugenerator: bool = True,
    ):
        self.cfg = cfg
        self.evaluator = evaluator
        self.toughness = toughness
        self.direction_law = direction_law
        self.step_controller = step_controller or StepController(cfg)
        self.updater = updater or GeometryUpdater()
        self.remesh_policy = remesh_policy
        self.netops = get_network_ops(prefer_ugenerator=prefer_ugenerator)

    # -------------------------
    # Main entry point
    # -------------------------
    def grow_one_increment(self, network: "CrackNetworkV4") -> PropagationResult:
        """Run a single growth increment: decide and apply tip extensions."""
        bundle = self._solve_bundle_with_escalation(network, escalations=0)

        # topology via backend
        deg = self.netops.degree_map(network)
        polylines = bundle.polylines

        # tip extraction via backend (returns TipState objects)
        tips = self.netops.extract_deg1_tips(network, polylines, deg)

        reports: List[TipPropagationReport] = []
        net_work = network

        def build_candidate(net_base, tip: "TipState", theta: float, delta_a: float):
            return self.updater.extend_tip(net_base, tip, theta=theta, delta_a=delta_a)

        for tip in tips:
            tev0 = self._safe_eval_tip(bundle, tip, reports)
            if tev0 is None:
                continue

            Kc = float(self.toughness.Kc(float(tip.x_tip[0]), float(tip.x_tip[1])))
            if float(tev0.keff) <= Kc:
                reports.append(
                    TipPropagationReport(
                        pid=int(tip.tip_id.pid),
                        which=str(tip.tip_id.which),
                        grew=False,
                        reason="below_toughness",
                        KI=float(tev0.KI),
                        KII=float(tev0.KII),
                        keff=float(tev0.keff),
                        theta=float(tev0.theta),
                        delta_a=0.0,
                    )
                )
                continue

            # Step choice (fixed vs adaptive is inside StepController)
            decision = self.step_controller.choose_step(
                tip=tip,
                tev0=tev0,
                total_length=float(bundle.total_length),
                build_candidate=build_candidate,
                evaluator=self.evaluator,
                toughness=self.toughness,
                direction_law=self.direction_law,
                base_network=net_work,
                reports=reports,
            )

            if not decision.accepted:
                reports.append(
                    TipPropagationReport(
                        pid=int(tip.tip_id.pid),
                        which=str(tip.tip_id.which),
                        grew=False,
                        reason=str(decision.reason or "rejected"),
                        KI=float(tev0.KI),
                        KII=float(tev0.KII),
                        keff=float(tev0.keff),
                        theta=float(tev0.theta),
                        delta_a=float(getattr(decision, "delta_a", 0.0) or 0.0),
                    )
                )
                continue

            # Apply chosen growth
            net_work = build_candidate(net_work, tip, float(decision.theta), float(decision.delta_a))
            reports.append(
                TipPropagationReport(
                    pid=int(tip.tip_id.pid),
                    which=str(tip.tip_id.which),
                    grew=True,
                    reason="grew",
                    KI=float(getattr(decision, "KI", tev0.KI)),
                    KII=float(getattr(decision, "KII", tev0.KII)),
                    keff=float(getattr(decision, "keff", tev0.keff)),
                    theta=float(decision.theta),
                    delta_a=float(decision.delta_a),
                )
            )

        # Optional remesh after accepted growth
        net_new = net_work
        if self.remesh_policy is not None and net_new is not network:
            try:
                net_new = self.remesh_policy.remesh_network(net_new, ne_half=int(self.evaluator.solver_kwargs.get("ne_half", 60)), cfg=self.cfg)
            except Exception:
                # remesh is optional; never break propagation
                net_new = net_work

        return PropagationResult(network_old=network, network_new=net_new, reports=reports)

    # -------------------------
    # Solve + bundle helpers
    # -------------------------
    def _solve_bundle_with_escalation(self, network: "CrackNetworkV4", escalations: int) -> _SolveBundle:
        # Polylines and length via backend
        polylines = self.netops.extract_open_polylines(network)
        total_length = 0.0
        for pl in polylines:
            for a, b in zip(pl.vids[:-1], pl.vids[1:]):
                p0 = network.vertex_coords(int(a))
                p1 = network.vertex_coords(int(b))
                total_length += float(np.hypot(*(p1 - p0)))

        # Solve once using evaluator
        res, sol = self.evaluator.solve_results(network)
        return _SolveBundle(res=res, sol=sol, total_length=float(total_length), polylines=polylines)

    def _safe_eval_tip(self, bundle: _SolveBundle, tip: "TipState", reports: List[TipPropagationReport]):
        try:
            tev0 = self.evaluator.eval_tip(bundle.res, tip)
            return tev0
        except Exception as e:
            reports.append(
                TipPropagationReport(
                    pid=int(tip.tip_id.pid),
                    which=str(tip.tip_id.which),
                    grew=False,
                    reason=f"eval_failed: {e}",
                    KI=np.nan,
                    KII=np.nan,
                    keff=np.nan,
                    theta=np.nan,
                    delta_a=0.0,
                )
            )
            return None
