# fracture_utils/Upropagation/propagator.py
# CrackPropagator: fixed_step works for arbitrary networks (multiple open polylines, multiple deg-1 tips).
# Key property: "two-phase" update (evaluate all tips on baseline solution, then apply updates),
# which avoids order dependence when multiple tips grow in the same increment.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple
import numpy as np

from .tip_state import TipState


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
        max_kink_deg: float = 0.0,
        initial_vertex_ids: Optional[Iterable[int]] = None,
        snapped_vertex_ids: Optional[Iterable[int]] = None,
    ):
        self.cfg = cfg
        self.evaluator = evaluator
        self.toughness = toughness
        self.direction_law = direction_law
        self.remesh_policy = remesh_policy
        self.updater = updater

        # Soft segment-to-segment kink clamp. theta from MTS is measured
        # relative to the local tip tangent (= direction of the immediately
        # preceding segment), so clamping |theta| <= max_kink_deg is a
        # direct clamp on the segment-to-segment direction change.
        # max_kink_deg <= 0 disables the clamp.
        # Tips whose v_tip is in initial_vertex_ids are on their first
        # emission (no previous *new* segment to compare to) and bypass
        # the clamp; subsequent emissions always clamp.
        self.max_kink_deg = float(max_kink_deg)
        self._initial_vertex_ids: Set[int] = (
            set(int(v) for v in initial_vertex_ids) if initial_vertex_ids is not None else set()
        )

        # Boundary-snapped tips: vids of deg-1 tips that have crossed the
        # disk boundary at any point in the run. Their positions are
        # projected radially onto r=R and growth_force is forced to 0
        # for every subsequent increment (permanent arrest at the
        # boundary). Caller threads this set across outer cycles so the
        # arrest persists when a fresh propagator is built each cycle.
        self._snapped_vertex_ids: Set[int] = (
            set(int(v) for v in snapped_vertex_ids) if snapped_vertex_ids is not None else set()
        )

        # Step controller is only used for adaptive_step (it's a no-op factory output otherwise).
        # The factory in step_control.make_step_controller picks Fixed/Adaptive based on cfg.step_mode.
        self.step_controller = None
        try:
            from .step_control import make_step_controller
            self.step_controller = make_step_controller(cfg)
        except Exception:
            self.step_controller = None

    def _disk_radius(self) -> float:
        return float(getattr(self.cfg, "disk_radius_m", 0.0) or 0.0)

    def _tip_outside_disk(self, tip: Any) -> bool:
        """True if the tip lies strictly outside the disk r > R.

        Brazilian-disk geometry: disk centered at origin, radius
        cfg.disk_radius_m. Returns False (no-op) when disk_radius_m <= 0.
        """
        R = self._disk_radius()
        if R <= 0.0:
            return False
        x = np.asarray(getattr(tip, "x_tip"), float).reshape(2,)
        return float(np.hypot(x[0], x[1])) > R

    def _snap_tip_to_boundary(self, network: Any, tip: Any) -> Any:
        """Project the tip's vertex radially onto the disk boundary r=R.

        Modifies a copy of the network: the snapped vertex now lies exactly
        at r=R along the same radial direction as the original (outside)
        tip. The new network is returned; the caller swaps it in. Disk
        is centered at origin (matches the BEM solver convention).
        """
        from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4

        R = self._disk_radius()
        v_tip = int(getattr(tip, "v_tip"))
        x = np.asarray(getattr(tip, "x_tip"), float).reshape(2,)
        r = float(np.hypot(x[0], x[1]))
        if r <= 0.0 or R <= 0.0:
            return network
        scale = R / r
        x_snap = (float(x[0]) * scale, float(x[1]) * scale)

        V_new = []
        for v in network.vertices:
            if int(v.id) == v_tip:
                V_new.append(VertexV4(v.id, x_snap[0], x_snap[1],
                                      list(v.edges), role=v.role))
            else:
                V_new.append(v)
        return CrackNetworkV4(vertices=V_new, edges=list(network.edges),
                              Nv_max=network.Nv_max)

    @property
    def snapped_vertex_ids(self) -> Set[int]:
        return set(self._snapped_vertex_ids)

    def _clamp_segment_kink(self, tip: Any, theta: float) -> float:
        """Clamp theta to +/- max_kink_deg, except on a tip's first emission.

        First emission := tip.v_tip is in self._initial_vertex_ids. After
        a tip emits a new edge, the new tip vertex has a fresh id (not in
        the initial set), and subsequent calls clamp.
        """
        if self.max_kink_deg <= 0.0:
            return float(theta)
        v_tip = int(getattr(tip, "v_tip", -1))
        if v_tip in self._initial_vertex_ids:
            return float(theta)
        max_rad = float(self.max_kink_deg) * np.pi / 180.0
        if abs(float(theta)) > max_rad:
            return float(max_rad if theta > 0.0 else -max_rad)
        return float(theta)

    def _get_updater(self):
        if self.updater is not None:
            return self.updater
        from .geometry_update import GeometryUpdater
        return GeometryUpdater()

    def _step_mode(self) -> str:
        return str(getattr(self.cfg, "step_mode", "fixed_step")).lower()

    def _simultaneous(self) -> bool:
        return bool(getattr(self.cfg, "simultaneous_tip_growth", True))

    def _fixed_step_delta_a(self, Ltot: float) -> float:
        # Disk-radius step (preferred for the Brazilian-disk pipeline):
        # Δa = f_disk_radius * R, constant in absolute terms regardless of
        # how the polyline grows. Falls back to legacy Δa = f_fixed * L_total
        # when disk_radius_m is not set.
        disk_R = float(getattr(self.cfg, "disk_radius_m", 0.0) or 0.0)
        if disk_R > 0.0:
            f = float(getattr(self.cfg, "f_disk_radius", 0.05))
            return float(f * disk_R)
        f = float(getattr(self.cfg, "f_fixed", getattr(self.cfg, "f0", 0.10)))
        return float(f * Ltot)

    def grow_one_increment(self, network: Any) -> PropagationResult:
        from .network_ops import get_netops

        # ---- Pre-pass: snap any deg-1 tip that has crossed outside the
        # disk to r=R, record its vid in the persistent snapped set, and
        # re-extract tips so the solve and evaluations use the snapped
        # geometry. Tips whose vid is already in the snapped set are also
        # treated as arrested (growth_force pinned to 0).
        netops = get_netops()
        deg = netops.degree_map(network)
        polylines = netops.extract_open_polylines(network)
        tips = netops.extract_deg1_tips(network, polylines, deg) or []

        if self._disk_radius() > 0.0 and len(tips) > 0:
            snapped_now = False
            for tip in tips:
                if self._tip_outside_disk(tip):
                    network = self._snap_tip_to_boundary(network, tip)
                    self._snapped_vertex_ids.add(int(getattr(tip, "v_tip", -1)))
                    snapped_now = True
            if snapped_now:
                deg = netops.degree_map(network)
                polylines = netops.extract_open_polylines(network)
                tips = netops.extract_deg1_tips(network, polylines, deg) or []

        if len(tips) == 0:
            return PropagationResult(network_new=network, reports=[])

        updater = self._get_updater()
        reports: List[TipPropagationReport] = []

        # ---- Solve baseline once (on the snapped network if any tips were
        # arrested above)
        base_sol = self.evaluator.solve(network)

        step_mode = self._step_mode()
        delta_a_tol = float(getattr(self.cfg, "delta_a_min", 0.0))

        # ---- Phase A.1: evaluate every deg-1 tip against the SAME baseline.
        # Defer Δa: we need K_eff for every tip before scaling.
        evals: List[dict] = []
        Ltot_max = 0.0
        for tip in tips:
            Ltot = float(getattr(tip, "total_length", 0.0))
            Ltot_max = max(Ltot_max, Ltot)
            vid = int(getattr(tip, "v_tip", -1))

            # Arrested tips: previously snapped to the disk boundary. They
            # are permanent gf=0 — reported as "snapped_at_boundary" so
            # post-processing can tell them apart from below_toughness /
            # below_tolerance.
            if vid in self._snapped_vertex_ids:
                evals.append(dict(
                    tip=tip, Ltot=Ltot,
                    KI=0.0, KII=0.0, keff=0.0, theta=0.0, Kc=0.0,
                    growth_force=0.0, reason="snapped_at_boundary",
                ))
                continue

            tev0 = self.evaluator.eval_tip(base_sol, tip)
            KI0 = float(tev0.KI); KII0 = float(tev0.KII)
            keff0 = float(tev0.keff); theta0 = float(tev0.theta)

            try:
                Kc = float(self.toughness.Kc(float(tip.x_tip[0]), float(tip.x_tip[1])))
            except Exception:
                Kc = 0.0

            growth_force = max(0.0, keff0 - Kc)
            evals.append(dict(
                tip=tip, Ltot=Ltot,
                KI=KI0, KII=KII0, keff=keff0, theta=theta0, Kc=Kc,
                growth_force=growth_force,
                reason=("" if growth_force > 0.0 else "below_toughness"),
            ))

        # ---- Phase A.2: build per-tip decisions.
        # tuple: (tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason)
        decisions: List[Tuple[Any, float, float, float, float, float, float, float, str]] = []

        if step_mode == "adaptive_step" and self.step_controller is not None:
            # Legacy: each tip resolves its own step independently via the
            # adaptive step controller. No global K-scaling here.
            for e in evals:
                tip = e["tip"]; Ltot = e["Ltot"]
                theta0 = e["theta"]; keff0 = e["keff"]
                KI0 = e["KI"]; KII0 = e["KII"]; Kc = e["Kc"]
                reason = e["reason"]

                if e["growth_force"] <= 0.0:
                    decisions.append((tip, 0.0, theta0, keff0, KI0, KII0, Kc, Ltot, reason))
                    continue

                def build_candidate(net_base, _tip, _theta, _delta_a):
                    new_net = updater.extend_tip(
                        net_base, _tip, theta=float(_theta), delta_a=float(_delta_a)
                    )
                    new_vid = int(max(int(v.id) for v in new_net.vertices))
                    new_v = next(v for v in new_net.vertices if int(v.id) == new_vid)
                    new_xy = np.array([float(new_v.x), float(new_v.y)], float)
                    other_eid = int(new_v.edges[0])
                    other_e = next(ee for ee in new_net.edges if int(ee.id) == other_eid)
                    other_vid = int(other_e.v1) if int(other_e.v0) == new_vid else int(other_e.v0)
                    other_v = next(v for v in new_net.vertices if int(v.id) == other_vid)
                    t = new_xy - np.array([float(other_v.x), float(other_v.y)], float)
                    tn = float(np.hypot(t[0], t[1]))
                    new_that = t / tn if tn > 0.0 else np.array([1.0, 0.0], float)
                    new_tot = float(getattr(_tip, "total_length", 0.0)) + float(_delta_a)
                    new_tip = TipState(
                        tip_id=_tip.tip_id,
                        v_tip=new_vid,
                        x_tip=new_xy,
                        t_hat=new_that,
                        total_length=new_tot,
                    )
                    return new_net, new_tip

                theta = theta0
                delta_a = 0.0
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
                        delta_a = self._fixed_step_delta_a(Ltot)
                        theta = theta0
                except Exception as ex:
                    reason = f"adaptive_failed:{ex}"
                    delta_a = self._fixed_step_delta_a(Ltot)
                    theta = theta0

                theta = self._clamp_segment_kink(tip, theta)

                # delta_a_min now acts as an *elimination* threshold: kinks
                # below tolerance are dropped (not bumped up).
                if delta_a < delta_a_tol:
                    decisions.append((tip, 0.0, theta, keff0, KI0, KII0, Kc, Ltot,
                                      reason or "below_tolerance"))
                    continue

                decisions.append((tip, delta_a, theta, keff0, KI0, KII0, Kc, Ltot, reason))
        else:
            # New global K-scaled fixed-step logic:
            #   1. Across all deg-1 tips, growth_force_i = max(K_eff_i - Kc_i, 0).
            #   2. Reference step ds_ref = f_disk_radius * R (constant in absolute
            #      terms) with legacy L_total fallback when disk_radius_m is unset.
            #   3. Each above-threshold tip grows by ds_ref * growth_force_i / max_gf.
            #      The dominant tip (max K_eff) gets exactly ds_ref; weaker tips
            #      get a Paris-law-style scaled fraction.
            #   4. Tips with the scaled Δa below cfg.delta_a_min are eliminated
            #      (skipped entirely rather than bumped up).
            max_gf = max((float(e["growth_force"]) for e in evals), default=0.0)
            ds_ref = self._fixed_step_delta_a(Ltot_max)

            for e in evals:
                tip = e["tip"]; Ltot = e["Ltot"]
                theta = self._clamp_segment_kink(tip, float(e["theta"]))
                keff0 = e["keff"]; KI0 = e["KI"]; KII0 = e["KII"]; Kc = e["Kc"]
                reason = e["reason"]

                if e["growth_force"] <= 0.0 or max_gf <= 0.0:
                    decisions.append((tip, 0.0, theta, keff0, KI0, KII0, Kc, Ltot,
                                      reason or "below_toughness"))
                    continue

                rel = float(e["growth_force"]) / float(max_gf)
                delta_a = float(ds_ref) * rel

                if delta_a < delta_a_tol:
                    decisions.append((tip, 0.0, theta, keff0, KI0, KII0, Kc, Ltot,
                                      "below_tolerance"))
                    continue

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

        # Optional remesh. All policies accept the same kwargs:
        #     remesh_network(network, *, n_crack_elements=...)
        # n_crack_elements is read from the evaluator's solver_kwargs when available; policies
        # that don't need it (e.g. GlobalRemeshPolicy) simply ignore the value.
        if self.remesh_policy is not None:
            sk = getattr(self.evaluator, "solver_kwargs", None) or {}
            n_crack_elements = sk.get("n_crack_elements", None)
            n_crack_elements = int(n_crack_elements) if n_crack_elements is not None else None
            net_new = self.remesh_policy.remesh_network(net_new, n_crack_elements=n_crack_elements)

        return PropagationResult(network_new=net_new, reports=reports)
