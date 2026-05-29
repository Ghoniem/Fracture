"""
network_growth.py
High-level orchestration for crack-network growth (inner/outer cycles),
supporting single polyline, multiple disconnected polylines, and junctioned
networks via open-polyline decomposition (split at deg != 2).

Core policy implemented:
- Extract open polylines (deg-1 to deg-1/deg>=3).
- Reference increment from longest polyline: ds_ref = f_ref * L_ref
  (defaults to propagator cfg.f_fixed).
- For each polyline, pick its representative driving tip as the deg-1 tip
  with maximum Keff on that polyline, and grow by ds = ds_ref*(Keff/Keff_ref).

This module does NOT modify solver, evaluator, or propagator internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import numpy as np


@dataclass
class GrowthLimits:
    max_outer_cycles: int = 50
    max_inner_steps: int = 50
    max_longest_component_length_mm: float = 200.0
    stop_if_no_growth: bool = True


@dataclass
class GrowthPolicy:
    f_ref: Optional[float] = None          # if None -> uses prop.cfg.f_fixed (or cfg.f0)
    keff_floor: float = 1e-12              # avoid divide-by-zero


@dataclass
class StepReport:
    outer_cycle: int
    inner_step: int
    global_step: int
    grew_any: bool
    n_polylines: int
    L_ref_mm: float
    ds_ref_mm: float
    keff_ref: float


@dataclass
class CycleReport:
    outer_cycle: int
    inner_steps_taken: int
    global_steps_taken: int
    grew_any: bool
    longest_component_length_mm: float
    # Optional metadata for cycle-by-cycle driving/plotting
    n_polylines: int = 0
    stop_reason: Optional[str] = None   # "length_limit" | "no_growth" | None


class NetworkGrower:
    def __init__(
        self,
        *,
        prop: Any,
        evaluator: Any,
        netops: Any,
        limits: GrowthLimits = GrowthLimits(),
        policy: GrowthPolicy = GrowthPolicy(),
    ):
        self.prop = prop
        self.evaluator = evaluator
        self.netops = netops
        self.limits = limits
        self.policy = policy

        self.global_step = 0

    # -------------------------
    # Public API
    # -------------------------
    def run(self, net0: Any) -> Tuple[Any, List[CycleReport], List[StepReport]]:
        net = net0
        cycle_reports: List[CycleReport] = []
        step_reports: List[StepReport] = []

        for oc in range(1, int(self.limits.max_outer_cycles) + 1):
            net, inner_taken, grew_any, step_rep = self._run_inner_cycle(net, oc)
            step_reports.extend(step_rep)

            L_longest_mm = self.longest_polyline_length_mm(net)
            cycle_reports.append(
                CycleReport(
                    outer_cycle=oc,
                    inner_steps_taken=inner_taken,
                    global_steps_taken=self.global_step,
                    grew_any=grew_any,
                    longest_component_length_mm=L_longest_mm,
                    n_polylines=int(step_rep[-1].n_polylines) if step_rep else 0,
                    stop_reason=None,
                )
            )

            # stop criteria
            if L_longest_mm >= float(self.limits.max_longest_component_length_mm):
                break
            if self.limits.stop_if_no_growth and (not grew_any):
                break

        return net, cycle_reports, step_reports


    def run_one_cycle(
        self,
        net0: Any,
        *,
        outer_cycle: int = 1,
        global_step_start: Optional[int] = None,
    ) -> Tuple[Any, CycleReport, List[StepReport]]:
        """Run exactly ONE outer cycle.

        Returns:
            net          : network after inner growth
            cycle_report : summary (includes stop_reason)
            step_reports : per-inner-step reports
        """
        if global_step_start is not None:
            self.global_step = int(global_step_start)

        net, inner_taken, grew_any, step_rep = self._run_inner_cycle(net0, int(outer_cycle))

        L_longest_mm = self.longest_polyline_length_mm(net)

        stop_reason: Optional[str] = None
        if L_longest_mm >= float(self.limits.max_longest_component_length_mm):
            stop_reason = "length_limit"
        elif self.limits.stop_if_no_growth and (not grew_any):
            stop_reason = "no_growth"

        n_polylines = int(step_rep[-1].n_polylines) if step_rep else 0

        cycle_report = CycleReport(
            outer_cycle=int(outer_cycle),
            inner_steps_taken=int(inner_taken),
            global_steps_taken=int(self.global_step),
            grew_any=bool(grew_any),
            longest_component_length_mm=float(L_longest_mm),
            n_polylines=n_polylines,
            stop_reason=stop_reason,
        )

        return net, cycle_report, step_rep


    @staticmethod
    def net_to_arrays(net: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Export network vertices/connectivity as arrays: V[:,]=[id,x,y], C[:,]=[v0,v1]."""
        V = np.array([[int(v.id), float(v.x), float(v.y)] for v in getattr(net, "vertices", [])], dtype=float)
        C = np.array([[int(e.v0), int(e.v1)] for e in getattr(net, "edges", [])], dtype=int)
        return V, C

    # -------------------------
    # Inner cycle
    # -------------------------
    def _run_inner_cycle(self, net: Any, outer_cycle: int) -> Tuple[Any, int, bool, List[StepReport]]:
        step_reports: List[StepReport] = []
        grew_any_cycle = False

        for inner in range(1, int(self.limits.max_inner_steps) + 1):
            grew_any, ds_ref, L_ref, keff_ref, n_polylines, net = self.grow_one_increment_scaled(net)
            self.global_step += 1
            grew_any_cycle = grew_any_cycle or grew_any

            step_reports.append(
                StepReport(
                    outer_cycle=outer_cycle,
                    inner_step=inner,
                    global_step=self.global_step,
                    grew_any=grew_any,
                    n_polylines=int(n_polylines),
                    L_ref_mm=float(L_ref) / self._mm(),
                    ds_ref_mm=float(ds_ref) / self._mm(),
                    keff_ref=float(keff_ref),
                )
            )

            if (not grew_any) and self.limits.stop_if_no_growth:
                break
            if self.longest_polyline_length_mm(net) >= float(self.limits.max_longest_component_length_mm):
                break

        return net, len(step_reports), grew_any_cycle, step_reports

    # -------------------------
    # Growth increment (K-scaled)
    # -------------------------
    def grow_one_increment_scaled(self, net: Any) -> Tuple[bool, float, float, float, int, Any]:
        deg = self.netops.degree_map(net)
        polylines = self.netops.extract_open_polylines(net) or []
        if not polylines:
            return False, 0.0, 0.0, 0.0, 0, net

        base_res = self.evaluator.solve_results(net)

        # vertex coordinate map
        vmap = {int(v.id): np.array([float(v.x), float(v.y)]) for v in net.vertices}

        poly_data: List[Dict[str, Any]] = []
        for pl in polylines:
            tips = self.netops.extract_deg1_tips(net, [pl], deg) or []
            if not tips:
                continue

            tip_keffs = []
            for tip in tips:
                tev = self.evaluator.eval_tip(base_res, tip)
                tip_keffs.append((tip, float(tev.keff), float(tev.theta)))

            tip, keff, theta = max(tip_keffs, key=lambda x: x[1])

            # polyline length along vids chain
            L = 0.0
            vids = list(pl.vids)
            for i in range(len(vids) - 1):
                a = int(vids[i]); b = int(vids[i + 1])
                L += float(np.linalg.norm(vmap[b] - vmap[a]))

            poly_data.append(dict(poly=pl, tip=tip, keff=keff, theta=theta, length=L))

        if not poly_data:
            return False, 0.0, 0.0, 0.0, len(polylines), net

        ref = max(poly_data, key=lambda d: d["length"])
        L_ref = float(ref["length"])
        keff_ref = max(float(ref["keff"]), float(self.policy.keff_floor))

        # Disk-radius step (preferred): Δa_ref = f_disk_radius * R, constant
        # in absolute terms regardless of network length. Fall back to legacy
        # L_total-fraction scaling when disk_radius_m is not set.
        cfg = getattr(self.prop, "cfg", None)
        disk_R = float(getattr(cfg, "disk_radius_m", 0.0)) if cfg is not None else 0.0
        if disk_R > 0.0:
            ds_ref = float(getattr(cfg, "f_disk_radius", 0.05)) * disk_R
        else:
            ds_ref = self._get_f_ref() * L_ref

        updater = self.prop._get_updater()
        net_new = net
        grew_any = False

        for d in poly_data:
            rel = float(d["keff"]) / keff_ref
            ds = ds_ref * rel
            if ds <= 0.0:
                continue
            net_new = updater.extend_tip(
                net_new,
                d["tip"],
                theta=float(d["theta"]),
                delta_a=float(ds),
            )
            grew_any = True

        return grew_any, ds_ref, L_ref, keff_ref, len(polylines), net_new

    # -------------------------
    # Metrics
    # -------------------------
    def longest_polyline_length_mm(self, net: Any) -> float:
        polylines = self.netops.extract_open_polylines(net) or []
        if not polylines:
            return 0.0
        vmap = {int(v.id): np.array([float(v.x), float(v.y)]) for v in net.vertices}
        Ls = []
        for pl in polylines:
            L = 0.0
            vids = list(pl.vids)
            for i in range(len(vids) - 1):
                a = int(vids[i]); b = int(vids[i + 1])
                L += float(np.linalg.norm(vmap[b] - vmap[a]))
            Ls.append(L)
        return float(max(Ls)) / self._mm()

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _mm() -> float:
        return 1e-3

    def _get_f_ref(self) -> float:
        if self.policy.f_ref is not None:
            return float(self.policy.f_ref)

        cfg = getattr(self.prop, "cfg", None)
        if cfg is None:
            return 0.1

        step_mode = str(getattr(cfg, "step_mode", "")).lower()
        if step_mode == "fixed_step" and hasattr(cfg, "f_fixed"):
            return float(getattr(cfg, "f_fixed"))
        return float(getattr(cfg, "f0", 0.1))
