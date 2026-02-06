"""Candidate evaluator for propagation.

Uses the existing, battle-tested SIF pipeline in `fracture_utils.Uprocessor.SIF_cod`
via `DisplacementSIF.euclid_from_edge(...)`.

Adds robustness for large crack-growth simulations:
- Adaptive tip-fit retry ladder (rmax_frac/min_pts) on "window too small" failures.
- Optional ne_half escalation (re-solve) if the fit still fails.

This module is intentionally solver-stack aware, but keeps imports local to avoid
heavy import-time costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np


@dataclass
class TipEval:
    KI: float
    KII: float
    keff: float
    theta: float
    meta: Dict[str, Any]


class CandidateEvaluator:
    """Adapter between Usolver and Upropagation for solving + tip evaluation."""

    def __init__(
        self,
        *,
        material,
        applied,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        direction_law=None,
        sif_kwargs: Optional[Dict[str, Any]] = None,
        # retry ladders
        rmax_frac_ladder: Optional[List[float]] = None,
        min_pts_ladder: Optional[List[int]] = None,
        # mesh escalation
        enable_ne_half_escalation: bool = True,
        ne_half_growth: float = 1.5,
        ne_half_max: int = 240,
        ne_half_escalations_max: int = 2,
    ):
        self.material = material
        self.applied = applied
        self.solver_kwargs = dict(solver_kwargs or {})
        self.direction_law = direction_law

        # kwargs passed to DisplacementSIF.euclid_from_edge (rmax_frac, min_pts, two_term, etc.)
        self.sif_kwargs = dict(sif_kwargs or {})

        # retry ladders
        self.rmax_frac_ladder = list(rmax_frac_ladder or [0.08, 0.12, 0.18, 0.25])
        self.min_pts_ladder = list(min_pts_ladder or [10, 8, 6])

        # mesh escalation controls
        self.enable_ne_half_escalation = bool(enable_ne_half_escalation)
        self.ne_half_growth = float(ne_half_growth)
        self.ne_half_max = int(ne_half_max)
        self.ne_half_escalations_max = int(ne_half_escalations_max)

    # -------------------------
    # Solve
    # -------------------------
    def solve(self, network) -> Dict[str, Any]:
        """Solve elastic BVP for the given network and return solver dict with private handles."""
        from fracture_utils.Usolver.parametrization import DCENetworkStaticV4  # local import

        calc = DCENetworkStaticV4(self.material, network, self.applied)
        sol = calc.solve(**self.solver_kwargs)

        # Attach private handles for downstream evaluation
        sol["_calc"] = calc
        sol["_network"] = network
        return sol

    def solve_results(self, network):
        """Solve and return (res, sol) where res is DCEResultsNetworkV4."""
        sol = self.solve(network)
        res = self._to_results(sol)
        return res, sol

    # -------------------------
    # Tip evaluation
    # -------------------------
    def eval_tip(self, res_or_sol: Union[Dict[str, Any], Any], tip) -> TipEval:
        """
        Compute (KI,KII), keff, theta for a tip on an already-solved configuration.

        Accepts:
        - DCEResultsNetworkV4-like object (preferred), OR
        - solver dict returned by `solve(...)`.
        """
        res = self._ensure_results(res_or_sol)

        KI, KII, meta = self._euclid_tip_fit_with_retries(res, tip)
        KI = float(KI); KII = float(KII)
        keff = float(np.sqrt(KI * KI + KII * KII))
        theta = float(self.direction_law.theta(KI, KII) if self.direction_law is not None else 0.0)
        return TipEval(KI=KI, KII=KII, keff=keff, theta=theta, meta=dict(meta or {}))

    # -------------------------
    # Robust euclid tip-fit
    # -------------------------
    def _euclid_tip_fit_with_retries(self, res, tip) -> Tuple[float, float, Dict[str, Any]]:
        """
        Try euclid tip-fit with an adaptive ladder over (rmax_frac, min_pts).

        Raises the last exception if all ladder attempts fail.
        """
        from fracture_utils.Uprocessor.SIF_cod import DisplacementSIF

        calc = getattr(res, "calc", None)
        if calc is None:
            raise RuntimeError("Results object does not expose .calc")

        network = getattr(calc, "net", None) or getattr(calc, "network", None)
        if network is None:
            # fall back to solver dict private handle
            sol = getattr(res, "sol", None)
            if isinstance(sol, dict):
                network = sol.get("_network", None)
        if network is None:
            raise RuntimeError("Unable to locate network from results object.")

        v_tip = self._tip_vertex_id(tip)
        tip_xy = np.asarray(getattr(tip, "x_tip", None), float).reshape(2,)
        edge_index = self._incident_edge_index(network, v_tip)

        a_fit = float(getattr(tip, "total_length", 0.0)) or float(self._total_crack_length(network))

        base_kwargs = dict(self.sif_kwargs)
        # Use current defaults if provided; still allow ladder escalation above them.
        r0 = float(base_kwargs.pop("rmax_frac", self.rmax_frac_ladder[0]))
        m0 = int(base_kwargs.pop("min_pts", self.min_pts_ladder[0]))

        r_ladder = [r0] + [r for r in self.rmax_frac_ladder if r != r0]
        m_ladder = [m0] + [m for m in self.min_pts_ladder if m != m0]

        last_err: Optional[Exception] = None

        for rmax_frac in r_ladder:
            for min_pts in m_ladder:
                try:
                    KI, KII, _, _, meta = DisplacementSIF.euclid_from_edge(
                        res,
                        edge_index=int(edge_index),
                        tip_xy=tip_xy,
                        a_fit=float(a_fit),
                        material=calc.material,
                        rotate=False,
                        rmax_frac=float(rmax_frac),
                        min_pts=int(min_pts),
                        **base_kwargs,
                    )
                    return float(KI), float(KII), dict(meta or {})
                except Exception as e:
                    last_err = e
                    if not self._is_window_too_small(e):
                        # For other failures, don't keep laddering blindly.
                        raise
                    # else: continue ladder

        # Exhausted ladder
        assert last_err is not None
        raise last_err

    @staticmethod
    def _is_window_too_small(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("window too small" in msg) or ("try increasing rmax_frac" in msg) or ("rr.size" in msg)

    # -------------------------
    # Utilities: results conversion + geometry helpers
    # -------------------------
    def _ensure_results(self, res_or_sol):
        if isinstance(res_or_sol, dict):
            return self._to_results(res_or_sol)
        return res_or_sol

    @staticmethod
    def _to_results(sol: Dict[str, Any]):
        # Preferred: your repo defines this in Uprocessor.results
        try:
            from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
            calc = sol.get("_calc", None)
            if calc is None:
                raise RuntimeError("Solution dict missing _calc. Use CandidateEvaluator.solve(network).")
            return DCEResultsNetworkV4(calc, sol)
        except Exception:
            # If results wrapper lives elsewhere, raise a clear error.
            raise ImportError("DCEResultsNetworkV4 not found. Expected in fracture_utils.Uprocessor.results.")

    @staticmethod
    def _tip_vertex_id(tip) -> int:
        for name in ("v_tip", "vid", "vertex_id"):
            if hasattr(tip, name):
                return int(getattr(tip, name))
        raise ValueError("Tip object does not expose v_tip/vid/vertex_id.")

    @staticmethod
    def _incident_edge_index(network, v_tip: int) -> int:
        """Pick a network edge index incident to v_tip. Works for simple chains."""
        v = network.V(int(v_tip))
        if len(v.edges) < 1:
            raise ValueError(f"tip vertex {v_tip} has no incident edges.")
        eid = int(v.edges[0])
        # Network edges list order is used as edge_index in many of your utilities.
        for i, e in enumerate(network.edges):
            if int(e.id) == eid:
                return int(i)
        # fallback: return 0
        return 0

    @staticmethod
    def _total_crack_length(network) -> float:
        L = 0.0
        for e in network.edges:
            p0 = network.vertex_coords(int(e.v0))
            p1 = network.vertex_coords(int(e.v1))
            L += float(np.hypot(*(p1 - p0)))
        return float(L)

    # -------------------------
    # Ne_half escalation helper (used by propagator)
    # -------------------------
    def escalate_ne_half(self) -> bool:
        """
        Increase ne_half in solver_kwargs (in-place) if possible.

        Returns True if increased, False if already at max or ne_half missing.
        """
        if "ne_half" not in self.solver_kwargs:
            return False
        cur = int(self.solver_kwargs["ne_half"])
        if cur >= self.ne_half_max:
            return False
        new = int(np.ceil(cur * self.ne_half_growth))
        new = min(new, self.ne_half_max)
        if new <= cur:
            new = min(cur + 10, self.ne_half_max)
        self.solver_kwargs["ne_half"] = int(new)
        return True
