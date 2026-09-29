# fracture_utils/Upropagation/evaluate.py
# CandidateEvaluator providing solve(network)->sol_dict and eval_tip(sol_or_res, tip).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class TipEval:
    KI: float
    KII: float
    keff: float
    theta: float
    meta: Dict[str, Any]


def _import_solver_stack():
    # Calc
    DCENetworkStaticV4 = None
    last = None
    for mod, name in [
        ("fracture_utils.Usolver.parametrization", "DCENetworkStaticV4"),
        ("fracture_utils.Usolver.parametrization", "DCENetworkStatic"),
    ]:
        try:
            m = __import__(mod, fromlist=[name])
            DCENetworkStaticV4 = getattr(m, name)
            break
        except Exception as e:
            last = e
    if DCENetworkStaticV4 is None:
        raise ImportError(f"Could not import DCENetworkStaticV4 from Usolver.parametrization. Last error: {last}")

    # Results
    DCEResultsNetworkV4 = None
    last = None
    for mod, name in [
        ("fracture_utils.Uprocessor.results", "DCEResultsNetworkV4"),
        ("fracture_utils.Uprocessor.results_v4", "DCEResultsNetworkV4"),
    ]:
        try:
            m = __import__(mod, fromlist=[name])
            DCEResultsNetworkV4 = getattr(m, name)
            break
        except Exception as e:
            last = e
    if DCEResultsNetworkV4 is None:
        raise ImportError(f"Could not import DCEResultsNetworkV4 from Uprocessor.results. Last error: {last}")

    return DCENetworkStaticV4, DCEResultsNetworkV4


class CandidateEvaluator:
    def __init__(
        self,
        *,
        material: Any,
        applied: Any,
        solver_kwargs: Dict[str, Any],
        direction_law: Any,
        rmax_frac: float = 0.12,
        min_pts: int = 10,
        two_term: bool = False,
        enable_n_crack_elements_escalation: bool = False,
        n_crack_elements_max: int = 240,
    ):
        self.material = material
        self.applied = applied
        self.solver_kwargs = dict(solver_kwargs)
        self.direction_law = direction_law

        self.rmax_frac = float(rmax_frac)
        self.min_pts = int(min_pts)
        self.two_term = bool(two_term)

        self.enable_n_crack_elements_escalation = bool(enable_n_crack_elements_escalation)
        self.n_crack_elements_max = int(n_crack_elements_max)

        self._DCENetworkStaticV4 = None
        self._DCEResultsNetworkV4 = None

    def _stack(self):
        if self._DCENetworkStaticV4 is None or self._DCEResultsNetworkV4 is None:
            self._DCENetworkStaticV4, self._DCEResultsNetworkV4 = _import_solver_stack()
        return self._DCENetworkStaticV4, self._DCEResultsNetworkV4

    # ---- Public API expected by StepController / Propagator

    def solve(self, network: Any) -> Dict[str, Any]:
        """Return solver dict and attach calc handle under '_calc' for downstream reconstruction."""
        DCENetworkStaticV4, _DCEResultsNetworkV4 = self._stack()
        calc = DCENetworkStaticV4(self.material, network, self.applied)
        sol = calc.solve(**self.solver_kwargs)
        if isinstance(sol, dict):
            sol["_calc"] = calc
        return sol

    def solve_results(self, network: Any):
        """Convenience: return DCEResultsNetworkV4."""
        DCENetworkStaticV4, DCEResultsNetworkV4 = self._stack()
        calc = DCENetworkStaticV4(self.material, network, self.applied)
        sol = calc.solve(**self.solver_kwargs)
        return DCEResultsNetworkV4(calc, sol)

    def _ensure_results(self, res_or_sol: Any):
        if hasattr(res_or_sol, "calc"):
            return res_or_sol
        if isinstance(res_or_sol, dict) and "_calc" in res_or_sol:
            _DCENetworkStaticV4, DCEResultsNetworkV4 = self._stack()
            calc = res_or_sol["_calc"]
            return DCEResultsNetworkV4(calc, res_or_sol)
        return res_or_sol

    def _edge_index_for_tip(self, network: Any, tip) -> int:
        v_tip = int(getattr(tip, "v_tip", getattr(tip, "vid", -1)))
        for i, e in enumerate(getattr(network, "edges", [])):
            if int(getattr(e, "v0")) == v_tip or int(getattr(e, "v1")) == v_tip:
                return int(i)
        raise RuntimeError(f"Could not find an edge incident to tip vertex {v_tip}.")

    def _euclid_tip_fit(self, res: Any, tip) -> Tuple[float, float, Dict[str, Any]]:
        from fracture_utils.Uprocessor.SIF_cod import DisplacementSIF

        calc = getattr(res, "calc", None)
        if calc is None:
            raise RuntimeError("Results object does not expose .calc")

        network = getattr(calc, "net", None) or getattr(calc, "network", None)
        if network is None:
            raise RuntimeError("res.calc does not expose .net or .network")

        edge_index = self._edge_index_for_tip(network, tip)
        tip_xy = np.asarray(getattr(tip, "x_tip"), float).reshape(2,)
        a_fit = float(getattr(tip, "total_length", 0.0)) or 1.0

        KI, KII, *_rest = DisplacementSIF.euclid_from_edge(
            res,
            edge_index=int(edge_index),
            tip_xy=tip_xy,
            a_fit=float(a_fit),
            material=self.material,
            rotate=False,
            rmax_frac=float(self.rmax_frac),
            min_pts=int(self.min_pts),
            two_term=bool(self.two_term),
        )
        meta = _rest[-1] if _rest else {}
        return float(KI), float(KII), dict(meta or {})

    def eval_tip(self, res_or_sol: Any, tip) -> TipEval:
        res = self._ensure_results(res_or_sol)

        KI = KII = 0.0
        meta: Dict[str, Any] = {}
        last_err: Optional[Exception] = None

        n_try = 1
        if self.enable_n_crack_elements_escalation:
            n_try = 4

        # n_crack_elements escalation is per-tip: temporarily bump solver_kwargs while retrying,
        # then restore. Without restore, every subsequent tip evaluation would inherit
        # the escalated value and silently consume far more memory / runtime.
        original_n_crack_elements = self.solver_kwargs.get("n_crack_elements", None)
        try:
            for _ in range(n_try):
                try:
                    KI, KII, meta = self._euclid_tip_fit(res, tip)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if (not self.enable_n_crack_elements_escalation) or ("window too small" not in msg):
                        break

                    n_crack_elements = int(self.solver_kwargs.get("n_crack_elements", 60))
                    n_crack_elements_new = min(self.n_crack_elements_max, max(n_crack_elements + 10, 2 * n_crack_elements))
                    if n_crack_elements_new <= n_crack_elements:
                        break
                    self.solver_kwargs["n_crack_elements"] = n_crack_elements_new

                    calc = getattr(res, "calc", None)
                    network = getattr(calc, "net", None) or getattr(calc, "network", None)
                    if network is None:
                        break
                    res = self.solve_results(network)
        finally:
            if original_n_crack_elements is None:
                self.solver_kwargs.pop("n_crack_elements", None)
            else:
                self.solver_kwargs["n_crack_elements"] = original_n_crack_elements

        if last_err is not None:
            raise last_err

        KI = float(KI); KII = float(KII)
        keff = float(np.hypot(KI, KII))

        # At the v_start tip the SIF extraction frame is left-handed
        # (ex outward, but ey kept along the polyline-direction normal so
        # KI stays positive at both tips). MTS assumes a right-handed
        # (ex outward, ey = R90_CCW(ex)) frame, so we flip KII here to
        # convert the start-tip SIFs to that convention before computing
        # the kink angle. With this flip the returned theta is the kink
        # CCW from the outward tangent at both tips, and extend_tip can
        # apply it uniformly (no separate d -> -d at the start).
        which = str(getattr(getattr(tip, "tip_id", None), "which", "")).lower()
        KII_for_theta = -KII if which == "start" else KII
        theta = float(self.direction_law.theta(KI, KII_for_theta) if self.direction_law is not None else 0.0)

        return TipEval(KI=KI, KII=KII, keff=keff, theta=theta, meta=meta)
