from __future__ import annotations

import numpy as np

from utils.dce_results_v2 import DCEResultsV2


def _applied_tensor(applied) -> np.ndarray:
    """Return 2x2 applied stress tensor (Pa)."""
    if hasattr(applied, "tensor") and callable(getattr(applied, "tensor")):
        T = applied.tensor()
    else:
        T = np.asarray(applied, dtype=float)
    T = np.asarray(T, dtype=float)
    if T.shape != (2, 2):
        raise ValueError(f"Applied stress tensor must be 2x2, got {T.shape}")
    return T


class DCEResultsV3:
    """Results wrapper for multi-crack v3 solve."""

    def __init__(self, calc_multi, sol_multi: dict):
        self.calc = calc_multi
        self.sol = sol_multi
        self._per_cache: dict[int, DCEResultsV2] = {}

    @property
    def ncracks(self) -> int:
        return int(self.sol.get("ncracks", 0))

    def per_crack_results(self, i: int) -> DCEResultsV2:
        i = int(i)
        if i not in self._per_cache:
            calc_i = self.calc._single[i]
            sol_i = self.sol["per_crack"][i]
            self._per_cache[i] = DCEResultsV2(calc_i, sol_i)
        return self._per_cache[i]

    def stress_field_global(
        self,
        Xg: np.ndarray,
        Yg: np.ndarray,
        add_remote: bool = True,
        nq_stress: int = 6,
    ):
        """Return (sxx, syy, sxy) on a grid in GLOBAL coordinates (Pa)."""

        Xg = np.asarray(Xg, dtype=float)
        Yg = np.asarray(Yg, dtype=float)
        if Xg.shape != Yg.shape:
            raise ValueError("Xg and Yg must have the same shape")

        sxx = np.zeros_like(Xg)
        syy = np.zeros_like(Xg)
        sxy = np.zeros_like(Xg)

        for i in range(self.ncracks):
            calc_i = self.calc._single[i]
            sol_i = self.sol["per_crack"][i]

            center = np.asarray(calc_i.crack.center, dtype=float).reshape(2,)
            R = calc_i.R
            Rt = R.T

            Pl = (Rt @ (np.vstack([Xg.ravel(), Yg.ravel()]) - center.reshape(2, 1))).reshape(2, *Xg.shape)
            Xl, Yl = Pl[0], Pl[1]

            # local stresses at points
            s11, s22, s12 = calc_i.evaluate_stress_field_local(
                Xl, Yl, sol_i, add_remote=False, nq_stress=int(nq_stress)
            )

            # rotate local -> global
            s11 = np.asarray(s11, dtype=float)
            s22 = np.asarray(s22, dtype=float)
            s12 = np.asarray(s12, dtype=float)

            sxx_i = (R[0, 0] ** 2) * s11 + (R[0, 1] ** 2) * s22 + 2 * R[0, 0] * R[0, 1] * s12
            syy_i = (R[1, 0] ** 2) * s11 + (R[1, 1] ** 2) * s22 + 2 * R[1, 0] * R[1, 1] * s12
            sxy_i = (R[0, 0] * R[1, 0]) * s11 + (R[0, 1] * R[1, 1]) * s22 + (
                R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]
            ) * s12

            sxx += sxx_i
            syy += syy_i
            sxy += sxy_i

        if add_remote:
            sig = _applied_tensor(self.calc.applied)
            sxx += sig[0, 0]
            syy += sig[1, 1]
            sxy += sig[0, 1]

        return sxx, syy, sxy
