"""Material and loading definitions for DCE v4 polyline solver.

Update (Coupled VCM–BEM, uncoupled mode):
- AppliedStress now supports either a constant stress tensor (legacy) OR a spatially-varying stress
  field via a callable sigma_func(X)->(N,2,2) in Pa. This enables driving an infinite-medium crack
  solve with a BEM-computed stress field without coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Material:
    E: float
    nu: float
    plane_stress: bool = False

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self) -> float:
        # Muskhelishvili kappa
        if self.plane_stress:
            return (3.0 - self.nu) / (1.0 + self.nu)
        return 3.0 - 4.0 * self.nu


SigmaFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AppliedStress:
    """
    Applied (remote/external) stress definition.

    Two modes (backward compatible):
      1) Constant stress: provide sigma_xx, sigma_yy, sigma_xy (legacy behavior)
      2) Spatial stress field: provide sigma_func(X)->(N,2,2) in Pa, where X is (N,2)

    NOTE:
      - tensor() is only valid for the constant-stress case.
      - Use tensor_at(X) universally (works for both).
    """
    sigma_xx: Optional[float] = None
    sigma_yy: Optional[float] = None
    sigma_xy: Optional[float] = None
    sigma_func: Optional[SigmaFunc] = None

    def tensor(self) -> np.ndarray:
        """Return the constant 2x2 stress tensor (Pa)."""
        if self.sigma_func is not None:
            raise RuntimeError("AppliedStress.tensor() is undefined for spatial sigma_func. Use tensor_at(X).")

        if self.sigma_xx is None or self.sigma_yy is None or self.sigma_xy is None:
            raise ValueError("Constant AppliedStress requires sigma_xx, sigma_yy, sigma_xy.")

        return np.array(
            [[float(self.sigma_xx), float(self.sigma_xy)],
             [float(self.sigma_xy), float(self.sigma_yy)]],
            dtype=float,
        )

    def tensor_at(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate stress tensor at points X.

        Args:
          X: (N,2) points

        Returns:
          S: (N,2,2) stress tensors in Pa
        """
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("AppliedStress.tensor_at expects X with shape (N,2).")

        if self.sigma_func is None:
            S = self.tensor()
            return np.repeat(S[None, :, :], X.shape[0], axis=0)

        S = self.sigma_func(X)
        S = np.asarray(S, float)
        if S.shape != (X.shape[0], 2, 2):
            raise ValueError("sigma_func must return array with shape (N,2,2).")
        return S
