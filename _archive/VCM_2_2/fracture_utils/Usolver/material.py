"""Material and loading definitions for DCE v4 polyline solver."""
from __future__ import annotations
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AppliedStress:
    sigma_xx: float
    sigma_yy: float
    sigma_xy: float

    def tensor(self) -> np.ndarray:
        return np.array(
            [[self.sigma_xx, self.sigma_xy],
             [self.sigma_xy, self.sigma_yy]],
            dtype=float,
        )
