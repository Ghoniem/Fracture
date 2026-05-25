"""Propagation direction laws: (K_I, K_II) -> theta.

theta is measured relative to the local crack tangent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


class DirectionLaw(Protocol):
    def theta(self, KI: float, KII: float) -> float:
        """Return propagation angle theta (radians) relative to local tangent."""


@dataclass(frozen=True)
class MaximumHoopStressLaw:
    """Maximum hoop stress (MTS) criterion (isotropic LEFM).

    Uses a common closed-form expression (Erdogan–Sih / MTS) for the
    kink angle relative to the current tangent:

        theta = 2 * atan2( KI - sqrt(KI^2 + 8*KII^2), 4*KII )

    Notes
    -----
    - Returns 0 when KII≈0.
    - This formula is typically used with KI >= 0. If KI < 0 (net closure),
      you may want to suppress growth upstream.
    """

    def theta(self, KI: float, KII: float) -> float:
        KI = float(KI)
        KII = float(KII)

        if (abs(KI) + abs(KII)) == 0.0:
            return 0.0

        if abs(KII) < 1e-30:
            return 0.0

        disc = KI * KI + 8.0 * KII * KII
        num = KI - float(np.sqrt(disc))
        den = 4.0 * KII
        return 2.0 * float(np.arctan2(num, den))
