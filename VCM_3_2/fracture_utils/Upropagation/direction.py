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

    @staticmethod
    def _wrap_pi(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        twopi = 2.0 * np.pi
        th = (float(theta) + np.pi) % twopi - np.pi
        return float(th)

    def theta(self, KI: float, KII: float) -> float:
        KI = float(KI)
        KII = float(KII)

        if (abs(KI) + abs(KII)) == 0.0:
            return 0.0

        if abs(KII) < 1e-30:
            return 0.0

        disc = KI * KI + 8.0 * KII * KII
        root = float(np.sqrt(disc))
        den = 4.0 * KII

        # Eq. gives two branches:
        #   theta = 2*atan( (KI ± sqrt(KI^2 + 8*KII^2)) / (4*KII) )
        # Choose the smaller-magnitude angle to avoid sharp kink jumps.
        th_plus = self._wrap_pi(2.0 * float(np.arctan2(KI + root, den)))
        th_minus = self._wrap_pi(2.0 * float(np.arctan2(KI - root, den)))
        return th_plus if abs(th_plus) <= abs(th_minus) else th_minus
