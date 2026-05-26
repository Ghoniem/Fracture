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

    Uses a common closed-form expression (Erdogan-Sih / MTS) for the
    kink angle relative to the current tangent:

        theta = 2 * atan2( KI - sqrt(KI^2 + 8*KII^2), 4*KII )

    Two regularizations are applied to prevent step-to-step direction
    flips (zigzag) when KII is dominated by numerical noise:

    1. KII noise threshold (``kii_noise_ratio``, default 0.02):
       if |KII| / (|KI| + |KII|) < threshold, the loading is treated as
       essentially pure mode I and theta = 0 (straight propagation). At
       a 2 % ratio this corresponds to a kink under ~1 deg, well below
       physical meaningfulness for a BEM-recovered SIF.

    2. Soft kink limit (``max_kink_deg``, default 70 deg):
       theta is clamped to +/- max_kink_deg. Mode-I-dominant cracks
       rarely turn more than this in one step; an unclamped result
       usually indicates a branch-selection artifact.

    Parameters
    ----------
    kii_noise_ratio : float, default 0.02
        Suppress kink to 0 when |KII|/(|KI|+|KII|) is below this.
    max_kink_deg : float, default 70.0
        Hard clamp on the magnitude of the returned theta (degrees).
    """

    kii_noise_ratio: float = 0.02
    max_kink_deg: float = 70.0

    @staticmethod
    def _wrap_pi(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        twopi = 2.0 * np.pi
        th = (float(theta) + np.pi) % twopi - np.pi
        return float(th)

    @staticmethod
    def _sigma_theta(KI: float, KII: float, theta: float) -> float:
        """Circumferential stress factor at angle theta (up to common positive scale)."""
        c = float(np.cos(0.5 * float(theta)))
        s = float(np.sin(0.5 * float(theta)))
        return float(KI * (c ** 3) - 3.0 * KII * s * (c ** 2))

    def theta(self, KI: float, KII: float) -> float:
        KI = float(KI)
        KII = float(KII)

        if (abs(KI) + abs(KII)) == 0.0:
            return 0.0

        # KII-noise rejection: if the mixed-mode component is tiny
        # relative to KI, treat as pure mode I (straight ahead). This
        # kills the noise-driven zigzag on near-mode-I cracks where the
        # BEM/COD-recovered KII can flip sign step-to-step.
        if abs(KII) / (abs(KI) + abs(KII)) < float(self.kii_noise_ratio):
            return 0.0

        if abs(KII) < 1e-30:
            return 0.0

        disc = KI * KI + 8.0 * KII * KII
        root = float(np.sqrt(disc))
        den = 4.0 * KII

        # Eq. gives two stationary-angle branches:
        #   theta = 2*atan( (KI +- sqrt(KI^2 + 8*KII^2)) / (4*KII) )
        # Select the branch that maximizes hoop stress (MTS criterion).
        # This avoids wrong-turn branch picks when KI is negative.
        th_plus = self._wrap_pi(2.0 * float(np.arctan2(KI + root, den)))
        th_minus = self._wrap_pi(2.0 * float(np.arctan2(KI - root, den)))
        s_plus = self._sigma_theta(KI, KII, th_plus)
        s_minus = self._sigma_theta(KI, KII, th_minus)

        if s_plus > s_minus:
            theta = th_plus
        elif s_minus > s_plus:
            theta = th_minus
        else:
            # Tie-breaker: preserve smooth/compact branch where equivalent.
            theta = th_plus if abs(th_plus) <= abs(th_minus) else th_minus

        # Soft clamp so a wild branch pick can't produce e.g. a -130 deg
        # back-fold. Sign of theta is preserved; this is a safety bound,
        # not a refinement of the MTS criterion.
        max_rad = float(self.max_kink_deg) * np.pi / 180.0
        if abs(theta) > max_rad:
            theta = max_rad if theta > 0 else -max_rad
        return theta
