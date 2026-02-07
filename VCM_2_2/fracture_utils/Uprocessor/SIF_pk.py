"""PK-based stress-intensity factor (SIF) utilities.

This module contains PK/J-based estimators that operate on a PKProcessor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import math



def pk_sif_from_window(
    pk,
    *,
    edge_index: int = 0,
    at: str = "end",
    window_panels: int = 12,
    window_frac: float = 0.5,
    exclude_self: bool = True,
    plane_strain: bool = True,
    theta_rot: float = 0.0,
):
    """
    Compute (KI, KII) from the Peach–Koehler (PK) window force.

    IMPORTANT (frame consistency)
    -----------------------------
    The PK output is a *vector* force F in the global frame. To compare against a
    target convention/frame (e.g. Cotterell–Rice "global" frame), you must rotate
    the *force* (and the local basis vectors) BEFORE converting (J1,J2)->(KI,KII).
    Rotating (KI,KII) afterwards is not equivalent in mixed mode.

    Parameters
    ----------
    pk : PKProcessor
        Instance constructed from a DCEResultsNetworkV4 object.
    edge_index : int
        Crack edge index (as used by results reconstructor).
    at : {'end','start'}
        Which tip to use.
    window_panels : int
        Number of panels to include in the PK force window (preferred).
    window_frac : float
        Fractional window length fallback (used if window_panels not supported).
    exclude_self : bool
        Whether to exclude self-stress in the stress evaluation.
    plane_strain : bool
        If True, use E' = E/(1-nu^2). If False, use E' = E (plane stress).
    theta_rot : float
        Optional *global* rotation angle (radians) applied to BOTH the PK force
        vector and the local basis (ex,ey) prior to projection. Use this to map
        your computational/global frame into the comparison frame.

    Returns
    -------
    KI, KII : floats
    meta : dict
        Diagnostics: raw/rotated forces, bases, J1,J2, Eprime.
    """
    # -------------------------------
    # 1) PK force in global frame
    # -------------------------------
    try:
        F = pk.F_PK_window(
            edge_index=int(edge_index),
            at=at,
            exclude_self=exclude_self,
            window_panels=window_panels,
            window_frac=window_frac,
        )
    except TypeError:
        # backward compatibility
        F = pk.F_PK_window(
            edge_index=int(edge_index),
            at=at,
            exclude_self=exclude_self,
            window_panels=window_panels,
        )

    if F is None:
        return float("nan"), float("nan"), {"reason": "F_PK_window returned None"}

    F = np.asarray(F, float).reshape(2,)

    # -------------------------------
    # 2) Local basis from reconstructor
    # -------------------------------
    _, _, _, extra = pk.res.reconstruct_cod_csd_panel_midpoints(
        edge_index=int(edge_index),
        enforce_global_tip_zero=False,
    )
    extra = dict(extra) if isinstance(extra, dict) else {}

    ex = np.asarray(extra.get("ex", [1.0, 0.0]), float).reshape(2,)
    ey = np.asarray(extra.get("ey", [0.0, 1.0]), float).reshape(2,)

    # -------------------------------
    # 3) Optional rotation (vector rotation) BEFORE J->K
    # -------------------------------
    def _rot(v, th):
        c = float(np.cos(th)); s = float(np.sin(th))
        return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]], float)

    theta_rot = float(theta_rot)
    F_use  = _rot(F,  theta_rot) if abs(theta_rot) > 0 else F.copy()
    ex_use = _rot(ex, theta_rot) if abs(theta_rot) > 0 else ex.copy()
    ey_use = _rot(ey, theta_rot) if abs(theta_rot) > 0 else ey.copy()

    # normalize basis defensively
    nex = float(np.linalg.norm(ex_use))
    ney = float(np.linalg.norm(ey_use))
    if nex > 0:
        ex_use /= nex
    if ney > 0:
        ey_use /= ney

    # -------------------------------
    # 4) Project PK force into local crack frame: J1 (tangent), J2 (normal)
    # -------------------------------
    J1 = float(np.dot(F_use, ex_use))
    J2 = float(np.dot(F_use, ey_use))

    # -------------------------------
    # 5) Convert (J1,J2)->(KI,KII)
    # -------------------------------
    E = float(pk.res.calc.material.E)
    nu = float(pk.res.calc.material.nu)
    Eprime = E / (1.0 - nu**2) if bool(plane_strain) else E

    # Relations (plane strain/stress via Eprime):
    #   J1 = (KI^2 + KII^2)/Eprime
    #   J2 = -(2*KI*KII)/Eprime
    A = Eprime * J1
    C = -0.5 * Eprime * J2

    disc = float(max(A*A - 4.0*C*C, 0.0))
    D = float(np.sqrt(disc))

    KI2  = float(max(0.5 * (A + D), 0.0))
    KII2 = float(max(0.5 * (A - D), 0.0))

    KI  = float(np.sqrt(KI2))
    KII = float(np.sign(C) * np.sqrt(KII2))  # enforce KI*KII sign via C

    meta = dict(
        F_raw=np.asarray(F, float),
        F=np.asarray(F_use, float),
        ex_raw=np.asarray(ex, float),
        ey_raw=np.asarray(ey, float),
        ex=np.asarray(ex_use, float),
        ey=np.asarray(ey_use, float),
        J1=J1,
        J2=J2,
        Eprime=Eprime,
        plane_strain=bool(plane_strain),
        theta_rot=theta_rot,
        window_panels=int(window_panels),
        window_frac=float(window_frac),
        exclude_self=bool(exclude_self),
        edge_index=int(edge_index),
        at=str(at),
    )
    return KI, KII, meta

# -------------------------
# Cotterell–Rice analytical SIFs (paper form with alpha/2)
# IMPORTANT: for arc crack use a_arc = R*alpha (half arc length)
# ------------------------


# =====================================================================
# OO Facade: EnergyReleaseRateSIF
#
# This module contains PK/J (energy-release-rate based) estimators.
# We expose them through a single class for notebook ergonomics.
# =====================================================================

class EnergyReleaseRateSIF:
    """
    Energy-release-rate / PK-window based SIF estimators.

    Primary method:
      - from_pk_window: wraps pk_sif_from_window
    """

    @staticmethod
    def from_pk_window(
        pk,
        *,
        edge_index: int = 0,
        at: str = "end",
        window_panels: int = 12,
        window_frac: float = 0.5,
        exclude_self: bool = True,
        plane_strain: bool = True,
        theta_rot: float = 0.0,
    ):
        return pk_sif_from_window(
            pk,
            edge_index=edge_index,
            at=at,
            window_panels=window_panels,
            window_frac=window_frac,
            exclude_self=exclude_self,
            plane_strain=plane_strain,
            theta_rot=theta_rot,
        )

__all__ = [
    "pk_sif_from_window",
    "EnergyReleaseRateSIF",
]
