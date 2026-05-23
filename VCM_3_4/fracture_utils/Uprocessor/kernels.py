"""Elastic kernels used by results post-processing (u and stress).

The dislocation kernels live in ``Usolver.solve_kernels`` as the single source
of truth (they are imported by the operator-assembly code). This module
re-exports them so post-processing call sites keep their existing import
``from fracture_utils.Uprocessor.kernels import ...`` and adds the
post-processing-specific ``applied_tensor`` helper that is not relevant
to the solver layer.
"""
from __future__ import annotations
import numpy as np

from fracture_utils.Usolver.solve_kernels import (  # noqa: F401  (re-export)
    edge_dislocation_u,
    stress_edge_dislocation,
)


def applied_tensor(applied) -> np.ndarray:
    if hasattr(applied, "tensor") and callable(applied.tensor):
        return np.asarray(applied.tensor(), float)
    return np.array([[float(applied.sigma_xx), float(applied.sigma_xy)],
                     [float(applied.sigma_xy), float(applied.sigma_yy)]], float)
