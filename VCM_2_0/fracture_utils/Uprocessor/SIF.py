"""
DEPRECATED: legacy aggregator module.

This repository used to expose all SIF utilities via Uprocessor/SIF.py.
It has been refactored into:
  - SIF_cod.py : COD/CSD-based estimators and Euclidean near-tip fits
  - SIF_pk.py  : PK/J-based estimators

This shim is kept for backward compatibility with older notebooks.
Prefer importing from SIF_cod or SIF_pk directly.
"""

from __future__ import annotations

from .SIF_cod import *  # noqa: F401,F403
from .SIF_pk import *   # noqa: F401,F403

# best-effort __all__ aggregation
try:
    from . import SIF_cod as _cod
    from . import SIF_pk as _pk
    __all__ = sorted(set(getattr(_cod, "__all__", []) + getattr(_pk, "__all__", [])))
except Exception:
    __all__ = []
