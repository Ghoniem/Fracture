"""
fracture_utils
==============
Namespaced utilities for the Variational Crack Method (VCM) / DCE tooling.

This top-level package is intentionally *thin* and uses lazy subpackage loading
to avoid circular-import failures in large repos.

Key improvement vs v3:
- Do NOT hardcode subpackage names (e.g. Uvalidation). Instead, discover
  available direct children of the package at runtime via pkgutil.

This prevents ModuleNotFoundError when a subpackage folder does not exist
or has a different name/casing.
"""

# fracture_utils/__init__.py
from __future__ import annotations
from typing import Any, Set
import pkgutil

__version__ = "4.1.0"


def _discover_children() -> Set[str]:
    return {m.name for m in pkgutil.iter_modules(__path__)}


_CHILDREN = _discover_children()
__all__ = sorted([*_CHILDREN, "__version__"])


def __getattr__(name: str) -> Any:
    if name in _CHILDREN:
        module = __import__(f"{__name__}.{name}", fromlist=[name])
        globals()[name] = module
        return module
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available subpackages: {sorted(_CHILDREN)}"
    )


def __dir__():
    return sorted(list(globals().keys()) + list(_CHILDREN))
