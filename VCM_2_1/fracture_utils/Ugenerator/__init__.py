"""
Crack Network Generator Package (Ugenerator)

Curated public API (Option B) with **lazy imports** for heavy/optional modules
(visualization + CAD). This preserves existing notebook imports like:

    from fracture_utils.Ugenerator import CrackNetworkGenerator, CADImporter

while reducing circular-import risk and avoiding importing heavy dependencies
at package import time.

Best practice inside the package:
- Use relative imports (from .foo import Bar) within Ugenerator modules.
- Do NOT import `fracture_utils` (package root) from inside fracture_utils.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.1.0"

# ---- Core (safe, lightweight) eager imports ----
from .generator import CrackNetworkGenerator  # noqa: F401
from .geometry import GeometryUtils, polyline_arc_network  # noqa: F401
from .analysis import NetworkAnalyzer  # noqa: F401
from .boundary import (  # noqa: F401
    Boundary,
    BoundarySegment,
    LinearSegment,
    CircularArc,
    BoundaryManager,
)

# ---- Lazy exports (avoid heavy / optional imports at package import time) ----
_LAZY = {
    # visualization
    "NetworkVisualizer": (".visualization", "NetworkVisualizer"),
    # CAD
    "CADImporter": (".cad_import", "CADImporter"),
    "CADExporter": (".cad_import", "CADExporter"),
    "BezierCurve": (".cad_import", "BezierCurve"),
    "SplineCurve": (".cad_import", "SplineCurve"),
}


def __getattr__(name: str) -> Any:
    """
    PEP 562: module attribute access hook.

    Enables lazy loading of optional/heavy symbols while preserving
    `from fracture_utils.Ugenerator import NetworkVisualizer, CADImporter, ...`.

    Raises AttributeError for unknown names (normal Python behavior).
    """
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = __import__(__name__ + mod_name, fromlist=[attr])
        value = getattr(module, attr)
        globals()[name] = value  # cache for future access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    # core
    "CrackNetworkGenerator",
    "GeometryUtils",
    "polyline_arc_network",
    "NetworkAnalyzer",
    "Boundary",
    "BoundarySegment",
    "LinearSegment",
    "CircularArc",
    "BoundaryManager",
    # lazy
    "NetworkVisualizer",
    "CADImporter",
    "CADExporter",
    "BezierCurve",
    "SplineCurve",
    # meta
    "__version__",
]
