"""
utils_half package facade.

Goal:
- Same public surface as `utils`
- Additionally exposes half-crack capable solver via DCENetworkStaticV4.solve(..., crack_mode="half")
"""

from __future__ import annotations

import importlib

# Re-export the standard facade modules
from .solver import *      # DCENetworkStaticV4, Material, AppliedStress, CrackNetworkV4, ...
from .processor import *   # DCEResultsNetworkV4, SIF utilities, ...
from .plotter import *     # DCEPlotterV4, StressPlotOptsV4, ...
from .generator import *   # CrackNetworkGenerator, ...

# Optional: mirror utils.reload_all() workflow
def reload_all():
    """
    Reload common utils_half submodules for notebook iteration.

    This mirrors the pattern you use in `utils.reload_all()`.
    """
    mods = [
        ".solver_kernels_half",
        ".solver_polyline_half",
        ".solver_parametrized_half",
        ".solver",
        ".processor_kernels_half",
        ".processor_results_half",
        ".processor",
        ".plotter",
        ".generator",
    ]
    pkg = __name__
    for m in mods:
        try:
            importlib.reload(importlib.import_module(m, pkg))
        except Exception:
            # Keep reload robust; failures here should not block interactive use.
            pass

# Keep __all__ simple and robust: export whatever is in the namespace
# except private names.
__all__ = sorted([k for k in globals().keys() if not k.startswith("_")])
