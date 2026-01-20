"""
Common Jupyter preamble for VariationalCracks / fracture_utils.

Usage:
    import preamble
    preamble.enable_autoreload()

Then access symbols as:
    preamble.CrackNetworkGenerator
    preamble.CrackNetworkV4
    ...
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Ensure repository root is on sys.path
# ------------------------------------------------------------
def _ensure_repo_root_on_path():
    cwd = Path(os.getcwd()).resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "fracture_utils").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return parent
    raise RuntimeError(
        "Could not locate repository root containing 'fracture_utils/'."
    )

REPO_ROOT = _ensure_repo_root_on_path()

# ------------------------------------------------------------
# Jupyter autoreload helper
# ------------------------------------------------------------
def enable_autoreload():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except Exception:
        pass

# ------------------------------------------------------------
# Solver / generator / processor / plotter imports
# ------------------------------------------------------------
from fracture_utils.Ugenerator.network_gen import CrackNetworkGenerator

from fracture_utils.Usolver.network import CrackNetworkV4
from fracture_utils.Usolver.material import *
from fracture_utils.Usolver.parametrization import *

from fracture_utils.Uprocessor.results import DCEResultsNetworkV4

from fracture_utils.Uplotter.core import (
    DCEPlotterV4,
    StressPlotOptsV4,
)

from fracture_utils.Uplotter.plot_PK import *

# ------------------------------------------------------------
# Re-exports
# ------------------------------------------------------------
__all__ = [
    "np",
    "plt",
    "Path",
    "REPO_ROOT",
    "enable_autoreload",

    "CrackNetworkGenerator",
    "CrackNetworkV4",
    "DCEResultsNetworkV4",
    "DCEPlotterV4",
    "StressPlotOptsV4",
    "Material",
    "AppliedStress",
    "DCENetworkStaticV4"
]
