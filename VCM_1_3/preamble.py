"""
Common Jupyter preamble for VariationalCracks / fracture_utils.

Usage:
    from preamble import *
    enable_autoreload()

This module:
  * Ensures the repo root (containing 'fracture_utils/') is on sys.path
  * Provides standard imports and re-exports for notebooks
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Ensure repository root is on sys.path
# ------------------------------------------------------------
def _ensure_repo_root_on_path() -> Path:
    cwd = Path(os.getcwd()).resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "fracture_utils").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return parent
    raise RuntimeError("Could not locate repository root containing 'fracture_utils/'.")

REPO_ROOT = _ensure_repo_root_on_path()

# ------------------------------------------------------------
# Jupyter autoreload helper
# ------------------------------------------------------------
def enable_autoreload():
    """Enable IPython autoreload(2) if running inside a notebook."""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except Exception:
        pass

# ------------------------------------------------------------
# Core solver / generator / processor imports
# ------------------------------------------------------------
from fracture_utils.Ugenerator.network_gen import CrackNetworkGenerator

from fracture_utils.Usolver.network import CrackNetworkV4 as CrackNetworkV4_base
from fracture_utils.Usolver.network_v1 import CrackNetworkV4 as CrackNetworkV4_v1
CrackNetworkV4 = CrackNetworkV4_v1
from fracture_utils.Usolver.material import Material, AppliedStress
# from fracture_utils.Usolver.parametrization import DCENetworkStaticV4  # type: ignore
# from fracture_utils.Usolver.parametrization_v1 import DCENetworkStaticV4  # type: ignore
from fracture_utils.Usolver.parametrization_v4 import DCENetworkStaticV4  # type: ignore


# import fracture_utils.Usolver.build_v2 as build
import fracture_utils.Usolver.build_v3 as build



from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Uprocessor import diagnostics as diagnostics

# ------------------------------------------------------------
# Plotter imports (reflecting your Uplotter structure)
# ------------------------------------------------------------
from fracture_utils.Uplotter.core import DCEPlotterV4
from fracture_utils.Uplotter.plot_stress import StressPlotOptsV4
from fracture_utils.Uplotter.plot_displacement import DCEPlotterDisplacementV4

# Deformed network plotter (new filename)
from fracture_utils.Uplotter.plot_deformed_network_v2 import DCEPlotterDeformedV4 as DCEPlotterDeformedV4



# PK and B-content plotter
from fracture_utils.Uplotter.plot_PK import DCEPlotterPK, VecPlotStyle

__all__ = [
    # numpy/mpl basics
    "np",
    "plt",
    "Path",
    "REPO_ROOT",
    "enable_autoreload",

    # generator / solver / results
    "CrackNetworkGenerator",
    "CrackNetworkV4",
    "Material",
    "AppliedStress",
    "DCENetworkStaticV4",
    "DCEResultsNetworkV4",

    # diagnostics module (namespace)
    "diagnostics",

    # plotters / opts
    "DCEPlotterV4",
    "StressPlotOptsV4",
    "DCEPlotterDeformedV4",
    "DCEPlotterPK",
    "VecPlotStyle",
]