"""
Common Jupyter preamble for VariationalCracks / fracture_utils.

Usage:
    from preamble import *
    enable_autoreload()
    
    # Output directory automatically set up
    print(f"Repo root: {REPO_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")

This module:
  * Ensures the repo root (containing 'fracture_utils/') is on sys.path
  * Provides standard imports and re-exports for notebooks
  * Auto-detects repository structure
  * Sets up output directories
  * Provides plotting helpers
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Auto-detect repository root
# ------------------------------------------------------------
def find_repo_root(start_path=None):
    """
    Find repository root by looking for fracture_utils/ directory
    
    Args:
        start_path: Starting path (default: current working directory)
        
    Returns:
        Path object pointing to repository root
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)
    
    for parent in [start_path] + list(start_path.parents):
        if (parent / "fracture_utils").is_dir():
            return parent
    raise RuntimeError("Could not find repository root (looking for 'fracture_utils/' directory)")

# Find repository root and add to path
REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Current notebook directory
NOTEBOOK_DIR = Path.cwd()

# ------------------------------------------------------------
# Output directory management
# ------------------------------------------------------------
def get_output_dir(subdir="", create=True):
    """
    Get output directory relative to repository root
    
    Args:
        subdir: Subdirectory name (e.g., "ArcValidation", "plots", etc.)
        create: If True, create directory if it doesn't exist
        
    Returns:
        Path object pointing to output directory
    
    Examples:
        >>> out_dir = get_output_dir("ArcValidation")
        >>> fig.savefig(out_dir / "plot.png")
    """
    if subdir:
        output_path = REPO_ROOT / "output" / subdir
    else:
        output_path = REPO_ROOT / "output"
    
    if create:
        output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path

# Default output directory (can be overridden in notebooks)
OUTPUT_DIR = get_output_dir()

# ------------------------------------------------------------
# Jupyter autoreload helper
# ------------------------------------------------------------
def enable_autoreload():
    """Enable IPython autoreload(2) if running inside a notebook."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except Exception:
        pass


# ----------------------------
# Helper: safe plotting calls
# ----------------------------
def _call(obj, name, *args, **kwargs):
    fn = getattr(obj, name, None)
    if fn is None:
        print(f"[skip] {obj.__class__.__name__}.{name} not found")
        return None
    try:
        return fn(*args, **kwargs)
    except TypeError as e:
        # In case signature differs slightly across versions
        print(f"[warn] {obj.__class__.__name__}.{name} signature mismatch: {e}")
        return fn(*args)


# ------------------------------------------------------------
# Generator imports - Now simplified using __init__.py
# ------------------------------------------------------------
from fracture_utils.Ugenerator import (
    CrackNetworkGenerator,
    GeometryUtils,
    NetworkAnalyzer,
    NetworkVisualizer,
    Boundary,
    BoundarySegment,
    LinearSegment,
    CircularArc,
    BoundaryManager,
    CADImporter,
    CADExporter,
    BezierCurve,
    SplineCurve,
    polyline_arc_network,
)

# ------------------------------------------------------------
# Solver imports
# ------------------------------------------------------------
from fracture_utils.Usolver.network import CrackNetworkV4 as CrackNetworkV4
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4
import fracture_utils.Usolver.build as build
from fracture_utils.Usolver.material import Material, AppliedStress

from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Uprocessor import diagnostics as diagnostics
from fracture_utils.Uprocessor.SIF import sif_from_cod_fit, cotterell_rice_K, rotate_sifs
from fracture_utils.Uprocessor.SIF import pk_sif_from_window,sif_from_cod_fit_euclid_tip, solve_K_polyline
from fracture_utils.Uprocessor.PK import PKProcessor

# ------------------------------------------------------------
# Plotter imports
# ------------------------------------------------------------
from fracture_utils.Uplotter.core import DCEPlotterV4
from fracture_utils.Uplotter.plot_stress import StressPlotOptsV4
from fracture_utils.Uplotter.plot_displacement import DCEPlotterDisplacementV4
from fracture_utils.Uplotter.plot_deformed_network import DCEPlotterDeformedV4 as DCEPlotterDeformedV4
from fracture_utils.Uplotter.plot_PK import DCEPlotterPK, VecPlotStyle

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
from fracture_utils.UValidation.validation_cod import *
from fracture_utils.UValidation.validation_sif import *


__all__ = [
    # Path management
    "Path",
    "REPO_ROOT",
    "NOTEBOOK_DIR",
    "OUTPUT_DIR",
    "get_output_dir",
    "find_repo_root",
    
    # Helpers
    "enable_autoreload",
    "_call",
    
    # numpy/mpl basics
    "np",
    "plt",

    # Generator (from Ugenerator.__init__)
    "CrackNetworkGenerator",
    "GeometryUtils",
    "NetworkAnalyzer",
    "NetworkVisualizer",
    "Boundary",
    "BoundarySegment",
    "LinearSegment",
    "CircularArc",
    "BoundaryManager",
    "CADImporter",
    "CADExporter",
    "BezierCurve",
    "SplineCurve",
    "polyline_arc_network",
    
    # Solver / results
    "CrackNetworkV4",
    "Material",
    "AppliedStress",
    "DCENetworkStaticV4",
    "build",
    

    # Plotters
    "DCEPlotterV4",
    "StressPlotOptsV4",
    "DCEPlotterDeformedV4",
    "DCEPlotterPK",
    "VecPlotStyle",
    
    # Validation 
    "CODSweepResult",
    "SIFSweepResult",
    "run_cod_sweep", 
    "plot_cod_overlay", 
    "plot_cod_error_vs_knob",
    "run_sif_sweep", 
    "plot_sif_error_vs_knob", 
    "plot_sif_estimators_overlay", 
    "SIFMeasures",
    
    # Processor
    "PKProcessor",
    "DCEResultsNetworkV4",
    "diagnostics",
    "sif_from_cod_fit",
    "cotterell_rice_K",
    "rotate_sifs",
    "pk_sif_from_window",
    "sif_from_cod_fit_euclid_tip",
    "solve_K_polyline",
]
