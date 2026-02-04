"""
Notebook preamble (repo-root)

Curated high-frequency VCM helpers for notebooks.

Adds (v13):
- Analytical / helper functions from Uprocessor:
    - cotterell_rice_K
    - sif_from_cod_fit
    - solve_K
    - rotate_sifs

These live in Uprocessor/SIF_cod.py and/or Uprocessor/SIF.py depending on branch.
We resolve them lazily and robustly across common module name variants.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib

import numpy as np
import matplotlib.pyplot as plt


def find_repo_root(start=None) -> Path:
    here = Path(start or os.getcwd()).resolve()
    for p in [here] + list(here.parents):
        if (p / "fracture_utils").is_dir():
            return p
    return here


REPO_ROOT: Path = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR: Path = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_output_dir(path: str | Path) -> Path:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(path).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def enable_autoreload():
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except Exception:
        pass


import fracture_utils as fu  # noqa: E402


def _try_import(dotted: str):
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def _try_import_many(dotted_list: list[str]):
    for d in dotted_list:
        m = _try_import(d)
        if m is not None:
            return m
    return None


def _get_attr(mod, name: str, default=None):
    return getattr(mod, name, default) if mod is not None else default


# -----------------------------
# Subpackage namespace handles
# -----------------------------
Usolver = getattr(fu, "Usolver", None)
Uplotter = getattr(fu, "Uplotter", None)
Uprocessor = getattr(fu, "Uprocessor", None)
Ugenerator = getattr(fu, "Ugenerator", None)
Ubem = getattr(fu, "Ubem", None)
Uvalidation = getattr(fu, "Uvalidation", None) or getattr(fu, "UValidation", None)

# -----------------------------
# Generator helpers
# -----------------------------
gen_mod = _try_import("fracture_utils.Ugenerator.generator")
geom_gen_mod = _try_import("fracture_utils.Ugenerator.geometry")
boundary_gen_mod = _try_import("fracture_utils.Ugenerator.boundary")

CrackNetworkGenerator = _get_attr(gen_mod, "CrackNetworkGenerator")
GeometryUtils = _get_attr(geom_gen_mod, "GeometryUtils")
polyline_arc_network = _get_attr(geom_gen_mod, "polyline_arc_network")
BoundaryManager = _get_attr(boundary_gen_mod, "BoundaryManager")
Boundary = _get_attr(boundary_gen_mod, "Boundary")
LinearSegment = _get_attr(boundary_gen_mod, "LinearSegment")
CircularArc = _get_attr(boundary_gen_mod, "CircularArc")

# -----------------------------
# Solver / network core
# -----------------------------
network = _try_import("fracture_utils.Usolver.network")
material_mod = _try_import("fracture_utils.Usolver.material")
param_mod = _try_import_many([
    "fracture_utils.Usolver.parametrization2_0_v2",
    "fracture_utils.Usolver.parametrization2_0_v1",
    "fracture_utils.Usolver.parametrization",
])

CrackNetworkV4 = _get_attr(network, "CrackNetworkV4")
Material = _get_attr(material_mod, "Material")
AppliedStress = _get_attr(material_mod, "AppliedStress")
DCENetworkStaticV4 = _get_attr(param_mod, "DCENetworkStaticV4")

# -----------------------------
# Processor / results + analytic SIF helpers
# -----------------------------
results_mod = _try_import_many([
    "fracture_utils.Uprocessor.results",
    "fracture_utils.Uprocessor.processor_results",
    "fracture_utils.Uprocessor.processor_results_patched_full",
])
DCEResultsNetworkV4 = _get_attr(results_mod, "DCEResultsNetworkV4")

# Prefer SIF_cod for COD-fit based SIF utilities; fall back to SIF module
sif_cod_mod = _try_import_many([
    "fracture_utils.Uprocessor.SIF_cod",
    "fracture_utils.Uprocessor.sif_cod",
])
sif_mod = _try_import_many([
    "fracture_utils.Uprocessor.SIF",
    "fracture_utils.Uprocessor.sif",
])

cotterell_rice_K = _get_attr(sif_cod_mod, "cotterell_rice_K")
sif_from_cod_fit = _get_attr(sif_cod_mod, "sif_from_cod_fit")
solve_K = _get_attr(sif_cod_mod, "solve_K")
euclid_tip_fit_from_edge = _get_attr(sif_cod_mod, "euclid_tip_fit_from_edge")
solve_K_polyline = _get_attr(sif_cod_mod, "solve_K_polyline")

# rotate_sifs sometimes lives in SIF.py
rotate_sifs = _get_attr(sif_cod_mod, "rotate_sifs")
if rotate_sifs is None:
    rotate_sifs = _get_attr(sif_mod, "rotate_sifs")

# If solve_K is not in SIF_cod, try SIF
if solve_K is None:
    solve_K = _get_attr(sif_mod, "solve_K")
# If sif_from_cod_fit moved, try SIF
if sif_from_cod_fit is None:
    sif_from_cod_fit = _get_attr(sif_mod, "sif_from_cod_fit")
# If cotterell_rice_K moved, try SIF
if cotterell_rice_K is None:
    cotterell_rice_K = _get_attr(sif_mod, "cotterell_rice_K")

# -----------------------------
# Plotting helpers
# -----------------------------
plotter_core = _try_import_many([
    "fracture_utils.Uplotter.core",
    "fracture_utils.Uplotter.plotter_core",
])
plotter_opts = _try_import_many([
    "fracture_utils.Uplotter.opts",
    "fracture_utils.Uplotter.options",
])
plotter_def = _try_import_many([
    "fracture_utils.Uplotter.plot_deformed_network",
    "fracture_utils.Uplotter.plotter_deformed_v4",
    "fracture_utils.Uplotter.plotter_deformed",
])
plotter_pk = _try_import_many([
    "fracture_utils.Uplotter.plot_PK",
    "fracture_utils.Uplotter.pk_plotter",
    "fracture_utils.Uplotter.plotter_pk",
])

DCEPlotterV4 = _get_attr(plotter_core, "DCEPlotterV4")
StressPlotOptsV4 = _get_attr(plotter_opts, "StressPlotOptsV4")
DCEPlotterDeformedV4 = _get_attr(plotter_def, "DCEPlotterDeformedV4")
DCEPlotterPK = _get_attr(plotter_pk, "DCEPlotterPK")

_call = _get_attr(plotter_opts, "_call")

VecPlotStyle = _get_attr(plotter_opts, "VecPlotStyle")
if VecPlotStyle is None:
    VecPlotStyle = _get_attr(plotter_pk, "VecPlotStyle")

plot_deformed_network = plotter_def
plot_PK = plotter_pk

# -----------------------------
# Validation (COD)
# -----------------------------
val_cod_mod = _try_import_many([
    "fracture_utils.UValidation.validation_cod",
    "fracture_utils.Uvalidation.validation_cod",
])
CODProfile = _get_attr(val_cod_mod, "CODProfile")
CODCaseResult = _get_attr(val_cod_mod, "CODCaseResult")
SIFCaseResult = _get_attr(val_cod_mod, "SIFCaseResult")
CODSweepResult = _get_attr(val_cod_mod, "CODSweepResult")
run_cod_sweep = _get_attr(val_cod_mod, "run_cod_sweep")
plot_cod_overlay = _get_attr(val_cod_mod, "plot_cod_overlay")
plot_cod_error_vs_knob = _get_attr(val_cod_mod, "plot_cod_error_vs_knob")
plot_cod_error_summary = _get_attr(val_cod_mod, "plot_cod_error_summary")


__all__ = [
    "np",
    "plt",
    "REPO_ROOT",
    "OUTPUT_DIR",
    "set_output_dir",
    "enable_autoreload",
    "fu",
    "Usolver",
    "Uplotter",
    "Uprocessor",
    "Ugenerator",
    "Ubem",
    "Uvalidation",
    # generator
    "CrackNetworkGenerator",
    "GeometryUtils",
    "polyline_arc_network",
    "BoundaryManager",
    "Boundary",
    "LinearSegment",
    "CircularArc",
    # solver/network
    "network",
    "CrackNetworkV4",
    "Material",
    "AppliedStress",
    "DCENetworkStaticV4",
    # processor/results + analytic SIF
    "DCEResultsNetworkV4",
    "cotterell_rice_K",
    "sif_from_cod_fit",
    "solve_K",
    "rotate_sifs",
    # plotting
    "DCEPlotterV4",
    "StressPlotOptsV4",
    "DCEPlotterDeformedV4",
    "DCEPlotterPK",
    "_call",
    "VecPlotStyle",
    "plot_deformed_network",
    "plot_PK",
    # validation
    "CODProfile",
    "CODCaseResult",
    "SIFCaseResult",
    "CODSweepResult",
    "run_cod_sweep",
    "plot_cod_overlay",
    "plot_cod_error_vs_knob",
    "plot_cod_error_summary",
    "euclid_tip_fit_from_edge",
    "solve_K_polyline",
]
