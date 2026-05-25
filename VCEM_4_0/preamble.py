"""
Notebook preamble (VCM_3_4)

Curated high-frequency helpers for notebooks.
This file intentionally targets the current module names in this repo.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Find the folder that directly contains `fracture_utils`."""
    candidates = [Path(start or os.getcwd()).resolve(), Path(__file__).resolve().parent]
    for base in candidates:
        for p in [base] + list(base.parents):
            if (p / "fracture_utils").is_dir():
                return p
    return Path(__file__).resolve().parent


def _try_import(dotted: str):
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def _get_attr(mod, name: str, default=None):
    return getattr(mod, name, default) if mod is not None else default


def _missing_function(name: str, hint: str):
    def _missing(*args, **kwargs):  # type: ignore
        raise NameError(f"{name} is not available. {hint}")

    return _missing


def _missing_class(name: str, hint: str):
    class _MissingClass:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"{name} is not available. {hint}")

    return _MissingClass


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


def enable_autoreload() -> None:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except Exception:
        pass


import fracture_utils as fu  # noqa: E402


# -----------------------------
# Subpackage namespace handles
# -----------------------------
Usolver = getattr(fu, "Usolver", None)
Uplotter = getattr(fu, "Uplotter", None)
Uprocessor = getattr(fu, "Uprocessor", None)
Ugenerator = getattr(fu, "Ugenerator", None)
Ubem = getattr(fu, "Ubem", None)
Uvalidation = getattr(fu, "Uvalidation", None)

# Some branches accidentally break Ugenerator package discovery.
if Ugenerator is None:
    Ugenerator = SimpleNamespace()


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
param_mod = _try_import("fracture_utils.Usolver.parametrization")

CrackNetworkV4 = _get_attr(network, "CrackNetworkV4")
Material = _get_attr(material_mod, "Material")
AppliedStress = _get_attr(material_mod, "AppliedStress")
DCENetworkStaticV4 = _get_attr(param_mod, "DCENetworkStaticV4")


# -----------------------------
# Processor / results + SIF
# -----------------------------
results_mod = _try_import("fracture_utils.Uprocessor.results")
sif_cod_mod = _try_import("fracture_utils.Uprocessor.SIF_cod")
sif_pk_mod = _try_import("fracture_utils.Uprocessor.SIF_pk")
pkproc_mod = _try_import("fracture_utils.Uprocessor.PK")

DCEResultsNetworkV4 = _get_attr(results_mod, "DCEResultsNetworkV4")

DisplacementSIF = _get_attr(
    sif_cod_mod,
    "DisplacementSIF",
    _missing_class("DisplacementSIF", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
DisplacementGradientSIF = _get_attr(
    sif_cod_mod,
    "DisplacementGradientSIF",
    _missing_class("DisplacementGradientSIF", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
AnalysisSIF = _get_attr(
    sif_cod_mod,
    "AnalysisSIF",
    _missing_class("AnalysisSIF", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
EnergyReleaseRateSIF = _get_attr(
    sif_pk_mod,
    "EnergyReleaseRateSIF",
    _missing_class("EnergyReleaseRateSIF", "Expected in fracture_utils.Uprocessor.SIF_pk."),
)

cotterell_rice_K = _get_attr(
    sif_cod_mod,
    "cotterell_rice_K",
    _missing_function("cotterell_rice_K", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
sif_from_cod_fit = _get_attr(
    sif_cod_mod,
    "sif_from_cod_fit",
    _missing_function("sif_from_cod_fit", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
solve_K_polyline = _get_attr(
    sif_cod_mod,
    "solve_K_polyline",
    _missing_function("solve_K_polyline", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
rotate_sifs = _get_attr(
    sif_cod_mod,
    "rotate_sifs",
    _missing_function("rotate_sifs", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
euclid_tip_fit_from_edge = _get_attr(
    sif_cod_mod,
    "euclid_tip_fit_from_edge",
    _missing_function("euclid_tip_fit_from_edge", "Expected in fracture_utils.Uprocessor.SIF_cod."),
)
estimate_KI_KII_from_jumps_near_tip = _get_attr(
    sif_cod_mod,
    "estimate_KI_KII_from_jumps_near_tip",
    _missing_function(
        "estimate_KI_KII_from_jumps_near_tip",
        "Expected in fracture_utils.Uprocessor.SIF_cod.",
    ),
)

PKProcessor = _get_attr(
    pkproc_mod,
    "PKProcessor",
    _missing_class("PKProcessor", "Expected in fracture_utils.Uprocessor.PK."),
)


# -----------------------------
# Plotting helpers
# -----------------------------
plotter_core = _try_import("fracture_utils.Uplotter.core")
plotter_opts = _try_import("fracture_utils.Uplotter.opts")
plotter_def = _try_import("fracture_utils.Uplotter.plot_deformed_network")
plotter_pk = _try_import("fracture_utils.Uplotter.plot_PK")

DCEPlotterV4 = _get_attr(plotter_core, "DCEPlotterV4")
StressPlotOptsV4 = _get_attr(plotter_opts, "StressPlotOptsV4")
DCEPlotterDeformedV4 = _get_attr(plotter_def, "DCEPlotterDeformedV4")
DCEPlotterPK = _get_attr(plotter_pk, "DCEPlotterPK")

_call = _get_attr(plotter_opts, "_call")
VecPlotStyle = _get_attr(plotter_pk, "VecPlotStyle")

# Keep historical exports as module handles.
plot_deformed_network = plotter_def
plot_PK = plotter_pk



# -----------------------------
# Propagation (crack growth)
# -----------------------------
prop_cfg_mod = _try_import("fracture_utils.Upropagation.config")
prop_tough_mod = _try_import("fracture_utils.Upropagation.toughness")
prop_dir_mod = _try_import("fracture_utils.Upropagation.direction")
prop_eval_mod = _try_import("fracture_utils.Upropagation.evaluate")
prop_prog_mod = _try_import("fracture_utils.Upropagation.propagator")

PropagationConfig = _get_attr(
    prop_cfg_mod,
    "PropagationConfig",
    _missing_class("PropagationConfig", "Expected in fracture_utils.Upropagation.config."),
)

ConstantToughness = _get_attr(
    prop_tough_mod,
    "ConstantToughness",
    _missing_class("ConstantToughness", "Expected in fracture_utils.Upropagation.toughness."),
)
CallableToughness = _get_attr(
    prop_tough_mod,
    "CallableToughness",
    _missing_class("CallableToughness", "Expected in fracture_utils.Upropagation.toughness."),
)

MaximumHoopStressLaw = _get_attr(
    prop_dir_mod,
    "MaximumHoopStressLaw",
    _missing_class("MaximumHoopStressLaw", "Expected in fracture_utils.Upropagation.direction."),
)

CandidateEvaluator = _get_attr(
    prop_eval_mod,
    "CandidateEvaluator",
    _missing_class("CandidateEvaluator", "Expected in fracture_utils.Upropagation.evaluate."),
)

CrackPropagator = _get_attr(
    prop_prog_mod,
    "CrackPropagator",
    _missing_class("CrackPropagator", "Expected in fracture_utils.Upropagation.propagator."),
)

# -----------------------------
# Validation \(COD \+ SIF\)
# -----------------------------
val_cod_mod = _try_import("fracture_utils.Uvalidation.validation_cod")
val_sif_mod = _try_import("fracture_utils.Uvalidation.validation_sif")

CODProfile = _get_attr(val_cod_mod, "CODProfile")
CODCaseResult = _get_attr(val_cod_mod, "CODCaseResult")
SIFCaseResult = _get_attr(val_sif_mod, "SIFCaseResult")
CODSweepResult = _get_attr(val_cod_mod, "CODSweepResult")
run_cod_sweep = _get_attr(val_cod_mod, "run_cod_sweep")
plot_cod_overlay = _get_attr(val_cod_mod, "plot_cod_overlay")
plot_cod_error_vs_knob = _get_attr(val_cod_mod, "plot_cod_error_vs_knob")
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
    # propagation
    "PropagationConfig",
    "ConstantToughness",
    "CallableToughness",
    "MaximumHoopStressLaw",
    "CandidateEvaluator",
    "CrackPropagator",
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
    # processor/results + SIF
    "DCEResultsNetworkV4",
    "cotterell_rice_K",
    "sif_from_cod_fit",
    "solve_K_polyline",
    "rotate_sifs",
    "euclid_tip_fit_from_edge",
    "estimate_KI_KII_from_jumps_near_tip",
    "PKProcessor",
    "DisplacementSIF",
    "DisplacementGradientSIF",
    "AnalysisSIF",
    "EnergyReleaseRateSIF",
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
]
