"""
Notebook preamble (repo-root)

Curated high-frequency VCM helpers for notebooks.

Adds (v16):
- Robust resolution of PKProcessor (moved out of legacy SIF.py during refactor)
  by searching likely Uprocessor modules.
- If not found, installs a callable stub with an actionable error message.
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


def _resolve_any(mods, names):
    for m in mods:
        if m is None:
            continue
        for n in names:
            v = getattr(m, n, None)
            if v is not None:
                return v
    return None


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

# After refactor: SIF_cod + SIF_pk (legacy: SIF)
sif_cod_mod = _try_import_many([
    "fracture_utils.Uprocessor.SIF_cod",
    "fracture_utils.Uprocessor.sif_cod",
])
sif_pk_mod = _try_import_many([
    "fracture_utils.Uprocessor.SIF_pk",
    "fracture_utils.Uprocessor.sif_pk",
])
sif_legacy_mod = _try_import_many([
    "fracture_utils.Uprocessor.SIF",
    "fracture_utils.Uprocessor.sif",
])

# --- OO facades (preferred notebook API)
DisplacementSIF = _resolve_any([sif_cod_mod], ['DisplacementSIF'])
DisplacementGradientSIF = _resolve_any([sif_cod_mod], ['DisplacementGradientSIF'])
AnalysisSIF = _resolve_any([sif_cod_mod], ['AnalysisSIF'])
EnergyReleaseRateSIF = _resolve_any([sif_pk_mod], ['EnergyReleaseRateSIF'])

if DisplacementSIF is None or DisplacementGradientSIF is None or AnalysisSIF is None:
    # If the repo hasn't been updated to the OO facade yet, provide informative stubs.
    class _MissingSIFClass:
        def __init__(self, name: str):
            self._name = name
        def __getattr__(self, attr):
            raise AttributeError(
                f"{self._name} is not available. Update fracture_utils/Uprocessor/SIF_cod.py to the OO facade version."
            )
    DisplacementSIF = DisplacementSIF or _MissingSIFClass('DisplacementSIF')
    DisplacementGradientSIF = DisplacementGradientSIF or _MissingSIFClass('DisplacementGradientSIF')
    AnalysisSIF = AnalysisSIF or _MissingSIFClass('AnalysisSIF')

if EnergyReleaseRateSIF is None:
    class _MissingERRClass:
        def __getattr__(self, attr):
            raise AttributeError(
                "EnergyReleaseRateSIF is not available. Update fracture_utils/Uprocessor/SIF_pk.py to include the class facade."
            )
    EnergyReleaseRateSIF = _MissingERRClass()


cotterell_rice_K = _resolve_any([sif_cod_mod, sif_legacy_mod], ["cotterell_rice_K"])
sif_from_cod_fit = _resolve_any([sif_cod_mod, sif_legacy_mod], ["sif_from_cod_fit"])
solve_K = _resolve_any([sif_cod_mod, sif_legacy_mod], ["solve_K"])
rotate_sifs = _resolve_any([sif_cod_mod, sif_legacy_mod], ["rotate_sifs"])
euclid_tip_fit_from_edge = _resolve_any([sif_cod_mod, sif_legacy_mod], ["euclid_tip_fit_from_edge"])

# Polyline wrapper
_SOLVE_K_POLYLINE_NAMES = [
    "solve_K_polyline",
    "solve_K_from_polyline",
    "solve_K_poly",
    "solve_K_polyline_tipfit",
    "solve_K_arc_polyline",
    "solve_K_polyline_from_edge",
]
solve_K_polyline = _resolve_any([sif_cod_mod, sif_legacy_mod], _SOLVE_K_POLYLINE_NAMES)
if solve_K_polyline is None:
    def solve_K_polyline(*args, **kwargs):  # type: ignore
        raise NameError(
            "solve_K_polyline is not available.\n"
            "Expected in fracture_utils.Uprocessor.SIF_cod (preferred) or legacy SIF.\n"
            "Try:\n"
            "  import fracture_utils.Uprocessor.SIF_cod as m\n"
            "  [n for n in dir(m) if 'solve' in n.lower() and 'poly' in n.lower()]\n"
        )

# Jump-based estimator
estimate_KI_KII_from_jumps_near_tip = _resolve_any(
    [sif_cod_mod, sif_legacy_mod],
    ["estimate_KI_KII_from_jumps_near_tip", "estimate_K_from_jumps_near_tip"]
)
if estimate_KI_KII_from_jumps_near_tip is None:
    def estimate_KI_KII_from_jumps_near_tip(*args, **kwargs):  # type: ignore
        raise NameError(
            "estimate_KI_KII_from_jumps_near_tip was not found.\n"
            "Try:\n"
            "  import fracture_utils.Uprocessor.SIF_cod as m\n"
            "  [n for n in dir(m) if 'jump' in n.lower()]\n"
        )

# --- PKProcessor: moved out of SIF modules during refactor in some branches.
pkproc_mod = _try_import_many([
    "fracture_utils.Uprocessor.PK_processor",
    "fracture_utils.Uprocessor.pk_processor",
    "fracture_utils.Uprocessor.PK",
    "fracture_utils.Uprocessor.pk",
    "fracture_utils.Uprocessor.processor_pk",
    "fracture_utils.Uprocessor.pk_window",
])

_PKPROCESSOR_NAMES = ["PKProcessor", "PKProcessorV4", "PKWindowProcessor", "PKWindow"]
PKProcessor = _resolve_any([pkproc_mod, sif_pk_mod, sif_legacy_mod], _PKPROCESSOR_NAMES)

if PKProcessor is None:
    def PKProcessor(*args, **kwargs):  # type: ignore
        raise NameError(
            "PKProcessor class was not found.\n"
            "Searched for: " + ", ".join(_PKPROCESSOR_NAMES) + "\n"
            "in likely modules:\n"
            "  fracture_utils.Uprocessor.PK_processor / pk_processor / PK / pk / processor_pk / pk_window\n\n"
            "To locate it in your repo, run:\n"
            "  import pkgutil, fracture_utils.Uprocessor as up\n"
            "  [m.name for m in pkgutil.iter_modules(up.__path__) if 'pk' in m.name.lower()]\n"
            "then import the likely module and inspect dir().\n"
        )

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
    "solve_K_polyline",
    "rotate_sifs",
    "euclid_tip_fit_from_edge",
    "estimate_KI_KII_from_jumps_near_tip",
    "PKProcessor",
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
    "DisplacementSIF",
    "DisplacementGradientSIF",
    "AnalysisSIF",
    "EnergyReleaseRateSIF",

]
