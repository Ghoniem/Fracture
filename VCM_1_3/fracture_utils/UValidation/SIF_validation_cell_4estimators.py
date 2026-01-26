# --- SIF Validation Cell (4 estimators: tip, jump, PK-window, J-contour) ---
import os, sys
from pathlib import Path
import numpy as np

# -------------------------
# Ensure repo root on sys.path (robust)
# -------------------------
cwd = Path(os.getcwd()).resolve()
repo_root = None
for p in [cwd] + list(cwd.parents):
    if (p / "fracture_utils").is_dir():
        repo_root = p
        break
if repo_root is None:
    repo_root = Path(r"C:\Users\Owner\Documents\Repos\Fracture\VCM_1_3")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print("Using repo_root =", repo_root)
print("fracture_utils importable =", (repo_root / "fracture_utils").is_dir())

from preamble import *
enable_autoreload()

# -------------------------
# Imports: updated results + PK + validation + SIF utilities
# -------------------------
from fracture_utils.Uprocessor.results_v1_PATHIND_updated import DCEResultsNetworkV4
from fracture_utils.Uprocessor.PK_PATHIND_updated import PKProcessor

from fracture_utils.UValidation.validation_sif_pathind import (
    SIFMeasures4, run_sif_sweep4, plot_sif_error_vs_knob4
)

from fracture_utils.Uprocessor.SIF import (
    sif_from_cod_fit,
    estimate_KI_KII_from_jumps_near_tip,
)

# -------------------------
# Output directory
# -------------------------
out_dir = Path(r"C:\Users\Owner\Documents\Repos\Fracture\VariationalCracks\output\SIF_validation")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Benchmark: straight crack, Mode I
# -------------------------
mm = 1e-3
a_geom = 5.0 * mm
sigma0 = 100e6

crack_meta = dict(
    a=a_geom,
    sigma=sigma0,
    E=200e9,
    nu=0.30,
    plane="strain",  # "strain" or "stress"
    edge_index=0,
)

# -------------------------
# Build a single-edge network
# -------------------------
vertices = np.array([
    [0, -a_geom, 0.0],
    [1, +a_geom, 0.0],
], dtype=float)

connectivity = np.array([[0, 1]], dtype=int)

network = CrackNetworkV4.from_vertices_connectivity(
    vertices=vertices,
    connectivity=connectivity,
    Nv_max=3,
    validate=True,
)

plane_stress = ("stress" in str(crack_meta["plane"]).lower())
material = Material(E=crack_meta["E"], nu=crack_meta["nu"], plane_stress=plane_stress)
applied  = AppliedStress(sigma_xx=0.0, sigma_yy=+crack_meta["sigma"], sigma_xy=0.0)
calc = DCENetworkStaticV4(material=material, network=network, applied=applied)

# -------------------------
# Solver hook: returns 4-estimator bundle
# -------------------------
def run_solver_sif(knobs, crack_meta) -> SIFMeasures4:
    sol = calc.solve(**knobs)
    res = DCEResultsNetworkV4(calc, sol)

    edge_index = crack_meta.get("edge_index", 0)
    if edge_index is Ellipsis:
        edge_index = 0
    edge_index = int(edge_index)

    # COD/CSD at panel midpoints
    x, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(
        edge_index=edge_index,
        enforce_global_tip_zero=True,
    )
    extra = dict(extra) if isinstance(extra, dict) else {}
    a_fit = float(extra.get("a_edge", float(crack_meta["a"])))

    # Elastic params for SIF utilities
    E  = float(res.calc.material.E)
    nu = float(res.calc.material.nu)
    mu = E / (2.0 * (1.0 + nu))
    plane_stress_local = bool(getattr(res.calc.material, "plane_stress", False))
    kappa  = (3.0 - nu) / (1.0 + nu) if plane_stress_local else (3.0 - 4.0 * nu)
    Eprime = E if plane_stress_local else (E / (1.0 - nu**2))

    # 1) Tip-fit
    KI_tip, KII_tip, meta_tip = sif_from_cod_fit(
        x=x, COD=COD, CSD=CSD,
        a=a_fit, mu=mu, kappa=kappa,
        tip="right",
        window="fixed",
        rho_min=1e-6,
        rho_max=0.80,
        min_pts=10,
        n_fit=400,
        two_term=True,
    )

    # 2) Jump estimator
    KI_jump, KII_jump = estimate_KI_KII_from_jumps_near_tip(
        x=x, COD=COD, CSD=CSD,
        a=a_fit,
        Eprime=Eprime,
        side="right",
        frac_window=0.08,
    )

    # 3) PK window estimator (Mode-I robust option)
    pk = PKProcessor(res)
    out_pk = pk.kink_angle_from_PK_window(
        edge_index=edge_index,
        at="end",
        window_panels=12,
        exclude_self=True,
        return_degrees=False,
        mode_I_only=True,
    )
    if out_pk is None:
        KI_pk = float("nan"); KII_pk = float("nan"); meta_pk = {"error":"kink_angle_from_PK_window returned None"}
    else:
        _theta, KI_pk, KII_pk, meta_pk = out_pk

    # 4) Path-independent contour J estimator (Mode I)
    outJ = pk.K_from_J_contour(
        edge_index=edge_index,
        tip="end",
        R=0.25 * a_fit,
        n_theta=241,
        add_remote=True,
    )
    if outJ is None:
        KI_path = float("nan"); meta_path = {"error":"K_from_J_contour returned None"}
    else:
        KI_path, meta_path = outJ

    return SIFMeasures4(
        KI_tip=float(KI_tip),
        KI_jump=float(KI_jump),
        KI_pk=float(KI_pk),
        KI_path=float(KI_path),
        KII_tip=float(KII_tip),
        KII_jump=float(KII_jump),
        KII_pk=float(KII_pk),
        KII_path=0.0,
        meta=dict(meta_tip=meta_tip, meta_pk=meta_pk, meta_path=meta_path, a_fit=a_fit, Eprime=Eprime),
    )

# -------------------------
# Analytical SIF
# -------------------------
def analytical_sif(crack_meta):
    a = float(crack_meta["a"])
    sigma = float(crack_meta["sigma"])
    return float(sigma * np.sqrt(np.pi * a)), 0.0

# -------------------------
# Knob sweep
# -------------------------
ne_half_list = [40, 80, 120, 160]

knob_sweep = []
for ne in ne_half_list:
    knob_sweep.append(dict(
        ne_half=int(ne),
        crack_mode="half",
        representation="cheb_quad",
        node_distribution="tip_dense",
        nq_col=4,
        nq_stress=3,
        ridge=1e-10,
        r0_factor=1e-3,
        solver_option="parametrized_crack",
        parametrization="polyline",
    ))

# -------------------------
# Run sweep + plot errors for 4 estimators
# -------------------------
sif_sweep = run_sif_sweep4(
    crack_meta=crack_meta,
    knob_sweep=knob_sweep,
    run_solver_sif=run_solver_sif,
    analytical_sif=analytical_sif,
)

plot_sif_error_vs_knob4(sif_sweep, knob="ne_half", title="Mode I SIF error vs ne_half (4 estimators)")
