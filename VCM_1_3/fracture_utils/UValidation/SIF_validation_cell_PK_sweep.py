# --- SIF Validation Cell (4 estimators; PK uses window_frac + normal_offset_frac sweep) ---
import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
# Imports: updated results + updated PK + SIF utilities
# -------------------------
from fracture_utils.Uprocessor.results_v1_PATHIND_updated import DCEResultsNetworkV4
from fracture_utils.Uprocessor.PK_PATHIND_updated_v2 import PKProcessor

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
# Analytical SIF
# -------------------------
def KI_analytic(a, sigma):
    return float(sigma * np.sqrt(np.pi * a))

# -------------------------
# Run one solve and return the 3 "good" estimators + PK (with tunables)
# -------------------------
def solve_and_estimate(ne_half: int, *, window_frac: float, normal_offset_frac: float, representation: str = "cheb_quad"):
    knobs = dict(
        ne_half=int(ne_half),
        crack_mode="half",
        representation=str(representation),
        node_distribution="tip_dense",
        nq_col=4,
        nq_stress=3,
        ridge=1e-10,
        r0_factor=1e-3,
        solver_option="parametrized_crack",
        parametrization="polyline",
    )
    sol = calc.solve(**knobs)
    res = DCEResultsNetworkV4(calc, sol)

    edge_index = int(crack_meta.get("edge_index", 0))

    x, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(edge_index=edge_index, enforce_global_tip_zero=True)
    extra = dict(extra) if isinstance(extra, dict) else {}
    a_fit = float(extra.get("a_edge", float(crack_meta["a"])))

    E  = float(res.calc.material.E)
    nu = float(res.calc.material.nu)
    mu = E / (2.0 * (1.0 + nu))
    plane_stress_local = bool(getattr(res.calc.material, "plane_stress", False))
    kappa  = (3.0 - nu) / (1.0 + nu) if plane_stress_local else (3.0 - 4.0 * nu)
    Eprime = E if plane_stress_local else (E / (1.0 - nu**2))

    # Tip-fit
    KI_tip, KII_tip, _ = sif_from_cod_fit(
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

    # Jump
    KI_jump, KII_jump = estimate_KI_KII_from_jumps_near_tip(
        x=x, COD=COD, CSD=CSD,
        a=a_fit,
        Eprime=Eprime,
        side="right",
        frac_window=0.08,
    )

    # PK (Mode I: KI = sqrt(E' * J1))
    pk = PKProcessor(res)
    F = pk.F_PK_window(
        edge_index=edge_index,
        at="end",
        exclude_self=True,
        window_panels=0,               # force frac-mode
        window_frac=float(window_frac),
        min_panels=12,
        normal_offset_frac=float(normal_offset_frac),
    )
    if F is None:
        KI_pk = float("nan")
        J1 = float("nan")
    else:
        # local tangent from reconstructor
        _, _, _, extra2 = res.reconstruct_cod_csd_panel_midpoints(edge_index=edge_index, enforce_global_tip_zero=False)
        extra2 = dict(extra2)
        ex = np.asarray(extra2["ex"], float).reshape(2,)
        J1 = float(np.dot(np.asarray(F, float).reshape(2,), ex))
        KI_pk = float(np.sqrt(max(Eprime * J1, 0.0)))

    # Mode I analytic
    KI_ref = KI_analytic(a_fit, float(crack_meta["sigma"]))

    return dict(
        ne_half=int(ne_half),
        a_fit=a_fit,
        Eprime=Eprime,
        KI_ref=KI_ref,
        KI_tip=float(KI_tip),
        KI_jump=float(KI_jump),
        KI_pk=float(KI_pk),
        J1=float(J1),
    )

# -------------------------
# Sweep ne_half with a small grid of PK tunables and plot
# -------------------------
ne_half_list = [40, 80, 120, 160]

# Choose a few combinations (3–5) to avoid crowding
pk_variants = [
    dict(window_frac=0.10, normal_offset_frac=0.0,    label="wf=0.10, off=0"),
    dict(window_frac=0.15, normal_offset_frac=0.0,    label="wf=0.15, off=0"),
    dict(window_frac=0.20, normal_offset_frac=0.0,    label="wf=0.20, off=0"),
    dict(window_frac=0.15, normal_offset_frac=1e-6,   label="wf=0.15, off=1e-6"),
    dict(window_frac=0.15, normal_offset_frac=5e-6,   label="wf=0.15, off=5e-6"),
]

# Precompute baseline tip/jump once per ne_half (reuse first variant result)
baseline = {}
pk_results = {v["label"]: [] for v in pk_variants}

for ne in ne_half_list:
    r0 = solve_and_estimate(ne, window_frac=pk_variants[0]["window_frac"], normal_offset_frac=pk_variants[0]["normal_offset_frac"])
    baseline[ne] = dict(KI_ref=r0["KI_ref"], KI_tip=r0["KI_tip"], KI_jump=r0["KI_jump"])

    for v in pk_variants:
        r = solve_and_estimate(ne, window_frac=v["window_frac"], normal_offset_frac=v["normal_offset_frac"])
        pk_results[v["label"]].append(r)

# -------------------------
# Plot error vs ne_half
# -------------------------
def err_pct(K, Kref):
    return 100.0 * (K - Kref) / Kref

fig, ax = plt.subplots()

# tip/jump
ax.plot(ne_half_list, [err_pct(baseline[ne]["KI_tip"], baseline[ne]["KI_ref"]) for ne in ne_half_list],
        marker="o", label="K_from_tip")
ax.plot(ne_half_list, [err_pct(baseline[ne]["KI_jump"], baseline[ne]["KI_ref"]) for ne in ne_half_list],
        marker="o", label="K_from_jump")

# PK variants
for lab, arr in pk_results.items():
    ax.plot(ne_half_list, [err_pct(a["KI_pk"], a["KI_ref"]) for a in arr], marker="o", label=f"K_from_PK ({lab})")

ax.set_xlabel("ne_half")
ax.set_ylabel("Error (%)")
ax.set_title("Mode I SIF error vs ne_half (PK: window_frac & normal_offset_frac sweep)")
ax.grid(True)
ax.legend()
plt.show()
