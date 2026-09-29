"""Generate the tutorial input workbooks and notebooks, and optionally execute them.

    python tutorials/_build_tutorials.py            # write workbooks + notebooks
    python tutorials/_build_tutorials.py --execute  # ... and run both notebooks in place

Edit the workbook rows or the cell sources below, then re-run. Hand edits to the
generated .xlsx / .ipynb files are overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tutorial_utils as tu  # noqa: E402

T1 = HERE / "01_inclined_crack_sif"
T2 = HERE / "02_crack_kinking_growth"


# ═════════════════════════════════════════════════════════════════════════════
# Input workbooks  —  rows are (Parameter, Symbol, Value, Units, Note)
# ═════════════════════════════════════════════════════════════════════════════

MATERIAL = [
    ("Young's modulus", "E", 200e9, "Pa", "steel-like elastic constants"),
    ("Poisson's ratio", "nu", 0.30, "-", ""),
    ("Plane condition", "plane", "strain", "-", "strain | stress"),
]

T1_SHEETS = {
    "Material": MATERIAL,
    "Geometry": [
        ("Crack half-length", "a", 5e-3, "m", "straight crack centred at the origin"),
        ("Crack angles for the sweep", "beta_list_deg", "0, 15, 30, 45, 60, 75, 90", "deg",
         "angle of the crack from the x-axis, counter-clockwise"),
        ("Crack angle for the convergence study", "beta_conv_deg", 30, "deg",
         "mixed mode, so both K_I and K_II are tested"),
    ],
    "Loading": [
        ("Remote stress sigma_xx", "sigma_xx", 0.0, "Pa", "uniform far-field stress"),
        ("Remote stress sigma_yy", "sigma_yy", 100e6, "Pa", "uniaxial tension along y"),
        ("Remote stress sigma_xy", "sigma_xy", 0.0, "Pa", ""),
    ],
    "Solver": [
        ("Engine", "engine", "python", "-", "python | cpp (falls back to python if bem_cpp is missing)"),
        ("Crack panels (production)", "n_crack_elements", 80, "-", "used for the angle sweep and COD profile"),
        ("Crack panels (convergence study)", "n_list", "20, 40, 80, 160", "-", "panel counts for the convergence study"),
        ("Crack mode", "crack_mode", "full", "-", "full (dipolar) | half (polar)"),
        ("Parametrization", "parametrization", "polyline", "-", "polyline | cspline"),
        ("SIF fit window", "rmax_frac", 0.25, "-", "fit window as a fraction of the crack length"),
        ("SIF fit minimum points", "min_pts", 8, "-", ""),
        ("Two-term SIF fit", "two_term", 1, "-", "1 = sqrt(r) plus r^(3/2) term"),
    ],
    "Verification": [
        ("Tolerance on K error", "tol_K", 0.05, "-", "max |K - K_exact| / K0 at the production panel count"),
        ("Tolerance on COD error", "tol_COD", 0.05, "-", "RMS COD error / max exact COD"),
    ],
    "Output": [
        ("Run label", "label", "inclined_crack", "-", "run directory is output/<timestamp>_<label>_<git-hash>"),
        ("Figure resolution", "dpi", 150, "dpi", ""),
    ],
}

T2_SHEETS = {
    "Material": MATERIAL + [
        ("Fracture toughness", "Kc", 1e6, "Pa*sqrt(m)", "tips grow while K_eff > Kc"),
    ],
    "Geometry": [
        ("Initial crack half-length", "a", 5e-3, "m", "straight crack centred at the origin"),
        ("Initial crack angle", "beta_deg", 45, "deg", "angle of the crack from the x-axis"),
    ],
    "Loading": [
        ("Remote stress sigma_xx", "sigma_xx", 0.0, "Pa", "uniform far-field stress"),
        ("Remote stress sigma_yy", "sigma_yy", 100e6, "Pa", "uniaxial tension along y"),
        ("Remote stress sigma_xy", "sigma_xy", 0.0, "Pa", ""),
    ],
    "Propagation": [
        ("Growth steps", "n_steps", 6, "-", "maximum number of growth increments"),
        ("Step fraction", "f_fixed", 0.10, "-", "delta_a = f_fixed * current crack length"),
        ("Kink clamp", "max_kink_deg", 0.0, "deg", "0 disables the clamp; the first kink is never clamped"),
        ("K_II noise threshold", "kii_noise_ratio", 0.05, "-", "theta = 0 when |K_II|/(|K_I|+|K_II|) is below this"),
        ("Length limit", "L_limit_mm", 40.0, "mm", "stop when the total crack length exceeds this"),
    ],
    "Solver": [
        ("Engine", "engine", "python", "-", "python | cpp (falls back to python if bem_cpp is missing)"),
        ("Crack panels", "n_crack_elements", 60, "-", "panels over the whole crack path"),
        ("Crack mode", "crack_mode", "half", "-", "half (polar) is the more stable choice for kinked paths"),
        ("Parametrization", "parametrization", "polyline", "-", "polyline | cspline"),
        ("SIF fit window", "rmax_frac", 0.25, "-", "fit window as a fraction of the crack length"),
        ("SIF fit minimum points", "min_pts", 8, "-", ""),
        ("Two-term SIF fit", "two_term", 1, "-", "1 = sqrt(r) plus r^(3/2) term"),
    ],
    "Verification": [
        ("Tolerance on first kink angle", "tol_theta_deg", 1.0, "deg", "|theta_1 - theta_MTS(exact K)|"),
        ("Tolerance on final K_II ratio", "tol_KII_ratio", 0.05, "-", "|K_II| / K_I at the last step (path is mode I)"),
        ("Tolerance on path symmetry", "tol_symmetry", 1e-6, "-", "point-symmetry residual of the two tips / a"),
    ],
    "Output": [
        ("Run label", "label", "kinking_growth", "-", "run directory is output/<timestamp>_<label>_<git-hash>"),
        ("Figure resolution", "dpi", 150, "dpi", ""),
        ("Stress grid points", "stress_n_grid", 160, "-", "per side, for the final stress contour"),
        ("Stress plot extent", "stress_extent", 1.6, "-", "plot half-width as a multiple of the crack extent"),
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# Shared notebook cells
# ═════════════════════════════════════════════════════════════════════════════

SETUP = r'''import sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TUT_DIR = Path.cwd()                          # this tutorial's folder
sys.path.insert(0, str(TUT_DIR.parent))       # tutorials/ -> tutorial_utils
import tutorial_utils as tu
tu.setup_paths()                              # vcem/ -> fracture_utils

from fracture_utils.Usolver.network import CrackNetworkV4
from fracture_utils.Usolver.material import Material, AppliedStress

# Read the workbook, apply this run's overrides, and flatten to {symbol: value}
case_path = (TUT_DIR / CASE_XLSX).resolve()
tables = tu.read_workbook(case_path)
overrides_applied = tu.apply_overrides(tables, OVERRIDES)
P = tu.flat_values(tables)

ENGINE, engine_note = tu.resolve_engine(P["engine"])
print(f"case     : {case_path.name}  (sha256 {tu.sha256(case_path)[:12]})")
print(f"sheets   : {', '.join(tables)}")
print(f"overrides: {overrides_applied or 'none'}")
print(f"engine   : {ENGINE}  — {engine_note}")
'''

MATERIAL_LOAD = r'''plane_stress = str(P["plane"]).lower().startswith("stress")
material = Material(E=float(P["E"]), nu=float(P["nu"]), plane_stress=plane_stress)
applied = AppliedStress(sigma_xx=float(P["sigma_xx"]), sigma_yy=float(P["sigma_yy"]),
                        sigma_xy=float(P["sigma_xy"]))

E, nu = float(P["E"]), float(P["nu"])
E_prime = E if plane_stress else E / (1.0 - nu**2)       # effective modulus
a = float(P["a"])
sig = np.array([[P["sigma_xx"], P["sigma_xy"]], [P["sigma_xy"], P["sigma_yy"]]], float)
K0 = float(P["sigma_yy"]) * np.sqrt(np.pi * a)           # reference SIF, sigma_yy * sqrt(pi a)


def straight_crack(beta_deg: float, half_length: float = a) -> CrackNetworkV4:
    """One straight crack through the origin at angle beta from the x-axis."""
    b = np.radians(beta_deg)
    c, s = np.cos(b), np.sin(b)
    V = np.array([[0, -half_length * c, -half_length * s],
                  [1, +half_length * c, +half_length * s]], float)
    return CrackNetworkV4.from_vertices_connectivity(
        vertices=V, connectivity=np.array([[0, 1]]), Nv_max=3, validate=True)


def exact_K(beta_deg: float, half_length: float = a):
    """Griffith crack in an infinite plate: K_I = sigma_nn sqrt(pi a), K_II = sigma_nt sqrt(pi a)."""
    b = np.radians(beta_deg)
    t = np.array([np.cos(b), np.sin(b)])      # crack tangent
    n = np.array([-np.sin(b), np.cos(b)])     # crack normal
    root = np.sqrt(np.pi * half_length)
    return float(n @ sig @ n) * root, float(t @ sig @ n) * root


print(f"E' = {E_prime/1e9:.1f} GPa,  a = {a*1e3:.2f} mm,  K0 = {K0/1e6:.3f} MPa*sqrt(m)")
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


# ═════════════════════════════════════════════════════════════════════════════
# Tutorial 01 — inclined crack, stress intensity factors
# ═════════════════════════════════════════════════════════════════════════════

def tutorial_01():
    return [
        md(r'''
# Tutorial 01 — Stress intensity factors of an inclined crack

**What you will learn**

- how a VCEM run is organized: an input workbook, this notebook as the orchestrator, and a
  stamped output directory with `provenance.md`;
- how to build a crack network, solve for its dislocation density, and extract $K_I$ and $K_{II}$;
- how to verify the solver against an exact solution and check that it converges.

**The problem.** A straight crack of length $2a$ sits in an infinite plate under a remote stress
$\boldsymbol{\sigma}^\infty$. The crack makes an angle $\beta$ with the $x$-axis. With unit
tangent $\mathbf{t} = (\cos\beta, \sin\beta)$ and normal $\mathbf{n} = (-\sin\beta, \cos\beta)$,
the exact stress intensity factors are

$$K_I = (\mathbf{n}\cdot\boldsymbol{\sigma}^\infty\cdot\mathbf{n})\sqrt{\pi a}, \qquad
  K_{II} = (\mathbf{t}\cdot\boldsymbol{\sigma}^\infty\cdot\mathbf{n})\sqrt{\pi a}.$$

Under uniaxial tension $\sigma_{yy}$ these become $K_I = \sigma\sqrt{\pi a}\cos^2\beta$ and
$K_{II} = \sigma\sqrt{\pi a}\sin\beta\cos\beta$. The crack opening displacement (COD) is the ellipse
$\delta(x) = (4\sigma_{nn}/E')\sqrt{a^2 - x^2}$.

**How VCEM solves it.** The crack is a distribution of edge dislocations on $N$ panels. The
dislocation densities are found by minimizing the traction on the crack faces subject to closure
(the KKT problem). $K_I$ and $K_{II}$ then come from a fit of the near-tip COD and sliding
displacement to $\sqrt{r}$.

**Runtime.** About three minutes with the Python engine.
'''),
        md("## 1. Pick the case workbook\n\n"
           "All parameters live in `input/inclined_crack.xlsx`. To change a value for one run only, "
           "put its `Symbol` in `OVERRIDES`. Overrides are recorded in `provenance.md`."),
        code(r'''
CASE_XLSX = "input/inclined_crack.xlsx"
OVERRIDES = {}              # e.g. {"n_crack_elements": 120, "sigma_xx": 50e6}
SHOW_PLOTS = True
QUIET = True                # hide solver log lines
'''),
        md("## 2. Setup: paths, workbook, engine"),
        code(SETUP),
        md(r'''
## 3. Build the model

`Material` holds the elastic constants and the plane condition. `AppliedStress` is the uniform
remote stress. `straight_crack(beta)` builds a one-edge crack network, and `exact_K(beta)` returns
the exact solution for comparison.
'''),
        code(MATERIAL_LOAD + r'''

from fracture_utils.Usolver.parametrization import DCENetworkStaticV4
from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Uprocessor.SIF_cod import DisplacementSIF

SOLVER_KW = dict(crack_mode=P["crack_mode"], parametrization=P["parametrization"], engine=ENGINE)
FIT_KW = dict(rmax_frac=float(P["rmax_frac"]), min_pts=int(P["min_pts"]), two_term=bool(P["two_term"]))


def solve_crack(beta_deg: float, n_panels: int):
    """Solve one crack; return (K_I, K_II, results) at the right-hand tip."""
    net = straight_crack(beta_deg)
    calc = DCENetworkStaticV4(material, net, applied)
    with tu.quiet(QUIET):
        sol = calc.solve(n_crack_elements=int(n_panels), **SOLVER_KW)
    res = DCEResultsNetworkV4(calc, sol)
    tip = np.array([net.vertices[1].x, net.vertices[1].y])
    KI, KII, *_ = DisplacementSIF.euclid_from_edge(
        res, edge_index=0, tip_xy=tip, a_fit=2 * a, material=material, rotate=False, **FIT_KW)
    return float(KI), float(KII), res
'''),
        md(r'''
## 4. Angle sweep

Solve the crack at every angle in `beta_list_deg` with the production panel count and compare
with the exact solution. Errors are normalized by $K_0 = \sigma_{yy}\sqrt{\pi a}$, because $K_I$ or
$K_{II}$ is zero at some angles.
'''),
        code(r'''
n_prod = int(P["n_crack_elements"])
t0 = time.perf_counter()
sweep = []
for beta in tu.parse_list(P["beta_list_deg"]):
    KI, KII, _ = solve_crack(beta, n_prod)
    KIe, KIIe = exact_K(beta)
    sweep.append(dict(beta_deg=beta, KI_num=KI, KII_num=KII, KI_exact=KIe, KII_exact=KIIe,
                      err_KI=abs(KI - KIe) / K0, err_KII=abs(KII - KIIe) / K0))
t_sweep = time.perf_counter() - t0

print(f"{'beta':>6} {'K_I/K0':>9} {'exact':>7} {'K_II/K0':>9} {'exact':>7} {'err_I':>7} {'err_II':>7}")
for r in sweep:
    print(f"{r['beta_deg']:6.1f} {r['KI_num']/K0:9.4f} {r['KI_exact']/K0:7.4f} "
          f"{r['KII_num']/K0:9.4f} {r['KII_exact']/K0:7.4f} {r['err_KI']:7.2%} {r['err_KII']:7.2%}")
print(f"\n{len(sweep)} solves in {t_sweep:.1f} s")
'''),
        md(r'''
## 5. Convergence with panel count

Repeat one mixed-mode case with more and more panels. The slope of the log–log line is the
observed convergence rate.

Expect the error to level off at around 1–2 %. Panels are clustered towards the tips, and panels
shorter than $10^{-3}L$ are merged to keep the KKT system well conditioned
(`min_panel_length_ratio`), so adding panels stops refining the tip. Set `QUIET = False` to see how
many panels each solve merges.
'''),
        code(r'''
beta_c = float(P["beta_conv_deg"])
KIe_c, KIIe_c = exact_K(beta_c)
t0 = time.perf_counter()
conv = []
for n in tu.parse_list(P["n_list"], int):
    KI, KII, _ = solve_crack(beta_c, n)
    conv.append(dict(n_panels=n, KI_num=KI, KII_num=KII,
                     err_KI=abs(KI - KIe_c) / K0, err_KII=abs(KII - KIIe_c) / K0))
t_conv = time.perf_counter() - t0

ns = np.array([c["n_panels"] for c in conv], float)
eI = np.array([c["err_KI"] for c in conv])
rate = -np.polyfit(np.log(ns), np.log(eI), 1)[0]
for c in conv:
    print(f"n = {c['n_panels']:4d}   err K_I = {c['err_KI']:.3%}   err K_II = {c['err_KII']:.3%}")
print(f"observed convergence rate of K_I: error ~ n^-{rate:.2f}   ({t_conv:.1f} s)")
'''),
        md(r'''
## 6. Crack opening profile

Reconstruct the COD along the crack for $\beta = 0$ (pure mode I) and compare it with the exact
ellipse.
'''),
        code(r'''
_, _, res0 = solve_crack(0.0, n_prod)
x, COD, CSD, extra = res0.reconstruct_cod_csd_panel_midpoints(edge_index=0, enforce_global_tip_zero=True)
x, COD = np.asarray(x, float), np.asarray(COD, float)
a_edge = float(extra.get("a_edge", a)) if isinstance(extra, dict) else a
x_c = x - 0.5 * (x.min() + x.max()) if x.min() >= 0 else x      # centre on the crack midpoint
COD_exact = 4.0 * float(P["sigma_yy"]) / E_prime * np.sqrt(np.clip(a**2 - x_c**2, 0, None))
cod_rms = float(np.sqrt(np.mean((COD - COD_exact) ** 2)) / COD_exact.max())
print(f"{x.size} points, max COD = {COD.max()*1e6:.3f} um (exact {COD_exact.max()*1e6:.3f} um), "
      f"RMS error = {cod_rms:.2%} of max")
'''),
        md("## 7. Output directory and plots\n\n"
           "Every run gets its own directory `output/YYYYMMDD_HHMMSS_<label>_<git-hash>/`. Runs never "
           "overwrite each other."),
        code(r'''
run_dir, run_label = tu.make_run_dir(TUT_DIR, P["label"])
plots = run_dir / "plots"
dpi = int(P["dpi"])
print("run directory:", run_dir.relative_to(TUT_DIR))

b = np.array([r["beta_deg"] for r in sweep])
bb = np.linspace(0, 90, 181)
ex = np.array([exact_K(v) for v in bb]) / K0

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(bb, ex[:, 0], "k-", lw=1.5, label=r"exact $K_I$")
ax.plot(bb, ex[:, 1], "k--", lw=1.5, label=r"exact $K_{II}$")
ax.plot(b, [r["KI_num"] / K0 for r in sweep], "o", color="tab:blue", label=r"VCEM $K_I$")
ax.plot(b, [r["KII_num"] / K0 for r in sweep], "s", color="tab:orange", label=r"VCEM $K_{II}$")
ax.set(xlabel=r"crack angle $\beta$ [deg]", ylabel=r"$K / \sigma\sqrt{\pi a}$",
       title=f"Inclined crack, N = {n_prod} panels")
ax.grid(alpha=0.3); ax.legend()
fig.savefig(plots / "K_vs_angle.png", dpi=dpi, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(ns, eI * 100, "o-", label=r"$K_I$")
ax.loglog(ns, np.array([c["err_KII"] for c in conv]) * 100, "s-", label=r"$K_{II}$")
ax.set(xlabel="crack panels N", ylabel=r"error [% of $K_0$]",
       title=rf"Convergence at $\beta$ = {beta_c:g}°  (rate ≈ {rate:.2f})")
from matplotlib.ticker import NullFormatter, ScalarFormatter
ax.set_xticks(ns); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(ScalarFormatter())
ax.grid(alpha=0.3, which="both"); ax.legend()
fig.savefig(plots / "K_convergence.png", dpi=dpi, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 3.5))
xs = np.linspace(-a, a, 400)
ax.plot(xs * 1e3, 4 * float(P["sigma_yy"]) / E_prime * np.sqrt(a**2 - xs**2) * 1e6, "k-", label="exact")
ax.plot(x_c * 1e3, COD * 1e6, ".", color="tab:blue", label="VCEM")
ax.set(xlabel="x [mm]", ylabel=r"COD [$\mu$m]", title=r"Crack opening, $\beta$ = 0")
ax.grid(alpha=0.3); ax.legend()
fig.savefig(plots / "COD_profile.png", dpi=dpi, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
plt.close("all")
'''),
        md(r'''
## 8. Results, diagnostics and provenance

- `angle_sweep.csv`, `convergence.csv`, `cod_profile.csv` — the result tables;
- `summary.csv` — one row per run, so the summaries of many runs concatenate into one table;
- `diagnostics.txt` — pass/fail against the tolerances in the `Verification` sheet;
- `provenance.md` — every input table with overrides applied, the user selections, the solver
  settings, and the runtime, machine and git state.
'''),
        code(r'''
tu.write_csv(run_dir / "angle_sweep.csv", sweep)
tu.write_csv(run_dir / "convergence.csv", conv)
tu.write_csv(run_dir / "cod_profile.csv",
             [dict(x_m=float(xi), COD_m=float(ci), COD_exact_m=float(ce)) for xi, ci, ce in zip(x_c, COD, COD_exact)])

max_err = max(max(r["err_KI"], r["err_KII"]) for r in sweep)
checks = [
    ("max K error over the angle sweep", max_err, float(P["tol_K"]), "x K0"),
    ("RMS COD error", cod_rms, float(P["tol_COD"]), "x max COD"),
]
ok = tu.write_diagnostics(run_dir / "diagnostics.txt", run_label, checks)

tu.write_csv(run_dir / "summary.csv", [dict(
    run=run_label, engine=ENGINE, n_panels=n_prod, a_m=a, sigma_yy_Pa=float(P["sigma_yy"]),
    max_K_err=max_err, cod_rms_err=cod_rms, conv_rate_KI=rate, status="PASS" if ok else "FAIL")])

tu.write_provenance(
    run_dir, run_label, title="VCEM tutorial 01 (inclined crack SIF)", tables=tables,
    overrides=overrides_applied,
    user_selections=dict(tutorial="01_inclined_crack_sif", case_workbook=tu.repo_relative(case_path),
                         case_sha256=tu.sha256(case_path), engine_requested=P["engine"],
                         engine_used=ENGINE, engine_note=engine_note, domain="infinite plate",
                         betas_deg=tu.parse_list(P["beta_list_deg"]), beta_conv_deg=beta_c),
    solver_config={**{f"solve.{k}": v for k, v in SOLVER_KW.items()},
                   **{f"sif_fit.{k}": v for k, v in FIT_KW.items()},
                   "sif_fit.a_fit": 2 * a, "sif_fit.method": "DisplacementSIF.euclid_from_edge",
                   "n_crack_elements": n_prod, "n_list": tu.parse_list(P["n_list"], int)},
    run_stats=dict(wall_clock_sweep_s=t_sweep, wall_clock_convergence_s=t_conv,
                   n_solves=len(sweep) + len(conv) + 1, run_status="completed",
                   diagnostics="PASS" if ok else "FAIL"))

print((run_dir / "diagnostics.txt").read_text())
print("files:", sorted(p.name for p in run_dir.iterdir()))
'''),
        md(r'''
## 9. Try this

1. **Biaxial load.** Set `OVERRIDES = {"sigma_xx": 100e6}`. Under equal biaxial tension $K_I$ no
   longer depends on $\beta$ and $K_{II} = 0$.
2. **Coarse mesh.** Set `{"n_crack_elements": 20}` and watch the diagnostics fail. Where does
   the error come from?
3. **Plane stress.** Set `{"plane": "stress"}`. The SIFs do not change, but the COD does,
   through $E'$.
4. **Compare runs.** Open two `provenance.md` files side by side: every difference between the runs is
   listed there.

Next: [Tutorial 02](../02_crack_kinking_growth/02_crack_kinking_growth.ipynb) lets the crack grow.
'''),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Tutorial 02 — kinking and growth of an inclined crack
# ═════════════════════════════════════════════════════════════════════════════

def tutorial_02():
    return [
        md(r'''
# Tutorial 02 — Kinking and growth of an inclined crack

**What you will learn**

- how the propagation loop works: solve, extract $K_I, K_{II}$ at every tip, choose a direction,
  extend, repeat;
- how the maximum tangential stress (MTS) criterion predicts the kink angle;
- how to record and verify a growth run with the same input → notebook → output contract as the
  full Brazilian-disk simulations.

**The problem.** A straight crack of length $2a$ at $\beta = 45°$ sits in an infinite plate under
uniaxial tension $\sigma_{yy}$. It is loaded in mixed mode, $K_I = K_{II} = \tfrac12\sigma\sqrt{\pi a}$.
Each tip kinks by the MTS angle

$$\theta = 2\arctan\frac{K_I - \sqrt{K_I^2 + 8K_{II}^2}}{4K_{II}},$$

which is $-53.13°$ for $K_I = K_{II}$. The crack then turns until it runs perpendicular to the
load, where $K_{II} \to 0$ (pure mode I).

**The growth rule.** A tip grows while $K_{\rm eff} = \sqrt{K_I^2 + K_{II}^2} > K_c$. At each step
every tip is evaluated on the *same* equilibrium solution, then all tips advance together by
$\Delta a = f\,L$, where $L$ is the current crack length.

**Runtime.** About one minute with the Python engine.
'''),
        md("## 1. Pick the case workbook\n\n"
           "All parameters live in `input/kinking_growth.xlsx`. Put a `Symbol` in `OVERRIDES` to change "
           "it for this run only."),
        code(r'''
CASE_XLSX = "input/kinking_growth.xlsx"
OVERRIDES = {}              # e.g. {"beta_deg": 30, "n_steps": 15}
SHOW_PLOTS = True
QUIET = True                # hide solver log lines
'''),
        md("## 2. Setup: paths, workbook, engine"),
        code(SETUP),
        md(r'''
## 3. Build the model

Next to the material, load and initial crack, the propagation loop needs four pieces:

| Object | Role |
|---|---|
| `CandidateEvaluator` | solves the network and fits $K_I, K_{II}$ at a tip |
| `MaximumHoopStressLaw` | turns $(K_I, K_{II})$ into a kink angle |
| `ConstantToughness` | supplies $K_c$ |
| `CrackPropagator` | runs one growth increment for all tips |
'''),
        code(MATERIAL_LOAD + r'''

from fracture_utils.Upropagation.config import PropagationConfig
from fracture_utils.Upropagation.evaluate import CandidateEvaluator
from fracture_utils.Upropagation.propagator import CrackPropagator
from fracture_utils.Upropagation.direction import MaximumHoopStressLaw
from fracture_utils.Upropagation.toughness import ConstantToughness

SOLVER_KW = dict(n_crack_elements=int(P["n_crack_elements"]), crack_mode=P["crack_mode"],
                 parametrization=P["parametrization"], engine=ENGINE)
FIT_KW = dict(rmax_frac=float(P["rmax_frac"]), min_pts=int(P["min_pts"]), two_term=bool(P["two_term"]))

law = MaximumHoopStressLaw(kii_noise_ratio=float(P["kii_noise_ratio"]))
evaluator = CandidateEvaluator(material=material, applied=applied, solver_kwargs=SOLVER_KW,
                               direction_law=law, **FIT_KW)
prop_cfg = PropagationConfig(step_mode="fixed_step", f_fixed=float(P["f_fixed"]))

net0 = straight_crack(float(P["beta_deg"]))
propagator = CrackPropagator(cfg=prop_cfg, evaluator=evaluator,
                             toughness=ConstantToughness(float(P["Kc"])), direction_law=law,
                             max_kink_deg=float(P["max_kink_deg"]),
                             initial_vertex_ids=[v.id for v in net0.vertices])


def mts_angle(KI, KII):
    """Closed-form MTS kink angle (rad)."""
    if abs(KII) < 1e-12 * max(abs(KI), 1.0):
        return 0.0
    return 2.0 * np.arctan((KI - np.sqrt(KI**2 + 8 * KII**2)) / (4 * KII))


KI0, KII0 = exact_K(float(P["beta_deg"]))
theta_exact = np.degrees(mts_angle(KI0, KII0))
print(f"exact initial SIFs: K_I = {KI0/1e6:.3f}, K_II = {KII0/1e6:.3f} MPa*sqrt(m)"
      f"  ->  MTS kink angle {theta_exact:.2f} deg")
'''),
        md(r'''
## 4. Growth loop

Each call to `grow_one_increment` returns the new network and one report per tip. The loop stops
after `n_steps`, when no tip is above toughness, or when the crack exceeds `L_limit_mm`.
'''),
        code(r'''
def tip_positions(net):
    """Map 'left' / 'right' -> xy of the two degree-1 vertices."""
    tips = [v for v in net.vertices if len(v.edges) == 1]
    tips.sort(key=lambda v: v.x)
    return {"left": np.array([tips[0].x, tips[0].y]), "right": np.array([tips[-1].x, tips[-1].y])}


def crack_length(net):
    xy = {v.id: np.array([v.x, v.y]) for v in net.vertices}
    return float(sum(np.linalg.norm(xy[e.v1] - xy[e.v0]) for e in net.edges))


net = net0
paths = {k: [p] for k, p in tip_positions(net).items()}
steps, stop_reason = [], "n_steps reached"
t0 = time.perf_counter()
for k in range(int(P["n_steps"])):
    pos = tip_positions(net)
    with tu.quiet(QUIET):
        result = propagator.grow_one_increment(net)
    for rep in result.reports:
        side = "right" if rep.which == "end" else "left"
        steps.append(dict(step=k + 1, tip=side, x_m=float(pos[side][0]), y_m=float(pos[side][1]),
                          KI=float(rep.KI), KII=float(rep.KII), Keff=float(rep.keff),
                          theta_deg=float(np.degrees(rep.theta)), delta_a_m=float(rep.delta_a),
                          grew=bool(rep.grew), reason=rep.reason or "", L_before_m=crack_length(net)))
    net = result.network_new
    for side, p in tip_positions(net).items():
        paths[side].append(p)
    r = [s for s in steps if s["step"] == k + 1 and s["tip"] == "right"][0]
    print(f"step {k+1:2d}: K_I = {r['KI']/1e6:6.2f}  K_II = {r['KII']/1e6:6.2f} MPa*sqrt(m)  "
          f"theta = {r['theta_deg']:7.2f} deg  L = {crack_length(net)*1e3:6.2f} mm")
    if not any(rep.grew for rep in result.reports):
        stop_reason = "all tips below toughness"; break
    if crack_length(net) * 1e3 > float(P["L_limit_mm"]):
        stop_reason = "length limit reached"; break
t_grow = time.perf_counter() - t0
print(f"\nstopped: {stop_reason}  ({len(steps)//2} steps, {t_grow:.1f} s)")
'''),
        md(r'''
## 5. Check the physics

- The first kink angle should match the MTS angle computed from the exact initial SIFs.
- By the last step the crack should run in mode I, so $|K_{II}|/K_I$ should be small.
- The problem is point-symmetric about the origin, so the two tips must mirror each other exactly.
- As the crack turns horizontal, $K_I$ approaches $\sigma\sqrt{\pi c}$, where $c$ is the half-width of
  the crack's projection on the $x$-axis. This is a guide, not an exact result.

**Watch the K_II column.** After the first kink the crack should run in pure mode I. The computed
$K_{II}$ instead drifts by one to two percent of $K_I$ with every new segment. This is a discretization
error of the kinked path, not physics. Once $|K_{II}|/(|K_I|+|K_{II}|)$ passes the noise threshold
(`kii_noise_ratio`, 5 %), MTS applies a small corrective kink. Both tips make the same correction,
and $K_{II}$ returns to near zero. That is the MTS law's self-correcting behaviour at work.
'''),
        code(r'''
right = [s for s in steps if s["tip"] == "right"]
left = [s for s in steps if s["tip"] == "left"]
theta_1 = right[0]["theta_deg"]
err_theta = abs(abs(theta_1) - abs(theta_exact))
KII_ratio = abs(right[-1]["KII"]) / abs(right[-1]["KI"])
PL, PR = np.array(paths["left"]), np.array(paths["right"])
sym = float(np.max(np.abs(PL + PR)) / a)

for s in right:
    c_half = abs(s["x_m"])                  # right tip x ~ projected half-width (symmetric crack)
    s["KI_projected"] = float(P["sigma_yy"]) * np.sqrt(np.pi * c_half)

print(f"first kink angle   : {theta_1:.2f} deg   (MTS {theta_exact:.2f} deg, diff {err_theta:.3f})")
print(f"final |K_II| / K_I : {KII_ratio:.3%}")
print(f"symmetry residual  : {sym:.2e} x a")
'''),
        md("## 6. Output directory and plots"),
        code(r'''
run_dir, run_label = tu.make_run_dir(TUT_DIR, P["label"])
plots = run_dir / "plots"
dpi = int(P["dpi"])
print("run directory:", run_dir.relative_to(TUT_DIR))

# Crack path
fig, ax = plt.subplots(figsize=(6.5, 4))
V0 = np.array([[v.x, v.y] for v in net0.vertices])
ax.plot(V0[:, 0] * 1e3, V0[:, 1] * 1e3, "k-", lw=3, label="initial crack")
for side, col in (("left", "tab:blue"), ("right", "tab:red")):
    p = np.array(paths[side]) * 1e3
    ax.plot(p[:, 0], p[:, 1], "o-", color=col, ms=3, label=f"{side} tip path")
ax.set(xlabel="x [mm]", ylabel="y [mm]", title=r"Crack path under $\sigma_{yy}$ (load is vertical)")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.savefig(plots / "crack_path.png", dpi=dpi, bbox_inches="tight")

# SIFs and kink angle along the growth
st = np.array([s["step"] for s in right])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(st, [s["KI"] / 1e6 for s in right], "o-", label=r"$K_I$")
ax1.plot(st, [s["KII"] / 1e6 for s in right], "s-", label=r"$K_{II}$")
ax1.plot(st, [s["KI_projected"] / 1e6 for s in right], "k--", lw=1, label=r"$\sigma\sqrt{\pi c}$ (projected)")
ax1.axhline(float(P["Kc"]) / 1e6, color="gray", ls=":", label=r"$K_c$")
ax1.set(xlabel="step", ylabel=r"K [MPa$\sqrt{m}$]", title="Right tip"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
ax2.plot(st, [s["theta_deg"] for s in right], "o-", color="tab:purple")
ax2.axhline(theta_exact, color="k", ls="--", lw=1, label=f"MTS, step 1 ({theta_exact:.1f}°)")
ax2.set(xlabel="step", ylabel=r"kink angle $\theta$ [deg]", title="Kink angle per step")
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(plots / "SIF_and_kink_vs_step.png", dpi=dpi, bbox_inches="tight")

# Stress field around the final crack
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4
from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Uplotter.core import DCEPlotterV4
from fracture_utils.Uplotter.opts import StressPlotOptsV4

calc_f = DCENetworkStaticV4(material, net, applied)
with tu.quiet(QUIET):
    res_f = DCEResultsNetworkV4(calc_f, calc_f.solve(**SOLVER_KW))
DCEPlotterV4(res_f, out_dir=plots).plot_stress_components_global(
    opts=StressPlotOptsV4(extent_factor=float(P["stress_extent"]), n_grid=int(P["stress_n_grid"]),
                          add_remote=True, n_bands=40, vmax_factor=2.0),
    components=("syy",), show=SHOW_PLOTS)

if SHOW_PLOTS:
    plt.show()
plt.close("all")
'''),
        md(r'''
## 7. Results, diagnostics and provenance

- `growth_steps.csv` — one row per tip per step: position, $K_I$, $K_{II}$, $K_{\rm eff}$, kink angle, $\Delta a$;
- `crack_path.csv` — the tip trajectories;
- `summary.csv`, `diagnostics.txt`, `provenance.md` — as in Tutorial 01.
'''),
        code(r'''
tu.write_csv(run_dir / "growth_steps.csv", steps)
tu.write_csv(run_dir / "crack_path.csv",
             [dict(point=i, tip=side, x_m=float(p[0]), y_m=float(p[1]))
              for side in ("left", "right") for i, p in enumerate(paths[side])])

checks = [
    ("first kink angle vs MTS", err_theta, float(P["tol_theta_deg"]), "deg"),
    ("final |K_II| / K_I", KII_ratio, float(P["tol_KII_ratio"]), "-"),
    ("path point-symmetry residual", sym, float(P["tol_symmetry"]), "x a"),
]
ok = tu.write_diagnostics(run_dir / "diagnostics.txt", run_label, checks)

tu.write_csv(run_dir / "summary.csv", [dict(
    run=run_label, engine=ENGINE, beta_deg=float(P["beta_deg"]), n_steps=len(right),
    L_final_mm=crack_length(net) * 1e3, theta_1_deg=theta_1, theta_MTS_deg=theta_exact,
    KI_final=right[-1]["KI"], KII_final=right[-1]["KII"], stop_reason=stop_reason,
    status="PASS" if ok else "FAIL")])

tu.write_provenance(
    run_dir, run_label, title="VCEM tutorial 02 (crack kinking and growth)", tables=tables,
    overrides=overrides_applied,
    user_selections=dict(tutorial="02_crack_kinking_growth", case_workbook=tu.repo_relative(case_path),
                         case_sha256=tu.sha256(case_path), engine_requested=P["engine"],
                         engine_used=ENGINE, engine_note=engine_note, domain="infinite plate",
                         direction_law="MaximumHoopStressLaw", toughness="ConstantToughness"),
    solver_config={**{f"solve.{k}": v for k, v in SOLVER_KW.items()},
                   **{f"sif_fit.{k}": v for k, v in FIT_KW.items()},
                   "propagation.step_mode": prop_cfg.step_mode, "propagation.f_fixed": prop_cfg.f_fixed,
                   "propagation.simultaneous_tip_growth": prop_cfg.simultaneous_tip_growth,
                   "propagation.max_kink_deg": float(P["max_kink_deg"]),
                   "propagation.kii_noise_ratio": float(P["kii_noise_ratio"])},
    run_stats=dict(wall_clock_growth_s=t_grow, n_steps=len(right), stop_reason=stop_reason,
                   L_final_mm=crack_length(net) * 1e3, run_status="completed",
                   diagnostics="PASS" if ok else "FAIL"))

print((run_dir / "diagnostics.txt").read_text())
print("files:", sorted(p.name for p in run_dir.iterdir()))
'''),
        md(r'''
## 8. Try this

1. **Other angles.** Set `{"beta_deg": 30}` or `{"beta_deg": 60}`. The first kink follows the MTS
   angle, and the path always turns perpendicular to the load.
2. **Tough material.** Set `{"Kc": 15e6}`. With $K_{\rm eff} < K_c$ the crack does not grow at all.
3. **Kink clamp.** Set `{"max_kink_deg": 10}`. After the first kink, the path is smoothed.
4. **Step size.** Halve `f_fixed` and double `n_steps`. Is the path converged?
5. **Crack mode.** Set `{"crack_mode": "full"}`. The $K_{II}$ drift is larger in the dipolar mode,
   and the corrective kinks come sooner and are no longer symmetric.

**Next.** The same loop, with the plate replaced by a finite disk solved by the BEM, drives the
full Brazilian-disk simulations in `vcem/notebooks/SimulationsCpp.ipynb`.
'''),
    ]


# ═════════════════════════════════════════════════════════════════════════════

def write_notebook(path: Path, cells) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, path)


def execute(path: Path) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbf.read(path, as_version=4)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(path.parent)}})
    nbf.write(nb, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run the notebooks in place after writing them")
    ap.add_argument("--only", choices=["01", "02"], help="build (and execute) one tutorial only")
    args = ap.parse_args()

    jobs = {
        "01": (T1 / "input" / "inclined_crack.xlsx", T1_SHEETS, T1 / "01_inclined_crack_sif.ipynb", tutorial_01),
        "02": (T2 / "input" / "kinking_growth.xlsx", T2_SHEETS, T2 / "02_crack_kinking_growth.ipynb", tutorial_02),
    }
    for key, (xlsx, sheets, nb, cells) in jobs.items():
        if args.only and key != args.only:
            continue
        tu.write_workbook(xlsx, sheets)
        write_notebook(nb, cells())
        print("wrote", xlsx.relative_to(HERE), "and", nb.relative_to(HERE))
        if args.execute:
            print("executing", nb.name, "...")
            execute(nb)
    print("done")


if __name__ == "__main__":
    main()
