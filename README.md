# VCEM

**Variational Crack Element Method for crack networks in finite elastic bodies.**

VCEM is a simulation suite for the nucleation, growth, branching, coalescence and
fragmentation of crack networks in two-dimensional elastic solids. The reference
problem is the **diametral (Brazilian) disk compression test**: a tensile field along
the loaded diameter drives pre-existing flaws to kink, branch and link up, until the
network spans the disk and the specimen fragments. The same machinery is being
extended to thermally shocked and polycrystalline **tungsten**, the plasma-facing
armour of fusion devices.

The distinguishing feature of the code is that a crack is not a pair of free surfaces
in a mesh. Each crack is a **continuous distribution of edge dislocations** whose
Burgers-vector density is found by constrained energy minimization — a KKT
least-squares problem that enforces traction-free faces, closure and junction
compatibility. The finite body enters through a **boundary-only Kelvin BEM**, so only
the outer boundary and the cracks are discretised. The cracks form a **graph**:
junctions, kinks and intersections are vertices, and growth changes the topology rather
than a volume mesh.

- **Languages** — Python (formulation, orchestration, post-processing) and C++17
  (Eigen + OpenMP backend for the BEM, KKT and topology hot paths, exposed through
  pybind11).
- **Version** — 4.1.1 ([release notes](CHANGELOG.md)). Releases are git tags `vX.Y.Z`;
  see [Versions and releases](#versions-and-releases).
- **License** — MIT.
- **Name** — *VCM* (Variational Crack Method) is the 1.x–3.x Python line; *VCEM*
  (Variational Crack Element Method) is the 4.x line with the C++ backend.
- **Primary references** — copies are kept in [`docs/Publications/`](docs/Publications/).
  - N. M. Ghoniem, *A constrained variational method for crack networks with
    criterion-based evolution*, J. Mater. Sci.: Mater. Theory **10** (2026) 6 —
    the formulation.
    [doi:10.1186/s41313-026-00076-6](https://doi.org/10.1186/s41313-026-00076-6)
  - N. M. Ghoniem, *Crack network evolution in finite domains*, J. Mater. Sci.:
    Mater. Theory **10** (2026) 9 — validation, the Brazilian disk, and fragmentation
    as a connectivity phase transition.
    [doi:10.1186/s41313-026-00077-5](https://doi.org/10.1186/s41313-026-00077-5)
- **Theory notes** — [`docs/VCM Code/`](docs/VCM%20Code/) (`VCM_Theory.ipynb`,
  `VCM_Architecture.ipynb`, `Variational_Fracture.pdf`).

---

## Contents

1. [Capabilities](#1-capabilities)
2. [Methodology](#2-methodology)
3. [Repository structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [Inputs](#6-inputs)
7. [Outputs and provenance](#7-outputs-and-provenance)
8. [Examples, tests and verification](#8-examples-tests-and-verification)
9. [Computational aspects](#9-computational-aspects)
10. [Documentation](#10-documentation)
11. [Publications and how to cite](#11-publications-and-how-to-cite)
12. [Repository conventions](#12-repository-conventions)
13. [License and contact](#13-license-and-contact)

---

## 1. Capabilities

| Capability | Where | Notes |
|---|---|---|
| Dislocation-density crack representation | `Usolver/solve_kernels.py`, `Usolver/build_discretize.py` | panels of constant opening ($b_I$) and sliding ($b_{II}$) density, clustered at tips, junctions and kinks |
| Arbitrary crack networks | `Usolver/network.py`, `build_dipolar_polyline.py`, `build_polar_polyline.py` | straight, polyline, circular-arc and cubic-spline edges; Y- and star-junctions, trees, closed cycles |
| Constrained energy minimization | `Usolver/KKT.py`, `Usolver/constraints.py` | traction-free faces as least squares, closure and junction compatibility as hard (or soft-penalty) constraints |
| Finite-body coupling by BEM | `Ubem/bem_solver.py`, `Ubem/direct_kkt_coupling.py` | constant-element Kelvin BEM with mixed displacement/traction BCs; one-way, iterative or monolithic (augmented KKT) coupling |
| Stress intensity factors | `Uprocessor/SIF_cod.py`, `Uprocessor/SIF_pk.py` | $K_I$, $K_{II}$ from near-tip COD/CSD fits, or from Peach–Koehler forces ($J$) |
| Crack propagation | `Upropagation/` | maximum-tangential-stress kink law, toughness-driven simultaneous tip growth, adaptive step control |
| Coalescence and topology change | `Ugenerator/generator.py`, `crack_network_simplifier.py` | intersection detection and splitting creates new junctions; near-collinear edges merged, short edges removed |
| Fragmentation detection | `Ugenerator/spanning_cluster.py` | a connected component touching the boundary at two or more points is a spanning (fragmenting) cluster |
| Sub-critical crack isolation | `Usolver/isolation.py` | isolated cracks below $K_{Ic}$ are replaced by their analytical Westergaard opening, with hysteresis |
| Random flaw populations | `Ugenerator/random_cracks.py`, `input/seed_crack_network.py` | truncated-normal or log-normal lengths, uniform centres and orientations in a disk or rectangle |
| Network generation and CAD import | `Ugenerator/generator.py`, `cad_import.py` | seeded junction/tip networks (uniform, clustered, grid; nearest-neighbour, random, Delaunay); DXF/IGES/STEP/SVG/STL |
| Fracture energetics | `Ubem/crack_energetics.py` | compliance, stored energy, potential energy, energy-release rate, load- and displacement-controlled |
| Network order parameters | `Ubem/crack_energetics.py` | crack density $\alpha$, spanning fraction $P_\infty$, orientation order $Q$, per step |
| C++ backend | `cpp/` | BEM, edge-dislocation operators, KKT solve, segment intersection; selected per run with `engine='cpp'` |
| Provenance-stamped output | `input/provenance.py` | every run writes a timestamped directory with the resolved configuration, workbook hash, git revision and machine |

**Under development.** An elastoplastic extension for two tungsten problems — residual
plastic strain from a constrained thermal shock driving brittle cracking on
cool-down, and grain-boundary sliding with crack-tip dislocation shielding in a 2D
polycrystal (the ductile-to-brittle transition). Plastic strain, boundary sliding and
tip plastic zones map onto the existing edge-dislocation kernel and KKT solve; no
discrete dislocation dynamics is added. See
[`docs/VCEM Elastoplasticity/plasticity_plan.tex`](docs/VCEM%20Elastoplasticity/plasticity_plan.tex).
A self-learning run campaign for the fragmentation paper is planned in
[`docs/Fragmentation_Phase_Transition/vcem_fragmentation_simulation_plan.tex`](docs/Fragmentation_Phase_Transition/vcem_fragmentation_simulation_plan.tex).

---

## 2. Methodology

### 2.1 Cracks as dislocation distributions

A crack is represented by a continuous distribution of edge dislocations along its
path. The path is discretised into panels $j$, each with unit normal $\mathbf{n}_j$,
tangent $\mathbf{t}_j$ and two constant densities: $b_I$ (opening, along
$\mathbf{n}$) and $b_{II}$ (sliding, along $\mathbf{t}$). The unknown vector
$\mathbf{q}$ collects these densities (plus junction jump unknowns, §2.3). The
plane-strain edge-dislocation kernel is

$$\sigma_{xx} = -\frac{\mu\, b_x}{2\pi(1-\nu)}\,\frac{y\,(3x^2+y^2)}{r^4}, \quad \ldots$$

with plane stress recovered by $\nu \to \nu/(1+\nu)$. The crack opening and sliding
displacements (COD, CSD) are the running integrals of $b_I$ and $b_{II}$.

### 2.2 Traction-free crack faces

At collocation points $\mathbf{x}_i$ the traction from all dislocation panels must
cancel the applied traction:

$$\sum_j \int_{\Gamma_j} \mathbf{n}_i \cdot \boldsymbol{\sigma}^{\rm disl}(\mathbf{x}_i;\, \mathbf{n}_j b_I + \mathbf{t}_j b_{II})\, ds \cdot \\{\mathbf{n}_i, \mathbf{t}_i\\}
 = -\\{\mathbf{n}_i\cdot\boldsymbol{\sigma}^{\rm app}\cdot\mathbf{n}_i,\ \mathbf{t}_i\cdot\boldsymbol{\sigma}^{\rm app}\cdot\mathbf{n}_i\\},$$

written $\mathbf{K}\mathbf{q} = \mathbf{r}$. The applied field
$\boldsymbol{\sigma}^{\rm app}$ is either a constant remote stress or a spatially
varying field — in practice, the BEM solution for the uncracked body (§2.5).

### 2.3 Constraints and junctions

Two crack parametrizations are available:

| Mode | Unknowns | Constraints $\mathbf{C}\mathbf{q} = \mathbf{d}$ |
|---|---|---|
| `full` (dipolar) | panel densities per polyline | closure $\int b_I\,ds = \int b_{II}\,ds = 0$ on every polyline |
| `half` (polar) | panel densities per branch, plus jump vectors $\mathbf{J}$ at branch ends | endpoint compatibility, vector closure at junctions of degree ≥ 3 (projected on the bisector normal at kinks), gauge $\mathbf{J}_{\rm tip} = 0$ |

In `half` mode the junction model is `strict` (hard constraints), `core` (per-branch
jump unknowns) or `soft` (compatibility as penalty rows $\sqrt{\eta}\,\mathbf{P}\mathbf{q}
\approx 0$). An optional active set enforces non-negative opening, COD ≥ 0.

### 2.4 The KKT problem

The densities minimize the traction residual subject to the constraints,

$$\min_{\mathbf{q}}\ \lVert \mathbf{K}\mathbf{q} - \mathbf{r} \rVert^2 + \lambda\lVert\mathbf{q}\rVert^2
\quad \text{s.t.} \quad \mathbf{C}\mathbf{q} = \mathbf{d}.$$

The Python solver uses a null-space method: $\mathbf{C}$ is compressed to full row
rank by SVD, $\mathbf{q} = \mathbf{q}_p + \mathbf{Z}\mathbf{y}$ with $\mathbf{Z}$ from a
complete QR of $\mathbf{C}^T$, and the reduced problem is solved by least squares. This
keeps the conditioning at $\mathrm{cond}(\mathbf{K}\mathbf{Z}) \approx
\mathrm{cond}(\mathbf{K})$, rather than the squared-and-worse conditioning of the
saddle-point form. The C++ backend solves the bordered KKT system by partial-pivot LU
with a BDCSVD fallback, behind a runtime backend switch.

### 2.5 The finite body: boundary elements

The uncracked body is solved by a constant-element collocation BEM on its outer
boundary,

$$\mathbf{c}\,\mathbf{u}(P) + \int_\Gamma \mathbf{p}^{*}\mathbf{u}\,d\Gamma = \int_\Gamma \mathbf{u}^{*}\mathbf{t}\,d\Gamma, \qquad \mathbf{c} = \tfrac12\mathbf{I},$$

with Kelvin kernels $\mathbf{u}^{*}, \mathbf{p}^{*}$, mixed displacement/traction BCs
and rigid-body closure. Interior stress follows from the displacement-gradient kernels.
For the Brazilian disk, the platen load is applied as pressure over an arc of half-angle
`arc_half_angle_deg` on a boundary mesh graded towards the platens.

Three coupling strategies connect cracks and body:

| Coupling | Description |
|---|---|
| one-way | the cracks see the BEM field of the uncracked body; the body does not see the cracks |
| iterative | the reversed crack tractions $-\boldsymbol{\sigma}^{\rm crack}\cdot\mathbf{n}$ are applied back to the BEM boundary every `BEM_correction_frequency` steps |
| direct (augmented KKT) | one monolithic system for $\mathbf{z} = [\mathbf{q};\, \mathbf{u}_{bc};\, \mathbf{t}_{bc}]$ |

The augmented system couples crack equilibrium $\mathbf{K}\mathbf{q} +
\mathbf{N}_{bc}\mathbf{y} = \mathbf{r}$, the BEM equation $(\mathbf{C}+\mathbf{H})\mathbf{u}
- \mathbf{G}\mathbf{t} = 0$, and boundary compatibility $\mathbf{S}_u(\mathbf{u}_{bc} +
\mathbf{M}_u\mathbf{q}) = \mathbf{S}_u\bar{\mathbf{u}}$,
$\mathbf{S}_t(\mathbf{t}_{bc} + \mathbf{M}_t\mathbf{q}) = \mathbf{S}_t\bar{\mathbf{t}}$,
where $\mathbf{M}_u$, $\mathbf{M}_t$ map crack densities to boundary displacement and
traction and $\mathbf{N}_{bc}$ maps boundary data to crack-face traction. Rows and
unknowns are scaled to physical units, and small ridges ($\lambda_q$, $\lambda_y$)
regularize the blocks.

### 2.6 Stress intensity factors

The primary estimator fits the near-tip COD and CSD in Euclidean distance
$r = \lVert\mathbf{x}-\mathbf{x}_{\rm tip}\rVert$ to
$A\sqrt{r/2\pi}\,(1 + B\,r)$ over a window $[r_{\min},\, r_{\max}]$ with
$r_{\max}$ = `rmax_frac` × crack half-length, giving

$$K_{I,II} = \frac{\mu}{\kappa+1}\,A_{I,II}, \qquad \kappa = 3-4\nu \ \text{(plane strain)},\ \ \frac{3-\nu}{1+\nu}\ \text{(plane stress)}.$$

An alternative sums Peach–Koehler forces $\mathbf{f} = (\boldsymbol{\sigma}\cdot\mathbf{b})
\times\mathbf{e}_z$ over panels near the tip and inverts
$J_1 = (K_I^2+K_{II}^2)/E'$, $J_2 = -2K_IK_{II}/E'$. Four estimators — tip fit, jump,
Peach–Koehler and path-independent — are compared in the validation module.

### 2.7 Propagation

Each growth step evaluates every active tip on the same equilibrium solution before any
geometry is changed, so growth is simultaneous and independent of tip order.

- **Direction** — maximum tangential stress,
  $\theta = 2\arctan\!\big[(K_I \pm \sqrt{K_I^2 + 8K_{II}^2})/(4K_{II})\big]$, taking the
  root that maximizes $\sigma_{\theta\theta}$. $\theta = 0$ when
  $|K_{II}|/(|K_I|+|K_{II}|) < 0.05$ to suppress zig-zag from noise, and $\theta$ is
  clamped to ±`max_kink_deg` except on a tip's first emission.
- **Driving force** — $K_{\rm eff} = \sqrt{K_I^2 + K_{II}^2}$ and growth force
  $g = \max(K_{\rm eff} - K_c(\mathbf{x}),\,0)$, with constant or position-dependent
  toughness.
- **Increment** — $\Delta a_i = \Delta s_{\rm ref}\, g_i / \max_k g_k$, with
  $\Delta s_{\rm ref}$ = `f_disk_radius` × $R$; an adaptive $f,\,2f,\,f/2$ step search is
  available.
- **Geometry and topology** — one straight segment per tip per step, global
  re-discretisation, intersection detection and splitting (coalescence creates new
  junction vertices), and periodic network simplification. Tips that leave the body are
  snapped to the boundary and arrested.
- **Termination** — all tips arrested or below toughness (default), all
  $K_{\rm eff} < K_c$, a spanning cluster (fragmentation), or step, length and runtime
  limits.

### 2.8 Energetics and order parameters

Per step, the code reports compliance $C = \delta/P$, stored energy $U = \tfrac12P^2C$,
potential energy, energy-release rate $G = (K_I^2+K_{II}^2)/E'$ with
$dC/dA = 2G/P^2$, and three network order parameters: crack density
$\alpha = L_{\rm tot}/A$, spanning fraction $P_\infty = L_{\rm span}/L_{\rm tot}$, and
orientation order $Q = \langle 2\cos^2\theta - 1\rangle$. These are the observables of
the fragmentation-as-phase-transition study, which asks whether long-range elasticity
places crack-network fragmentation in a universality class distinct from ordinary
percolation.

### 2.9 Units and conventions

SI throughout: metres, pascals, $K$ in Pa·√m (displayed as MPa·√m). Plane strain by
default; $E' = E/(1-\nu^2)$ in plane strain and $E$ in plane stress,
$\mu = E/2(1+\nu)$. Boundaries are counter-clockwise with outward normals; a pressure
$p$ gives traction $\mathbf{t} = -p\,\mathbf{n}$. Disk thickness $h$ enters only through
$p = P/(L h)$.

---

## 3. Repository structure

```
Fracture/
├── vcem/                       # The code — recommended entry point
│   ├── fracture_utils/             # Python package (v4.1.1)
│   │   ├── Usolver/                    # network graph, panels, kernels, constraints, KKT,
│   │   │                               # isolation, C++ dispatch (cpp_dispatch.py)
│   │   ├── Ubem/                       # BEM solver, BCs, coupling (one-way / iterative /
│   │   │                               # augmented), disk growth loop, energetics
│   │   ├── Upropagation/               # kink law, toughness, propagator, step control, remesh
│   │   ├── Ugenerator/                 # network generation, random cracks, intersections,
│   │   │                               # simplifier, spanning clusters, CAD import
│   │   ├── Uprocessor/                 # results reconstruction, SIFs (COD fit, PK), diagnostics
│   │   ├── Uplotter/                   # stress, network, connectivity, deformed-network plots
│   │   ├── Uvalidation/                # COD and SIF convergence sweeps
│   │   └── Udiagnostics/
│   ├── cpp/                        # C++17 backend (Eigen, OpenMP, pybind11)
│   │   ├── include/vcem/, src/         # kelvin, bem_solver, edge_dislocation,
│   │   │                               # crack_assemblers, kkt, edge_intersection
│   │   ├── bindings/                   # bem_cpp pybind11 module
│   │   ├── python/                     # bem_cpp package, parity tests, benchmarks, diagnostics
│   │   └── CMakeLists.txt
│   ├── input/                      # case workbook, loader, provenance, network seeder
│   ├── notebooks/                  # SimulationsCpp.ipynb — the main driver
│   ├── tools/                      # run videos, symmetry diagnostic
│   ├── preamble.py                 # notebook path and import helper
│   └── output/                     # timestamped run directories (gitignored)
├── _archive/                   # VCM_1_0 … VCM_3_4 and early multi-crack code (read-only)
├── tutorials/                  # two self-checking tutorials on simple crack problems
├── docs/
│   ├── Publications/                     # the two published VCEM papers (PDF)
│   ├── Fragmentation_Phase_Transition/   # manuscript and simulation-campaign plan
│   ├── VCEM Elastoplasticity/            # tungsten plasticity plan
│   ├── VCM Code/                         # theory, architecture and organization notes
│   ├── Data/                             # Lo (1978) branched/kinked crack reference data
│   └── Literature/                       # ~45 reference papers (PDF)
├── requirements.txt
└── LICENSE
```

### Module status

| Module | Status | Description |
|---|---|---|
| `vcem/` | **Active** | Excel-driven workflow, C++ backend, random flaw populations, fragmentation diagnostics. Recommended starting point. |
| `_archive/VCM_3_4/` | Archived (last Python baseline) | VCM 3.3 frozen for displacement-controlled network runs; carries the analytical validation outputs and `ValidationRunner.ipynb`. |
| `_archive/VCM_3_0` – `VCM_3_3` | Archived | BEM coupling (3.0), coupled evolution (3.1), disk compression (3.2), network energetics and topology operations (3.3). |
| `_archive/VCM_2_0` – `VCM_2_2` | Archived | Network evolution in uniform and non-uniform fields; kinked, branched and curved-crack validation; mesh-budget policy. |
| `_archive/VCM_1_0` – `VCM_1_3` | Archived | First VCM: arbitrary polyline networks with junctions and intersections, arc parametrization. |
| `_archive/Multi-Cracks/` | Archived | 2023 pre-VCM multi-crack runs. |

Archived modules are kept so that earlier results remain reproducible; they receive no
further development.

### Branches

| Branch | Contents |
|---|---|
| `main` | Released line. |
| `vcm_development` | Active development of VCEM. |
| `vcm_3_4_refactor` | Refactor of the VCM_3_4 Python package. |

### Versions and releases

VCEM uses [semantic versioning](https://semver.org). The version lives in git tags and
releases, never in folder names. The code is VCEM; the GitHub repository keeps its
original name, [`Ghoniem/Fracture`](https://github.com/Ghoniem/Fracture), and releases are
listed under its [Releases](https://github.com/Ghoniem/Fracture/releases) page.

| Where | Name |
|---|---|
| Git tag on the release commit | `v4.1.0` |
| GitHub release title | `VCEM 4.1.0` |
| Downloadable archive (attached to the release) | `VCEM-4.1.0.zip` |
| Separate local copy of a release | `VCEM-4.1.0` |

The version in `vcem/fracture_utils/__init__.py` (`__version__`) and in the C++ module
(`bem_cpp.__version__`) matches the latest release. Every release is listed in
[`CHANGELOG.md`](CHANGELOG.md). The history before 4.1.0 is VCM 1.0–3.4 (pure Python,
kept in `_archive/`) and VCEM 4.0 (the C++ port).

To reproduce a published result, check out its tag, `git checkout v4.1.0`, or download
that release's archive.

---

## 4. Installation

### 4.1 Python environment

```bash
git clone https://github.com/Ghoniem/Fracture.git
cd Fracture
python -m pip install -r requirements.txt
```

Requirements: Python 3.10 or newer (tested on 3.14) with `numpy`, `scipy`,
`pandas`, `matplotlib`, `networkx`, `jupyterlab` and `ipykernel`; `openpyxl` for the case
workbook; `imageio` and `imageio-ffmpeg` for run videos (the latter bundles ffmpeg); and
`pybind11` to build the C++ backend. `cvxopt`, `osqp` and `numba` are listed but optional.

The Python layer alone runs every case: the C++ backend is an accelerator, not a
requirement. If `bem_cpp` cannot be imported, the driver falls back to
`engine='python'` automatically.

On Windows with conda, run from an activated environment. Outside `conda activate`, the
MKL DLLs under `<env>/Library/bin` are not on the path and NumPy's LAPACK calls fail
with `0xc06d007f`; prepend that directory to `PATH` when calling the interpreter
directly.

### 4.2 C++ backend

Required:

- a C++17 compiler (MSVC 2022, GCC or Clang) and CMake ≥ 3.18;
- **Eigen 3.4.0**, expected in a `Libraries/` directory that is a sibling of the
  repository root (`Libraries/eigen-3.4.0/`);
- `pybind11` importable from the target Python interpreter.

OpenMP is optional and auto-detected; without it the BEM and KKT assembly run
single-threaded. MSVC's bundled OpenMP 2.0 is sufficient.

Build and install into the Python package:

```bash
cd vcem/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)"
cmake --build build --config Release --target bem_cpp
cmake --install build --config Release
```

On Windows add `-G "Visual Studio 17 2022" -A x64`. The install step copies the
extension (`bem_cpp.*.pyd` / `.so`) into `cpp/python/bem_cpp/`, where the notebook
imports it. The extension is not distributed with the repository. It is built for one
Python version (`cp314`, for example), so rebuild it after changing interpreter; also
delete `build/` after moving the repository, because the CMake cache records absolute paths. Smoke test:

```bash
python -c "import sys; sys.path.insert(0, 'vcem/cpp/python'); import bem_cpp; print(bem_cpp.__version__)"
```

See [`vcem/cpp/README.md`](vcem/cpp/README.md) for details.

---

## 5. Quick start

**New to VCEM?** Start with [`tutorials/`](tutorials/). Two short notebooks run the full
workbook → notebook → stamped output workflow on problems with exact answers, using the Python
engine only:

| Tutorial | What it does |
|---|---|
| [01 — Inclined crack SIFs](tutorials/01_inclined_crack_sif/01_inclined_crack_sif.ipynb) | $K_I$, $K_{II}$ and COD of an inclined crack against the exact solution, plus panel convergence |
| [02 — Kinking and growth](tutorials/02_crack_kinking_growth/02_crack_kinking_growth.ipynb) | a 45° crack kinks by the MTS angle and turns perpendicular to the load |

### 5.1 Notebook (recommended)

Open `vcem/notebooks/SimulationsCpp.ipynb`. It has three steps:

1. **Pick the case workbook** — `CASE_XLSX = "input/disk_compression_data.xlsx"`.
2. **Preamble, Excel load, engine setup** — loads the case, selects the engine from the
   workbook (falling back to Python if the extension is missing), and prints the active
   engine and OpenMP thread count.
3. **Run** — `run_disk_compression_2(cfg)`: the BEM baseline for the uncracked disk,
   then the growth steps with iterative or direct coupling, per-step figures, metrics,
   provenance and videos.

Setting `dry_run` in the workbook defines the run function without executing it.

### 5.2 Script

```python
from input.excel_io import load_case

cfg = load_case("input/disk_compression_data.xlsx")   # -> CaseConfig
print(cfg.engine, cfg.coupling_method, cfg.crack_growth.Kc)
```

`vcem/cpp/python/run_disk_compression_2.py` and `run_disk_compression_2_cpp.py`
run the same case end-to-end outside Jupyter.

### 5.3 A random flaw population

```bash
cd vcem
python input/seed_crack_network.py input/disk_compression_data.xlsx \
       --n 20 --mean-mm 2 --std-mm 4 --seed 0 --domain disk
```

This overwrites the `CRACK_NETWORK` sheet with 20 randomly placed and oriented straight
cracks. Lengths are drawn from a truncated-normal or log-normal distribution.

---

## 6. Inputs

Every case is one workbook, e.g. `vcem/input/disk_compression_data.xlsx`, read by
`input/excel_io.py:load_case` into a `CaseConfig` dataclass. Its sheets are:

| Sheet | Contents |
|---|---|
| `BEM` | disk radius `R_m`, total load `P_total_N`, `E_Pa`, `nu`, thickness `h_thickness_m`, `plane_strain`; boundary elements, quadrature order, platen arc half-angle, mesh grading (`boundary_mesh`, `boundary_concentration`, `boundary_taper_exponent`); `max_steps`, `BEM_correction_frequency` |
| `GEOMETRY` | boundary type and centre; evaluation grid source (`builtin_polar`, `builtin_rect`, `external_file`) and its resolution and padding |
| `CRACK_NETWORK` | initial network: vertices `id, x_m, y_m` and edges `v0, v1` |
| `CONFIGURATION_LOGIC` | switches: `engine` (`cpp`/`python`), `coupling_method`, `dry_run`, `output_dir_name`; growth policy (`simultaneous_tip_growth`, `simplify_each_step`, `intersection_detect_mode`); solver (`parametrization` = `cspline`, `crack_mode`, `junction_model`); augmented-coupling scaling; plot options |
| `CONFIGURATION_PARAM` | numbers: `crack_growth.Kc`, `L_limit_mm`, `f_disk_radius`, `rmax_frac`, `max_kink_deg`, simplifier tolerances; `solver_kwargs.n_crack_elements`, `soft_eta`; augmented `gauss_n`, `ridge_q`, `ridge_y`; plot levels and DPI |

Dotted keys (`crack_growth.X`, `solver_kwargs.X`, `augmented.X`, `plot.X`) map onto the
nested dataclasses, and a key that appears in both configuration sheets is an error.
Parameters are changed by editing the workbook, or by pointing `CASE_XLSX` at a
different one, so the workbook is always the complete record of a case.

A fresh workbook with the dataclass defaults is generated by
`python input/_build_template.py [case_name]` (this overwrites the target file).

---

## 7. Outputs and provenance

Each run writes a self-describing directory:

```
vcem/output/YYYYMMDD_HHMMSS_<output_dir_name>/
├── provenance_<id>.json / .md    # flattened CaseConfig, workbook path and SHA-256,
│                                 # engine, OpenMP threads, host, CPU, Python/NumPy, git rev
├── STEP_00 … STEP_NN/            # per-step stress contours, deformed network,
│                                 # network graph and connectivity (fragments coloured)
├── Sxx.npy, Syy.npy, Sxy.npy, *_polar.npy, xs/ys, rs/thetas   # BEM baseline field
├── brazilian_disk_*_polar.png, brazilian_disk_line_stresses.png
├── total_{sxx,syy,sxy}_{initial,step_NN}.png
├── step_history.json, step_metrics.npz        # per-step tips, SIFs, lengths, metrics
├── metrics_{Q,alpha,SIF_per_tip}_vs_step.png
└── videos/*.mp4                                # per-step animations
```

Videos are assembled at the end of each run by `vcem/tools/make_run_videos.py`, which
composites the network frames onto the baseline stress field; a failure there does not
fail the run. It can also be called on any existing run directory:

```bash
python vcem/tools/make_run_videos.py vcem/output/<run_dir> --fps 5
```

Because the workbook hash, the resolved configuration and the git revision are written
with every run, a figure can always be traced back to the code and inputs that produced
it.

---

## 8. Examples, tests and verification

**Driver notebooks** — `vcem/notebooks/SimulationsCpp.ipynb` (Excel-driven,
C++/Python switchable); `_archive/VCM_3_4/notebooks/Simulations.ipynb` (cases
`disk_compression_2`, `disk_compression_inclined`, `disk_energetics`,
`disk_experiments`) and `_archive/VCM_3_4/notebooks/ValidationRunner.ipynb`.

**Analytical validation** — results in `_archive/VCM_3_4/output/`:

| Case | Reference |
|---|---|
| Straight crack, SIF and COD convergence | $K_I = \sigma\sqrt{\pi c}$ and the infinite-plate opening profile |
| Mode II sliding | analytical CSD under remote shear |
| Inclined crack, mixed mode | $K_I$, $K_{II}$ versus inclination (Rooke & Cartwright) |
| Circular-arc crack, spline and 25-segment polyline | Cotterell & Rice (1980) |
| Branched crack | Lo (1978); digitized data in `docs/Data/` |
| Brazilian disk BEM (direct, iterative, uncoupled) | concentrated-load analytical solution, $\sigma_x(0) = 2P/\pi d h$ |
| BEM patch tests | rigid translation and uniform pressure |

Kinked and curved-crack propagation, Y- and star-junctions, trees, intersecting networks
and network simplification are kept as worked examples in the same folder.
`fracture_utils/Uvalidation/` provides the COD and SIF sweeps over solver knobs (panel
count, fit window) used to produce them.

**C++ parity tests** — `vcem/cpp/python/test_*_vs_python.py` check each ported
component against the Python reference: Kelvin kernels and quadrature, edge-dislocation
kernels, the crack operator, the boundary operators $\mathbf{M}_t$, $\mathbf{M}_u$,
$\mathbf{N}_{bc}$, the KKT solve, the BEM solve, segment intersection, and the
end-to-end engine dispatch. The BEM agrees with Python to about $10^{-8}$ relative.

**Regression tests** — small-panel merging, spatial-hash vertex merging against brute
force, and two SIF symmetry bugs (panel allocation on kinked polylines, tip orientation).

**Diagnostics** — `check_symmetry.py` and `check_bc_symmetry.py` (a symmetric BC on a
symmetric mesh must give a symmetric field to machine precision; residuals above
~$10^{-6}$ indicate a bug, not discretization noise); `diag_conditioning.py` (cond of
$\mathbf{K}$, $\mathbf{K}^T\mathbf{K}$ and the KKT matrix); `vcem/tools/diag_panel_mirror.py`
(mirror-symmetric network to locate the source of SIF asymmetry).

---

## 9. Computational aspects

**Where the time goes.** A growth step is one BEM solve (or correction), one KKT solve
over all crack panels, SIF extraction at every tip, and a topology update. The dense
operator assemblies scale as $O(N^2)$ in panel count and the dense KKT solve as
$O(N^3)$.

**C++ backend.** The hot paths are ported line-by-line from Python to C++/Eigen with
OpenMP over the natural outer loop, and selected per run with `engine='cpp'` through
`Usolver/cpp_dispatch.py`, which falls back to Python for missing features. Measured
speedups:

| Component | Speedup over Python |
|---|---|
| Brazilian-disk BEM solve and stress evaluation | ~10⁴ (≈30 min → 170 ms) |
| KKT assembly and solve | 16–27×, peaking at 500–800 panels, then limited by the dense $O(N^3)$ solve |
| Segment-intersection detection (inner kernel) | 117–173× |

Not everything was ported: close-vertex merging was made fast by replacing an $O(V^2)$
search with a spatial hash in Python (66–147×), and already-vectorized or graph-bound
code was left alone.

**Conditioning.** Tip-clustered panels can make $\mathbf{K}^T\mathbf{K}$ nearly singular,
so panels shorter than `min_panel_length_ratio` × $L$ are merged at discretization, and
the KKT block carries a small diagonal stabilization. The null-space formulation of
§2.4 avoids the conditioning blow-up of the saddle-point form in the Python path.

**Quadrature near boundaries.** Interior stress is near-singular within about one
element length of a boundary panel. Evaluation grids are padded away from the boundary,
or the stress quadrature order is raised where the grid must hug the boundary (the polar
grid). With a graded mesh the pad must match the largest element, not the smallest.

**Scaling path.** At more than ~5000 panels the dense KKT solve costs minutes and about
a gigabyte. The KKT backend is a runtime enum (`AutoLU`, `DenseLU`, `BDCSVD`, with
`SparseLU`, `GMRES` and `CudaDense` reserved), so sparse-iterative and GPU solvers can
be added without touching callers.

**Portability.** The backend builds with MSVC, GCC and Clang, and runs on the UCLA
Hoffman2 cluster. `-march=native` binaries are not portable across node types.

---

## 10. Documentation

| Location | Contents |
|---|---|
| `docs/Publications/` | the two VCEM papers in J. Mater. Sci.: Mater. Theory (2026): the formulation, and validation plus finite-domain network evolution |
| `docs/Fragmentation_Phase_Transition/` | manuscript on fragmentation of compressed disks as a phase transition; the self-learning simulation-campaign plan; bibliography |
| `docs/VCEM Elastoplasticity/` | plan for plastic deformation in VCEM for two tungsten fracture problems |
| `docs/VCM Code/` | VCM theory, architecture and code organization notebooks; `Variational_Fracture.pdf` |
| `docs/Data/` | Lo (1978) branched- and kinked-crack reference data |
| `docs/Literature/` | the reference library: BEM, dislocation-based fracture, curved, kinked and branched cracks, Brazilian disk testing, variational fracture, tungsten under heat loads |
| `vcem/cpp/README.md` | C++ backend layout, build and smoke test |
| `fracture_utils/Upropagation/PropagationReadMe.md`, `Ugenerator/NetworkGeneratorReadme.md` | module notes |

LaTeX sources sit next to their `.bib` files and compile from inside their own folder.

---

## 11. Publications and how to cite

If you use this code, please cite the two VCEM papers (PDFs in
[`docs/Publications/`](docs/Publications/)):

> N. M. Ghoniem, *A constrained variational method for crack networks with
> criterion-based evolution*, Journal of Materials Science: Materials Theory **10**
> (2026) 6. [doi:10.1186/s41313-026-00076-6](https://doi.org/10.1186/s41313-026-00076-6)
>
> N. M. Ghoniem, *Crack network evolution in finite domains*, Journal of Materials
> Science: Materials Theory **10** (2026) 9.
> [doi:10.1186/s41313-026-00077-5](https://doi.org/10.1186/s41313-026-00077-5)

```bibtex
@article{Ghoniem2026VCEMa,
  author  = {Ghoniem, N. M.},
  title   = {A constrained variational method for crack networks with
             criterion-based evolution},
  journal = {Journal of Materials Science: Materials Theory},
  volume  = {10},
  pages   = {6},
  year    = {2026},
  doi     = {10.1186/s41313-026-00076-6}
}

@article{Ghoniem2026VCEMb,
  author  = {Ghoniem, N. M.},
  title   = {Crack network evolution in finite domains},
  journal = {Journal of Materials Science: Materials Theory},
  volume  = {10},
  pages   = {9},
  year    = {2026},
  doi     = {10.1186/s41313-026-00077-5}
}
```

The first paper sets out the formulation: distributed Burgers-density cracks, the
constrained variational problem and its KKT structure, and graph-based half-crack
elements for tips, kinks, branches and junctions. The second validates VCEM against
analytical solutions and applies it to the Brazilian disk, where multi-crack networks
under force control fragment the disk through a connectivity phase transition.

Work in progress, in [`docs/Fragmentation_Phase_Transition/`](docs/Fragmentation_Phase_Transition/):

- N. M. Ghoniem, *Fragmentation of Compressed Disks as a Phase Transition: A
  Variational Crack Element Method Framework* (in preparation).

To cite the software itself, give the release you used, e.g. *VCEM 4.1.0*
(git tag `v4.1.0`, https://github.com/Ghoniem/Fracture).

Related work from the group:

- M. Alabdullah and N. M. Ghoniem, *Crack initiation and propagation in the diametral
  (Brazilian) disk compression test* (2023).
- G. Sheng and N. M. Ghoniem, *A discrete crack element method for modelling crack
  growth and interaction* (2024).
- N. M. Ghoniem, S.-H. Tong and L. Z. Sun, *Parametric dislocation dynamics: a
  thermodynamics-based approach to investigations of mesoscopic plastic deformation*,
  Phys. Rev. B **61** (2000) 913–927.
  [doi:10.1103/PhysRevB.61.913](https://doi.org/10.1103/PhysRevB.61.913)
- N. M. Ghoniem and X. Han, *Affine covariant–contravariant vector forms for the
  elastic field of parametric dislocations in isotropic crystals* (2005).

Validation references: B. Cotterell and J. R. Rice, Int. J. Fract. **16** (1980) 155–169
(curved cracks); K. K. Lo, J. Appl. Mech. **45** (1978) 797–802 (branched cracks).
Full reference lists are in the `.bib` files under `docs/`.

---

## 12. Repository conventions

- **One active module, archived predecessors.** Development happens in `vcem/`;
  `_archive/` (VCM 1.0 through 3.4) is read-only, so earlier results stay reproducible.
- **The workbook is the case.** Parameters live in the Excel workbook with a key, value
  and description; code reads them and does not hard-code them.
- **Outputs are immutable and stamped.** Runs never overwrite one another;
  `vcem/output/` and `build/` are gitignored.
- **Python first, then C++.** New physics is developed in Python, which is the
  reference implementation. A C++ port mirrors the Python arithmetic line by line and
  lands with a parity test in `cpp/python/` before it is wired into the dispatcher. The
  default engine stays usable without the extension.
- **Symmetry is a test.** Symmetric loading of a symmetric geometry must give a
  symmetric answer; asymmetry is debugged, not tolerated.

Issues and pull requests are welcome. Changes that touch kernels, constraints or the
KKT solve should come with the relevant parity or validation check and its result.

---

## 13. License and contact

Released under the [MIT License](LICENSE), © 2026 Nasr Ghoniem.

**Nasr M. Ghoniem** — Mechanical and Aerospace Engineering Department,
University of California, Los Angeles · ghoniem@ucla.edu
