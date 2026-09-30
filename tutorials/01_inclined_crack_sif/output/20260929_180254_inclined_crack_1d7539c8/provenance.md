# VCEM tutorial 01 (inclined crack SIF) run — 20260929_180254_inclined_crack_1d7539c8

## (1) Input Data

_All input tables from the Excel workbook with parameter overrides applied._

### Material

- E: 2e+11
- nu: 0.3
- plane: strain

### Geometry

- a: 0.005
- beta_conv_deg: 30
- beta_list_deg: 0, 15, 30, 45, 60, 75, 90

### Loading

- sigma_xx: 0
- sigma_xy: 0
- sigma_yy: 1e+08

### Solver

- crack_mode: full
- engine: python
- min_pts: 8
- n_crack_elements: 80
- n_list: 20, 40, 80, 160
- parametrization: polyline
- rmax_frac: 0.25
- two_term: 1

### Verification

- tol_COD: 0.05
- tol_K: 0.05

### Output

- dpi: 150
- label: inclined_crack

### Overrides (workbook -> run)

_(empty)_

## (2) User Selections

### Run configuration

- beta_conv_deg: 30
- betas_deg: [0, 15, 30, 45, 60, 75, 90]
- case_sha256: 16353d2bc682f397f90c563f7c0d9704b80db4e22dec5ec04f294fbcb9555967
- case_workbook: tutorials/01_inclined_crack_sif/input/inclined_crack.xlsx
- domain: infinite plate
- engine_note: python engine requested
- engine_requested: python
- engine_used: python
- tutorial: 01_inclined_crack_sif

## (3) Solver Configuration

### Solver settings

- n_crack_elements: 80
- n_list: [20, 40, 80, 160]
- sif_fit.a_fit: 0.01
- sif_fit.method: DisplacementSIF.euclid_from_edge
- sif_fit.min_pts: 8
- sif_fit.rmax_frac: 0.25
- sif_fit.two_term: True
- solve.crack_mode: full
- solve.engine: python
- solve.parametrization: polyline

## (4) Run Statistics

### Runtime and machine

- cpu_count: 24
- diagnostics: PASS
- git_branch: vcm_development
- git_dirty: True
- git_sha: 1d7539c8
- hostname: Nasr-Workstation
- n_solves: 12
- numpy: 2.4.6
- omp_num_threads: 24
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 85 Stepping 4, GenuineIntel
- python: 3.14.3
- ram_total_GB: 126.66
- run_status: completed
- scipy: 1.17.1
- vcem_version: 4.1.1
- wall_clock_convergence_s: 64.8701
- wall_clock_sweep_s: 91.6634

