# VCEM tutorial 02 (crack kinking and growth) run — 20260929_180344_kinking_growth_1d7539c8

## (1) Input Data

_All input tables from the Excel workbook with parameter overrides applied._

### Material

- E: 2e+11
- Kc: 1e+06
- nu: 0.3
- plane: strain

### Geometry

- a: 0.005
- beta_deg: 45

### Loading

- sigma_xx: 0
- sigma_xy: 0
- sigma_yy: 1e+08

### Propagation

- L_limit_mm: 40
- f_fixed: 0.1
- kii_noise_ratio: 0.05
- max_kink_deg: 0
- n_steps: 6

### Solver

- crack_mode: half
- engine: python
- min_pts: 8
- n_crack_elements: 60
- parametrization: polyline
- rmax_frac: 0.25
- two_term: 1

### Verification

- tol_KII_ratio: 0.05
- tol_symmetry: 1e-06
- tol_theta_deg: 1

### Output

- dpi: 150
- label: kinking_growth
- stress_extent: 1.6
- stress_n_grid: 160

### Overrides (workbook -> run)

_(empty)_

## (2) User Selections

### Run configuration

- case_sha256: a5b6e8267fcddc5940198394d2acdccf398a80928d38023b1201e9297e4d10b8
- case_workbook: tutorials/02_crack_kinking_growth/input/kinking_growth.xlsx
- direction_law: MaximumHoopStressLaw
- domain: infinite plate
- engine_note: python engine requested
- engine_requested: python
- engine_used: python
- toughness: ConstantToughness
- tutorial: 02_crack_kinking_growth

## (3) Solver Configuration

### Solver settings

- propagation.f_fixed: 0.1
- propagation.kii_noise_ratio: 0.05
- propagation.max_kink_deg: 0
- propagation.simultaneous_tip_growth: True
- propagation.step_mode: fixed_step
- sif_fit.min_pts: 8
- sif_fit.rmax_frac: 0.25
- sif_fit.two_term: True
- solve.crack_mode: half
- solve.engine: python
- solve.n_crack_elements: 60
- solve.parametrization: polyline

## (4) Run Statistics

### Runtime and machine

- L_final_mm: 29.8598
- cpu_count: 24
- diagnostics: PASS
- git_branch: vcm_development
- git_dirty: True
- git_sha: 1d7539c8
- hostname: Nasr-Workstation
- n_steps: 6
- numpy: 2.4.6
- omp_num_threads: 24
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 85 Stepping 4, GenuineIntel
- python: 3.14.3
- ram_total_GB: 126.66
- run_status: completed
- scipy: 1.17.1
- stop_reason: n_steps reached
- vcem_version: 4.1.1
- wall_clock_growth_s: 43.9416

