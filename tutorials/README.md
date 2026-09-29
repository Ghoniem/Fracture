# VCEM tutorials

Two short, self-checking tutorials that introduce the Variational Crack Element Method on
simple fracture problems with exact solutions. Each one is organized exactly like a full VCEM
simulation:

```
input/<case>.xlsx  ──>  <tutorial>.ipynb  ──>  output/YYYYMMDD_HHMMSS_<label>_<git-hash>/
 (all parameters)       (orchestrator)          ├── provenance.md     inputs, selections, solver, machine
                                                ├── summary.csv       one row per run
                                                ├── diagnostics.txt   PASS/FAIL against exact results
                                                ├── *.csv             result tables
                                                └── plots/*.png
```

| # | Tutorial | Physics | Verified against | Runtime |
|---|---|---|---|---|
| 01 | [Inclined crack SIFs](01_inclined_crack_sif/01_inclined_crack_sif.ipynb) | static mixed-mode crack in an infinite plate | $K_I = \sigma_{nn}\sqrt{\pi a}$, $K_{II} = \sigma_{nt}\sqrt{\pi a}$, elliptical COD, panel convergence | ~3 min |
| 02 | [Kinking and growth](02_crack_kinking_growth/02_crack_kinking_growth.ipynb) | a 45° crack kinks and turns perpendicular to the load | MTS kink angle $-53.13°$, mode-I path ($K_{II} \to 0$), point symmetry | ~1 min |

Start with Tutorial 01. Tutorial 02 reuses its model-building steps and adds the propagation
loop that drives the full Brazilian-disk simulations in
[`VCEM_4_0/notebooks/SimulationsCpp.ipynb`](../VCEM_4_0/notebooks/SimulationsCpp.ipynb).

## Running

From the repository root, with the environment of the main [README](../README.md#4-installation)
(`numpy`, `scipy`, `matplotlib`, `networkx`, `openpyxl`, `jupyter`):

```bash
jupyter lab tutorials/01_inclined_crack_sif/01_inclined_crack_sif.ipynb
```

Run the notebook from its own folder (Jupyter does this by default). Both tutorials run with the
pure-Python engine; set `engine = cpp` in the workbook's `Solver` sheet to use the C++ backend
where it is built. If the extension cannot be imported, the run falls back to Python and the
provenance file records the fallback.

Every executed run writes a new stamped directory, so runs never overwrite one another. The
reference run committed with each tutorial shows what a correct run produces.

## Changing a case

- **One run only** — put the parameter's `Symbol` in `OVERRIDES` in the first cell,
  e.g. `OVERRIDES = {"beta_deg": 30}`. Overrides are listed in `provenance.md` as
  `workbook -> run`.
- **Permanently** — edit the workbook. Each row has a `Parameter` name, `Symbol`, `Value`,
  `Units` and a `Note`.

## Provenance

`provenance.md` follows the four-section layout used by RadCluster:

1. **Input Data** — every workbook table with overrides applied;
2. **User Selections** — tutorial, workbook path and SHA-256, requested and used engine;
3. **Solver Configuration** — solve, SIF-fit and propagation settings;
4. **Run Statistics** — wall-clock time, stop reason, git SHA/branch/dirty flag, machine,
   Python and NumPy versions.

## Files

| File | Purpose |
|---|---|
| `tutorial_utils.py` | shared helpers: workbook reader, overrides, run directory, provenance, CSV and diagnostics writers |
| `_build_tutorials.py` | generates both workbooks and notebooks from source; `--execute` runs them, `--only 01` builds one |
| `0N_*/input/*.xlsx` | case workbooks |
| `0N_*/*.ipynb` | the orchestrator notebooks |
| `0N_*/output/` | stamped run directories |

The notebooks and workbooks are generated: edit `_build_tutorials.py` and re-run
`python tutorials/_build_tutorials.py --execute` rather than editing them by hand.
