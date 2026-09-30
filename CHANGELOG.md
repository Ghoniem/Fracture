# Changelog

All notable changes to VCEM are recorded here. Versions follow
[semantic versioning](https://semver.org). Each release is a git tag `vX.Y.Z` and a
GitHub release titled `VCEM X.Y.Z`.

## [Unreleased]

## [4.1.1] — 2026-09-29

A patch release: build, packaging and tooling fixes. Solver results are unchanged.

### Changed

- The C++ backend is built against the repository's `venv_fracture` (Python 3.14) instead
  of the retired conda env `vcem_4_0`; `vcem/cpp/README.md` documents the build and the
  agreement-test check.
- `requirements.txt` now includes `openpyxl`, `imageio`, `imageio-ffmpeg` and `pybind11`,
  so a single install covers workbooks, run videos and the C++ build.

### Fixed

- `vcem/cpp/python/run_disk_compression_2.py` loads `disk_compression_data.xlsx` (the
  workbook renamed in 4.1.0) and accepts a workbook path as its argument.
- The 4.1.0 note that "the bundled `bem_cpp` binary is built for CPython 3.10" was
  wrong: the extension is not distributed and must be built for the interpreter in use.

### Removed

- Committed macOS `.DS_Store` files (repository root and `_archive/`); `.gitignore`
  already excludes them.

## [4.1.0] — 2026-09-29

The first tagged release. It covers the VCEM 4.0 development line (the C++ port of
VCM 3.4, from 2026-05-25) together with the restructuring into a released package.

### Added

- **C++ backend** (`vcem/cpp/`, Eigen + OpenMP, exposed as the pybind11 module `bem_cpp`),
  selected per run with `engine='cpp'`, with automatic fallback to Python:
  - boundary-element solver with Kelvin kernels and mixed BCs, about 10⁴× faster than Python
    on the Brazilian-disk BEM step;
  - edge-dislocation kernels, crack operator, boundary operators ($\mathbf{M}_t$, $\mathbf{M}_u$,
    $\mathbf{N}_{bc}$) and the KKT solve, 16–27× faster;
  - segment-intersection detection for topology updates, 117–173× faster on the inner kernel;
  - a C++-vs-Python agreement test for every ported component, plus benchmarks (`vcem/cpp/python/`).
- **Excel-driven workflow**: one case workbook (`vcem/input/`) read by `load_case`; the
  driver notebook `vcem/notebooks/SimulationsCpp.ipynb`; per-run provenance (`provenance_<id>.json/.md`).
- **Brazilian-disk simulation features**:
  - a boundary mesh graded towards the platens, and a polar evaluation grid that follows the disk;
  - random and log-normal flaw populations (`seed_crack_network.py`);
  - sub-critical isolated cracks removed from the KKT solve;
  - termination when every tip's growth force is zero, with tips that reach the boundary arrested;
  - graceful interrupt, a runtime limit, and automatic finalization of interrupted runs;
  - per-step contour frames and end-of-run videos (`vcem/tools/make_run_videos.py`);
  - network connectivity plots and the order parameters α, P∞ and Q per step;
  - a KKT conditioning diagnostic and a symmetry diagnostic (`vcem/tools/diag_panel_mirror.py`).
- **Tutorials** (`tutorials/`): two self-checking notebooks, an inclined crack's SIFs and the
  kinking and growth of an inclined crack. Each has an input workbook, stamped output and
  RadCluster-style `provenance.md`.
- **Documentation**: top-level `README.md`, this changelog, the MIT `LICENSE`, and the two
  published VCEM papers in `docs/Publications/`, now cited from the LaTeX documents.

### Changed

- The code folder `VCEM_4_0/` is renamed `vcem/`. The version now lives in tags, not folder
  names.
- Package versions are aligned with the release: `fracture_utils.__version__` and
  `bem_cpp.__version__` are `4.1.0`. The compiled extension reports the new version once it
  is rebuilt.
- `tools/` moved to `vcem/tools/`.
- `VCM_3_4/` moved to `_archive/VCM_3_4/`.
- Solver keyword `ne_half` renamed `n_crack_elements`. Workbook `Kc_demo` renamed `Kc`.
  `disk_compression_2.xlsx` renamed `disk_compression_data.xlsx`.

### Fixed

- Crack-path zigzag: a K_II noise threshold and a kink clamp in the MTS law.
- Boundary-condition selectors now carry a 10⁻⁹° tolerance at angular endpoints, which removes
  a spurious asymmetry in the disk stress field.
- Tip-clustered panels shorter than 10⁻³·L are merged, which prevents KKT conditioning
  blow-up. Short polylines get a length-aware panel count.
- Close-vertex merging uses a spatial hash, O(V) instead of O(V²).

### Known issues

- On kinked polyline paths the computed K_II drifts by 1–2 % of K_I per added segment; the
  K_II noise threshold then triggers small corrective kinks. The drift is larger, and loses
  tip symmetry, in the dipolar (`crack_mode="full"`) mode. See Tutorial 02.
- The bundled `bem_cpp` binary is built for CPython 3.10 on Windows. Other interpreters need
  a rebuild (`vcem/cpp/README.md`), or run with the Python engine.

## Earlier versions (untagged)

| Version | Date | Highlights |
|---|---|---|
| VCEM 4.0 | 2026-05-25 | seeded from VCM 3.4 for the C++ port; development line folded into 4.1.0 |
| VCM 3.4 | 2026-03-03 | VCM 3.3 frozen for displacement-controlled network runs |
| VCM 3.3 | 2026-02-26 | fracture-network energetics; topology operations (trim, merge, boundary detection) |
| VCM 3.2 | 2026-02-18 | Brazilian-disk simulations and validation |
| VCM 3.1 | 2026-02-13 | BEM coupling combined with crack evolution |
| VCM 3.0 | 2026-02-12 | outer boundary conditions through a 2D BEM |
| VCM 2.0–2.2 | 2026-02 | network evolution in uniform and non-uniform fields; kinked, branched and curved-crack validation |
| VCM 1.0–1.3 | 2026-01-20 | first Variational Crack Method: arbitrary polyline networks with junctions |

The VCM versions are preserved unchanged in `_archive/`.

[Unreleased]: https://github.com/Ghoniem/Fracture/compare/v4.1.1...HEAD
[4.1.1]: https://github.com/Ghoniem/Fracture/releases/tag/v4.1.1
[4.1.0]: https://github.com/Ghoniem/Fracture/releases/tag/v4.1.0
