"""Run a single ValidationRunner case in this process. Invoked by _run_all_cases.py.

Usage:  python _run_one_case.py <case_name>

Loads the ValidationRunner.ipynb case definitions (skipping the selector and
dispatcher), calls run_<case>(), and exits 0 on success, 1 on exception.
Stdout/stderr from the case run is suppressed (only the final OK/FAIL marker
is printed) so the orchestrator can stream concise progress.

To keep this a *smoke test* — verifying every case actually runs to
completion rather than verifying numerical results — heavy size knobs in the
inlined case body are clamped to small values via textual substitution. The
clamp only lowers values; cases that intentionally set a knob below the cap
(e.g. `n_junctions=0`) are left untouched.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

# Per-parameter cap. The body's `<param>=<int>` is rewritten to `<param>=cap`
# only when the literal value exceeds the cap.
SMOKE_CAPS = {
    "ne_half": 50,  # Np = 2*ne_half must exceed sum(min_panels_per_segment); propagation cases grow to 15+ segments.
    "n_elem": 30,
    "gauss_n": 4,
    "n_grid": 21,
    "nq_col": 3,
    "nq_stress": 4,
    "n_crack_paths": 2,
    "segments_per_path_mean": 3,
    "segments_per_path_std": 1,
    "n_junctions": 2,
    "n_tips": 4,
    "n_steps": 1,
    "max_cycles": 1,    # propagation outer-loop cap; keeps the network from growing past the smoke ne_half budget.
    "inner_max": 1,     # NetworkGrower max_inner_steps — without this, grower.run_one_cycle can grow networks beyond capped ne_half.
    "max_inner_steps": 1,
    "OUTER_CYCLES": 1,  # simulation notebooks (disk_energetics / disk_experiments) drive their own outer cycle counter.
    "n_inner_max_for_index": 1,
    "vertex_high": 10,  # trigger simplification early so each solve fits within capped ne_half.
    "cod_max_iter": 5,
    "n_iter": 5,
    "max_iter": 5,
    "iterations": 3,
}


_SWEEP_RANGE_RE = re.compile(
    r"(\w+_list\s*=\s*)list\(range\(\s*(\d+)[^)]*\)\)"
)
_SWEEP_ARANGE_RE = re.compile(
    r"(\w+_list\s*=\s*)np\.arange\(\s*(\d+)[^)]*\)"
)
_SWEEP_LINSPACE_RE = re.compile(
    r"(\w+_list\s*=\s*)np\.linspace\(\s*([0-9.eE+-]+)[^)]*\)"
)


def _apply_smoke_caps(src: str) -> str:
    """Lower heavy numeric kwargs in ``src`` to the SMOKE_CAPS values, then
    collapse explicit convergence sweeps (``*_list = list(range(...))`` /
    ``np.arange(...)``) to a single starting point so validation-style
    parameter studies don't dominate runtime in the smoke test."""
    for name, cap in SMOKE_CAPS.items():
        pattern = re.compile(rf"\b({re.escape(name)}\s*=\s*)(\d+)")

        def repl(m, _cap=cap):
            head, val = m.group(1), int(m.group(2))
            return f"{head}{_cap}" if val > _cap else m.group(0)

        src = pattern.sub(repl, src)

    src = _SWEEP_RANGE_RE.sub(r"\1[\2]", src)
    src = _SWEEP_ARANGE_RE.sub(r"\1np.array([\2])", src)
    src = _SWEEP_LINSPACE_RE.sub(r"\1np.array([\2])", src)
    return src

if len(sys.argv) not in (2, 3):
    print("usage: _run_one_case.py <case_name> [<notebook>]", file=sys.stderr)
    sys.exit(2)

case = sys.argv[1]
# Optional second arg selects which bundled notebook to load the case from
# (ValidationRunner.ipynb or Simulations.ipynb). Default keeps existing
# orchestrator usage backwards compatible.
notebook_name = sys.argv[2] if len(sys.argv) == 3 else "ValidationRunner.ipynb"

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

NB_DIR = Path(__file__).resolve().parent
REPO_DIR = NB_DIR.parent
os.chdir(NB_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

with open(NB_DIR / notebook_name, encoding="utf-8") as f:
    nb = json.load(f)

ns: dict = {"__name__": "__main__", "CASE": case}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "CASE = " in src and "options:" in src:
        continue
    if src.lstrip().startswith("CASES = {"):
        continue
    if src.lstrip().startswith(f"def run_{case}("):
        src = _apply_smoke_caps(src)
    exec(compile(src, f"<{notebook_name}>", "exec"), ns)

fn = ns.get(f"run_{case}")
if fn is None:
    print(f"MISSING: run_{case} not defined", flush=True)
    sys.exit(2)

t0 = time.time()
try:
    fn()
    elapsed = time.time() - t0
    print(f"OK {elapsed:.1f}s", flush=True)
    sys.exit(0)
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAIL {elapsed:.1f}s {type(e).__name__}: {e}", flush=True)
    print("--- traceback ---", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)
finally:
    plt.close("all")
