"""Orchestrator: smoke-test every case in Simulations.ipynb.

Mirrors _run_all_cases.py but targets the simulation notebook (every
``disk_*`` case). Each case runs in its own subprocess with the same
smoke-cap rewrites applied by _run_one_case.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
LOG_DIR = NB_DIR / "_sim_logs"
LOG_DIR.mkdir(exist_ok=True)
for _stale in LOG_DIR.glob("*.log"):
    _stale.unlink()

CASES = [
    "disk_compression_2",
    "disk_compression_inclined",
    "disk_energetics",
    "disk_experiments",
]

PER_CASE_TIMEOUT_S = 2400  # 40 minutes; disk simulations include heavier solves.

child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"
child_env["PYTHONUTF8"] = "1"

results: list[tuple[str, str, float]] = []
for case in CASES:
    log_path = LOG_DIR / f"{case}.log"
    cmd = [
        sys.executable,
        str(NB_DIR / "_run_one_case.py"),
        case,
        "Simulations.ipynb",
    ]
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=PER_CASE_TIMEOUT_S,
                cwd=NB_DIR,
                env=child_env,
            )
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = f"TIMEOUT(>{PER_CASE_TIMEOUT_S}s)"
    results.append((case, status, elapsed))
    print(f"[{status:18s} {elapsed:7.1f}s] {case}   (log: {log_path.name})", flush=True)

print()
print("=" * 72)
print("Summary")
print("=" * 72)
for case, status, elapsed in results:
    print(f"  {case:32s} {status:18s} {elapsed:7.1f}s")

n_ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\n{n_ok}/{len(results)} cases passed")
sys.exit(0 if n_ok == len(results) else 1)
