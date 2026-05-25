"""Orchestrator: run every ValidationRunner case in its own subprocess.

Each case is launched as `python _run_one_case.py <case>` with a per-case
timeout. Stdout/stderr from the case (success or failure) is captured into
``_case_logs/<case>.log`` so the orchestrator chat stream stays concise.

The default Python is the same interpreter that runs this script
(``sys.executable``) so the orchestrator and the per-case processes share an
environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
LOG_DIR = NB_DIR / "_case_logs"
LOG_DIR.mkdir(exist_ok=True)
for _stale in LOG_DIR.glob("*.log"):
    _stale.unlink()

CASES = [
    "bem_displacement",
    "crack_net",
    "direct_method_verification",
    "function_diagnostics",
    "graph_net",
    "propagation",
    "validation",
]

PER_CASE_TIMEOUT_S = 1200  # 20 minutes — propagation/validation iterate many subcases even after smoke caps.

child_env = os.environ.copy()
# Force UTF-8 stdout in the per-case subprocess so emoji-using diagnostics
# (e.g. function_locator's "📋", visualizers' "✓") don't crash under the
# Windows default cp1252 stdout encoding.
child_env["PYTHONIOENCODING"] = "utf-8"
child_env["PYTHONUTF8"] = "1"

results: list[tuple[str, str, float]] = []
for case in CASES:
    log_path = LOG_DIR / f"{case}.log"
    cmd = [sys.executable, str(NB_DIR / "_run_one_case.py"), case]
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
    print(f"[{status:18s} {elapsed:6.1f}s] {case}   (log: {log_path.name})", flush=True)

print()
print("=" * 72)
print("Summary")
print("=" * 72)
for case, status, elapsed in results:
    print(f"  {case:32s} {status:18s} {elapsed:7.1f}s")

n_ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\n{n_ok}/{len(results)} cases passed")
sys.exit(0 if n_ok == len(results) else 1)
