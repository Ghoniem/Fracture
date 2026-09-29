"""provenance.py -- write a per-run provenance record for a VCEM run.

The provenance file captures everything a future reader needs to know
*exactly* how a run was produced, without re-reading the source Excel
workbook (which may have been edited since):

    1. All input parameters (every field of CaseConfig, recursively
       flattened from its nested dataclasses).
    2. Run stats: wall-clock time, BEM/solver engine, OpenMP threads,
       machine, hostname, OS, CPU count, Python version, NumPy version,
       git revision (if available).
    3. The path of the source workbook and its sha256.

Two files are written side by side:
    <run_dir>/provenance_<YYYYMMDD>_<HHMMSS>_<case>.json
    <run_dir>/provenance_<YYYYMMDD>_<HHMMSS>_<case>.md

The JSON is the machine-readable source of truth; the markdown is a
human-readable rendering of the same payload as a table.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


# ============================================================
# Public API
# ============================================================

def run_id(
    case_xlsx: str | Path,
    when: Optional[datetime] = None,
    *,
    file_name: Optional[str] = None,
) -> str:
    """Return the canonical run id: ``<YYYYMMDD>_<HHMMSS>_<name>``.

    ``name`` is ``file_name`` when given (typically ``cfg.output_dir_name``
    sourced from the Excel workbook), otherwise the workbook stem.
    """
    when = when or datetime.now()
    name = str(file_name).strip() if file_name else Path(case_xlsx).stem
    if not name:
        name = Path(case_xlsx).stem
    return f"{when.strftime('%Y%m%d_%H%M%S')}_{name}"


def write_provenance(
    *,
    cfg: Any,
    case_xlsx: str | Path,
    run_dir: str | Path,
    run_stats: Mapping[str, Any],
    started_at: datetime,
    finished_at: Optional[datetime] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Write the provenance JSON + Markdown into ``run_dir``.

    Returns a dict with the absolute paths to the two files written.
    """
    finished_at = finished_at or datetime.now()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id(case_xlsx, started_at)

    payload: Dict[str, Any] = {
        "run_id": rid,
        "case_xlsx": str(Path(case_xlsx).resolve()),
        "case_xlsx_sha256": _sha256(Path(case_xlsx)),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "wall_time_s": (finished_at - started_at).total_seconds(),
        "run_dir": str(run_dir.resolve()),
        "input_parameters": _flatten_dataclass("cfg", cfg),
        "run_stats": _augment_run_stats(dict(run_stats)),
        "environment": _environment_info(),
    }
    if extra:
        payload["extra"] = dict(extra)

    json_path = run_dir / f"provenance_{rid}.json"
    md_path = run_dir / f"provenance_{rid}.md"

    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    md_path.write_text(_render_markdown(payload))
    return {"json": json_path.resolve(), "md": md_path.resolve()}


# ============================================================
# Internals
# ============================================================

def _sha256(p: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _flatten_dataclass(prefix: str, obj: Any) -> Dict[str, Any]:
    """Recursively flatten a dataclass instance into {dotted_key: value}.

    Lists/tuples/dicts are stored whole (after JSON-safe coercion); ndarrays
    are converted to nested lists. Non-dataclass scalars are passed through.
    """
    out: Dict[str, Any] = {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            sub = getattr(obj, f.name, None)
            key = f"{prefix}.{f.name}" if prefix else f.name
            if dataclasses.is_dataclass(sub) and not isinstance(sub, type):
                out.update(_flatten_dataclass(key, sub))
            else:
                out[key] = _coerce(sub)
    else:
        out[prefix] = _coerce(obj)
    return out


def _coerce(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [_coerce(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _coerce(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted(_coerce(v) for v in x)
    if isinstance(x, Path):
        return str(x)
    return repr(x)


def _json_default(x: Any) -> Any:
    return _coerce(x)


def _augment_run_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce caller-supplied stats and ensure standard keys are present."""
    out = {k: _coerce(v) for k, v in stats.items()}
    out.setdefault("engine", "")
    out.setdefault("openmp_threads", "")
    return out


def _environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_count_logical": os.cpu_count(),
        "cwd": str(Path.cwd().resolve()),
        "numpy": _try_version("numpy"),
        "scipy": _try_version("scipy"),
        "pandas": _try_version("pandas"),
        "matplotlib": _try_version("matplotlib"),
        "bem_cpp": _try_version("bem_cpp"),
    }
    info["git"] = _git_info()
    return info


def _try_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return str(getattr(mod, "__version__", ""))
    except Exception:
        return ""


def _git_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {"rev": "", "branch": "", "dirty": ""}
    for key, args in (
        ("rev", ["rev-parse", "HEAD"]),
        ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            r = subprocess.run(
                ["git"] + args, capture_output=True, text=True,
                timeout=5, check=False,
            )
            if r.returncode == 0:
                out[key] = r.stdout.strip()
        except Exception:
            pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        if r.returncode == 0:
            out["dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    return out


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = []
    lines.append(f"# Provenance - {payload['run_id']}")
    lines.append("")
    lines.append(f"- Case workbook: `{payload['case_xlsx']}`")
    if payload.get("case_xlsx_sha256"):
        lines.append(f"- Workbook sha256: `{payload['case_xlsx_sha256']}`")
    lines.append(f"- Started:  {payload['started_at']}")
    lines.append(f"- Finished: {payload['finished_at']}")
    lines.append(f"- Wall time (s): {payload['wall_time_s']:.3f}")
    lines.append(f"- Run dir: `{payload['run_dir']}`")
    lines.append("")

    lines.append("## Run stats")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for k, v in sorted(payload["run_stats"].items()):
        lines.append(f"| `{k}` | {_fmt_cell(v)} |")
    lines.append("")

    lines.append("## Environment")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    env = payload["environment"]
    for k in sorted(env):
        v = env[k]
        if isinstance(v, dict):
            for kk in sorted(v):
                lines.append(f"| `{k}.{kk}` | {_fmt_cell(v[kk])} |")
        else:
            lines.append(f"| `{k}` | {_fmt_cell(v)} |")
    lines.append("")

    lines.append("## Input parameters")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    for k in sorted(payload["input_parameters"]):
        lines.append(f"| `{k}` | {_fmt_cell(payload['input_parameters'][k])} |")
    lines.append("")

    if "extra" in payload:
        lines.append("## Extra")
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        for k in sorted(payload["extra"]):
            lines.append(f"| `{k}` | {_fmt_cell(payload['extra'][k])} |")
        lines.append("")

    return "\n".join(lines)


def _fmt_cell(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        s = json.dumps(_coerce(v), default=_json_default)
        if len(s) > 200:
            s = s[:197] + "..."
        return f"`{s}`"
    if isinstance(v, dict):
        s = json.dumps(_coerce(v), default=_json_default)
        if len(s) > 200:
            s = s[:197] + "..."
        return f"`{s}`"
    if v is None:
        return "`null`"
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, float):
        return f"`{v!r}`"
    s = str(v)
    if "\n" in s:
        s = s.replace("\n", " ")
    if len(s) > 200:
        s = s[:197] + "..."
    if not s:
        return ""
    return f"`{s}`"
