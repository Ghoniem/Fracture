"""Assembles ValidationRunner.ipynb and Simulations.ipynb from the existing per-case notebooks.

This script is the source-of-truth for how the two runner notebooks are
built. It is committed alongside them so the bundling is reproducible: if
one of the per-case notebooks evolves, re-run this script to regenerate
both runners.

The case split is intentional:

    Simulations.ipynb  -> every "disk_*" case (production simulations).
    Validation.ipynb   -> everything else (BEM checks, validation sweeps,
                          crack-net/graph tools, propagation, diagnostics).

Both notebooks share the same skeleton: a header, a CASE selector with a
DRY_RUN guard, a common preamble, one ``def run_<case>():`` per case, and a
dispatcher at the bottom. State is isolated per case because each body
runs inside its own function scope.

Notebook-level markdown cells from the originals become a docstring at the
top of the wrapper function. Hardcoded data arrays stay where they were.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


NB_DIR = Path(__file__).resolve().parent

CASE_TO_FILE = {
    "bem_displacement": "BEM_displacement.ipynb",
    "crack_net": "crack_net.ipynb",
    "direct_method_verification": "direct_method_verification.ipynb",
    "disk_compression_2": "disk_compression_2.ipynb",
    "disk_compression_inclined": "disk_compression_inclined.ipynb",
    "disk_energetics": "disk_energetics.ipynb",
    "disk_experiments": "disk_experiments.ipynb",
    "function_diagnostics": "function_diagnostics.ipynb",
    "graph_net": "graph_net.ipynb",
    "propagation": "propagation.ipynb",
    "validation": "validation.ipynb",
}

SIMULATION_CASES = [c for c in CASE_TO_FILE if c.startswith("disk_")]
VALIDATION_CASES = [c for c in CASE_TO_FILE if not c.startswith("disk_")]


# -----------------------------
# Cell builders
# -----------------------------
def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join((prefix + line) if line else line for line in text.split("\n"))


def _markdown_from_source(src) -> str:
    if isinstance(src, list):
        return "".join(src)
    return str(src or "")


def _strip_illegal_function_body_imports(src: str) -> str:
    """Strip imports that aren't allowed inside a function body.

    ``from X import *`` is illegal inside a function body, so we hoist
    ``from preamble import *`` to the common preamble of the runner
    notebook and drop the per-case redundant copies. The
    ``graph_utils.crack_network_generator`` import is a stale reference (the
    module does not exist anywhere in the repo) and is dropped silently.

    ``from __future__ import ...`` must be at the top of a file; since we are
    wrapping every notebook in a single function inside one file, these
    statements are stripped as well (Python 3.10+ already has annotations as
    strings on demand and the other __future__ flags used here are no-ops on
    current Python).
    """
    out_lines = []
    for line in src.split("\n"):
        stripped = line.strip()
        if stripped.startswith("from preamble import *"):
            continue
        if stripped.startswith("from graph_utils.crack_network_generator import *"):
            continue
        if stripped.startswith("from __future__ import"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _wrap_case(case: str, nb_path: Path) -> dict:
    """Read a per-case notebook and emit one code cell that defines ``run_<case>()``."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    md_chunks = []
    code_chunks = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            md_chunks.append(_markdown_from_source(c.get("source", "")))
        elif c.get("cell_type") == "code":
            code_chunks.append(_markdown_from_source(c.get("source", "")))

    docstring = "\n\n".join(s.strip() for s in md_chunks if s.strip()) or (
        f"Case extracted from {nb_path.name}."
    )
    docstring = docstring.replace('"""', "'''")

    body = "\n\n# ---- next cell ----\n\n".join(code_chunks)
    body = body.replace("\r\n", "\n")
    body = _strip_illegal_function_body_imports(body)
    body_indented = _indent(body, "    ")

    src = (
        f'def run_{case}():\n'
        f'    """{docstring}"""\n'
        f'{body_indented}\n'
        f'    return locals()\n'
    )

    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# -----------------------------
# Notebook assembly
# -----------------------------
def build_notebook(cases: List[str], title: str, blurb: str) -> dict:
    options_inline = " | ".join(cases)

    header_md = (
        f"# {title} — VCM 3.4\n"
        "\n"
        f"{blurb}\n"
        "\n"
        "Set `CASE` in the next cell and run the notebook top to bottom; the\n"
        "dispatcher at the bottom calls the matching `run_<case>()` function.\n"
        "\n"
        "Available cases (matching the per-case notebooks in this directory):\n"
        + "".join(f"- `{c}`\n" for c in cases) +
        "\n"
        "Each `run_<case>()` is a faithful inlining of the corresponding notebook's\n"
        "code cells. Markdown cells from the originals become the function docstring.\n"
        "If a case notebook changes, regenerate this file with `_build_runners.py`.\n"
    )

    case_selector = (
        "# Pick which workflow to run. Must be one of the keys in CASES at the bottom.\n"
        f'CASE = "{cases[0]}"  # options: {options_inline}\n'
        "\n"
        "# Safety guard: keep DRY_RUN = True the first time you open this notebook so\n"
        "# Run-All only defines the case functions without executing them. Set to\n"
        "# False to actually invoke run_<CASE>() at the bottom.\n"
        "DRY_RUN = True\n"
    )

    common_preamble = (
        "from __future__ import annotations  # hoisted from per-case notebooks\n"
        "\n"
        "from pathlib import Path\n"
        "import os, sys\n"
        "import numpy as np  # noqa: F401  (every case imports it)\n"
        "import matplotlib.pyplot as plt  # noqa: F401  (most cases need it)\n"
        "\n"
        "nb_dir = Path(os.getcwd()).resolve()\n"
        "repo_root = nb_dir.parent\n"
        "if str(repo_root) not in sys.path:\n"
        "    sys.path.insert(0, str(repo_root))\n"
        "\n"
        "# Several legacy notebooks did `from preamble import *` at the top of every\n"
        "# cell, which is illegal inside a function body. Hoist that wildcard import\n"
        "# here at module scope so every case function inherits the same namespace,\n"
        "# and the per-case bodies (stripped of their own `from preamble import *`\n"
        "# lines by _build_runners.py) continue to work unchanged.\n"
        "from preamble import *  # noqa: F401,F403\n"
        "\n"
        "print(f'Running case: {CASE!r}')\n"
        "print(f'Repo root:    {repo_root}')\n"
    )

    cells: list[dict] = [
        _md_cell(header_md),
        _md_cell("## 1. Case selector\n"),
        _code_cell(case_selector),
        _md_cell("## 2. Common preamble (imports + repo path)\n"),
        _code_cell(common_preamble),
        _md_cell("## 3. Per-case workflows\n\n"
                 "Each cell below defines `run_<case>()`. Only the function matching `CASE`\n"
                 "is invoked by the dispatcher; the others are defined but not executed.\n"),
    ]

    for case in cases:
        nb_path = NB_DIR / CASE_TO_FILE[case]
        if not nb_path.exists():
            continue
        cells.append(_md_cell(f"### Case `{case}` (from `{nb_path.name}`)\n"))
        cells.append(_wrap_case(case, nb_path))

    dispatcher = (
        "CASES = {\n"
        + "".join(f'    "{c}": run_{c},\n' for c in cases) +
        "}\n"
        "\n"
        'if CASE not in CASES:\n'
        '    raise ValueError(f"Unknown CASE={CASE!r}. Expected one of {list(CASES)}.")\n'
        "\n"
        "if DRY_RUN:\n"
        "    print(f'DRY RUN — case functions defined but not executed.')\n"
        "    print(f'Set DRY_RUN = False (in the selector cell) and re-run to invoke run_{CASE}().')\n"
        "    result = None\n"
        "else:\n"
        "    result = CASES[CASE]()\n"
        "    if isinstance(result, dict):\n"
        "        print(f'Finished case {CASE!r}; returned-locals keys: {list(result)[:8]}...')\n"
        "    else:\n"
        "        print(f'Finished case {CASE!r}.')\n"
    )
    cells.append(_md_cell("## 4. Dispatch\n"))
    cells.append(_code_cell(dispatcher))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    targets = [
        (
            "Simulations.ipynb",
            "Simulations",
            "Production simulation runs: Brazilian-disk + crack-network coupling, "
            "every `disk_*` case from this directory.",
            SIMULATION_CASES,
        ),
        (
            # Named ValidationRunner.ipynb to avoid colliding with the source
            # validation.ipynb on case-insensitive filesystems (Windows, default macOS).
            "ValidationRunner.ipynb",
            "Validation Runner",
            "Validation, verification, and exploration cases: BEM checks, COD/SIF "
            "convergence sweeps, crack-network and graph tools, propagation, and "
            "function diagnostics.",
            VALIDATION_CASES,
        ),
    ]
    for fname, title, blurb, cases in targets:
        nb = build_notebook(cases, title, blurb)
        out = NB_DIR / fname
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(
            f"Wrote {out.name:24s}  ({len(cases)} cases, "
            f"{len(nb['cells'])} cells, {out.stat().st_size / 1024:.1f} KB)"
        )


if __name__ == "__main__":
    main()
