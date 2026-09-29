"""Conditioning diagnostic for the VCEM iterative coupling.

Wraps the KKT solver dispatcher to log cond(K), cond(K^T K), and cond(KKT)
on every solve, then runs the disk_compression_2 case for one outer cycle
with a small inner-step cap. Writes a CSV/log so we can see if the matrix
is blowing up around the inner step where the kernel got stuck.

Run from the vcem/ directory:
    python cpp/python/diag_conditioning.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make MKL DLLs visible if launched outside `conda activate`.
_env_bin = Path(r"C:/Users/Owner/anaconda3/envs/vcem_4_0/Library/bin")
if _env_bin.exists():
    os.environ["PATH"] = str(_env_bin) + os.pathsep + os.environ.get("PATH", "")

_here = Path(__file__).resolve()
_repo_root = _here.parents[2]                   # .../vcem
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_cpp_pkg = _here.parent                         # .../vcem/cpp/python
if str(_cpp_pkg) not in sys.path:
    sys.path.insert(0, str(_cpp_pkg))

import numpy as np

from input.excel_io import load_case
from dataclasses import replace as _dc_replace

import fracture_utils.Usolver.cpp_dispatch as _disp
from fracture_utils.Usolver.KKT import _compress_constraints

# --------------------------------------------------------------------
# Patch the KKT dispatcher with condition logging.
# --------------------------------------------------------------------
_LOG: list[dict] = []
_CALL_IDX = [0]

_orig_solve_eq = _disp.solve_kkt_lsq_eq
_orig_solve = _disp.solve_kkt_lsq


def _cond_or_inf(M: np.ndarray) -> float:
    if M.size == 0:
        return float("nan")
    try:
        return float(np.linalg.cond(M))
    except Exception:
        return float("inf")


def _log_conditioning(K, rhs, C, ridge, tag):
    K = np.asarray(K, float)
    C = np.asarray(C, float) if C is not None else np.zeros((0, K.shape[1]), float)
    t0 = time.perf_counter()
    cond_K = _cond_or_inf(K)
    t1 = time.perf_counter()
    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0])
    cond_KTK = _cond_or_inf(A)
    t2 = time.perf_counter()
    # Mimic KKT.py: add eps*I, then assemble [[A2, Cc.T],[Cc, 0]]
    eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-12
    A2 = A + eps * np.eye(A.shape[0])
    Cc, _ = _compress_constraints(C, d=np.zeros(C.shape[0], float), tol=1e-12)
    m = Cc.shape[0]
    KKT = np.block([[A2, Cc.T], [Cc, np.zeros((m, m), float)]])
    cond_KKT = _cond_or_inf(KKT)
    t3 = time.perf_counter()

    sv_min = sv_max = float("nan")
    try:
        s = np.linalg.svd(K, compute_uv=False)
        if s.size:
            sv_max, sv_min = float(s[0]), float(s[-1])
    except Exception:
        pass

    rec = {
        "call": _CALL_IDX[0],
        "tag": tag,
        "K_rows": int(K.shape[0]),
        "K_cols": int(K.shape[1]),
        "C_rows": int(C.shape[0]),
        "Cc_rows": int(Cc.shape[0]),
        "ridge": float(ridge),
        "cond_K": cond_K,
        "cond_KTK": cond_KTK,
        "cond_KKT": cond_KKT,
        "sv_max_K": sv_max,
        "sv_min_K": sv_min,
        "rhs_norm": float(np.linalg.norm(rhs)),
        "rhs_max": float(np.max(np.abs(rhs))) if np.asarray(rhs).size else 0.0,
        "t_cond_K_s": t1 - t0,
        "t_cond_KTK_s": t2 - t1,
        "t_cond_KKT_s": t3 - t2,
    }
    _LOG.append(rec)
    print(
        f"[cond #{rec['call']:3d} {tag:>14s}] "
        f"K={rec['K_rows']:4d}x{rec['K_cols']:4d} "
        f"C={rec['C_rows']:3d}(eff {rec['Cc_rows']:3d}) "
        f"cond(K)={cond_K:.2e}  "
        f"cond(K^TK)={cond_KTK:.2e}  "
        f"cond(KKT)={cond_KKT:.2e}  "
        f"sv(K)=[{sv_min:.2e},{sv_max:.2e}]",
        flush=True,
    )
    _CALL_IDX[0] += 1


def _patched_solve_eq(K, rhs, C, d=None, ridge=0.0, ridge_diag=None, engine="python"):
    _log_conditioning(K, rhs, C, ridge, tag="kkt_lsq_eq")
    return _orig_solve_eq(K, rhs, C, d=d, ridge=ridge,
                          ridge_diag=ridge_diag, engine=engine)


def _patched_solve(K, rhs, C, ridge=0.0, engine="python"):
    _log_conditioning(K, rhs, C, ridge, tag="kkt_lsq")
    return _orig_solve(K, rhs, C, ridge=ridge, engine=engine)


_disp.solve_kkt_lsq_eq = _patched_solve_eq
_disp.solve_kkt_lsq = _patched_solve

# Also patch parametrization.solve_kkt_lsq / _eq if they were star-imported.
import fracture_utils.Usolver.parametrization as _para_mod
_para_mod.solve_kkt_lsq = _patched_solve         # type: ignore[assignment]
_para_mod.solve_kkt_lsq_eq = _patched_solve_eq   # type: ignore[assignment]

# --------------------------------------------------------------------
# Engine wiring, mirroring the notebook.
# Allow a CLI override so we can point at the HEAD-committed workbook
# while the working-tree copy is mid-edit.
# --------------------------------------------------------------------
if len(sys.argv) > 1:
    xlsx = Path(sys.argv[1]).resolve()
else:
    xlsx = _repo_root / "input" / "disk_compression_2.xlsx"
print(f"[case] {xlsx}")
cfg = load_case(xlsx)
ENGINE = cfg.engine

import fracture_utils.Usolver.parametrization as _para
import fracture_utils.Ugenerator.generator as _gen
import fracture_utils.Ubem.brazilian_disk_bem as _bdb
import fracture_utils.Ubem.disk_iterative_coupling as _dic

_orig_solve_dce = _para.DCENetworkStaticV4.solve
def _solve_with_engine(self, *args, **kwargs):
    kwargs.setdefault("engine", ENGINE)
    return _orig_solve_dce(self, *args, **kwargs)
_para.DCENetworkStaticV4.solve = _solve_with_engine

_orig_det = _gen.CrackNetworkGenerator.detect_and_split_intersections
def _det_with_engine(self, *args, **kwargs):
    kwargs.setdefault("engine", ENGINE)
    return _orig_det(self, *args, **kwargs)
_gen.CrackNetworkGenerator.detect_and_split_intersections = _det_with_engine

_orig_ensure  = _bdb.ensure_bem_field
_orig_compute = _bdb.compute_bem_brazilian_disk_field
def _ensure_with_engine(*args, **kwargs):
    kwargs.setdefault("engine", ENGINE)
    return _orig_ensure(*args, **kwargs)
def _compute_with_engine(*args, **kwargs):
    kwargs.setdefault("engine", ENGINE)
    return _orig_compute(*args, **kwargs)
_bdb.ensure_bem_field = _ensure_with_engine
_bdb.compute_bem_brazilian_disk_field = _compute_with_engine

_orig_sbwet = _dic.solve_bem_with_extra_boundary_tractions
def _sbwet_with_engine(*args, **kwargs):
    kwargs.setdefault("engine", ENGINE)
    return _orig_sbwet(*args, **kwargs)
_dic.solve_bem_with_extra_boundary_tractions = _sbwet_with_engine

# --------------------------------------------------------------------
# Run one outer cycle with a small inner-cycle cap.
# --------------------------------------------------------------------
from fracture_utils.Ubem.brazilian_disk_bem import ensure_bem_field
from fracture_utils.Ubem.disk_network_propagation import run_network_growth_uncoupled
from fracture_utils.Ubem.coupling_blocks_v2 import compose_solver_kwargs_v2

from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
bem_dir = _repo_root / "output" / f"{ts}_diag_conditioning"
bem_dir.mkdir(parents=True, exist_ok=True)

ensure_bem_field(cfg.disk, bem_dir, recompute=True, show=False, save_contours=False)

one_way_kwargs = compose_solver_kwargs_v2(
    "one_way",
    base_solver_kwargs={
        "n_crack_elements": cfg.n_crack_elements,
        "parametrization": cfg.parametrization,
    },
)

# Cap inner cycles -- we just need to see the trend.
INNER_CAP = 8
grow = _dc_replace(
    cfg.crack_growth,
    solver_kwargs=one_way_kwargs,
    max_inner_cycles_per_outer=INNER_CAP,
    enable_inner_cycle_plot_save=False,
)

out_growth = bem_dir / "outer_01_growth"
out_growth.mkdir(parents=True, exist_ok=True)

print(f"\nRunning growth for {INNER_CAP} inner steps (engine={ENGINE})...\n")
res = run_network_growth_uncoupled(
    bem_dir=bem_dir,
    out_dir=out_growth,
    vertices=cfg.vertices.copy(),
    connectivity=cfg.connectivity.copy(),
    params=grow,
    plot_hook=None,
    original_initial_vertex_ids=set(int(v) for v in cfg.vertices[:, 0]),
    plot_params=cfg.plot,
)

# --------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------
import csv
csv_path = bem_dir / "kkt_conditioning.csv"
if _LOG:
    keys = list(_LOG[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in _LOG:
            w.writerow(r)
    print(f"\nWrote per-call conditioning log to: {csv_path}")

print("\n========== SUMMARY ==========")
print(f"  KKT solves logged: {len(_LOG)}")
if _LOG:
    cK   = np.array([r["cond_K"]   for r in _LOG])
    cKTK = np.array([r["cond_KTK"] for r in _LOG])
    cKKT = np.array([r["cond_KKT"] for r in _LOG])
    print(f"  cond(K)    : min={np.nanmin(cK):.3e}  max={np.nanmax(cK):.3e}  median={np.nanmedian(cK):.3e}")
    print(f"  cond(K^TK) : min={np.nanmin(cKTK):.3e}  max={np.nanmax(cKTK):.3e}  median={np.nanmedian(cKTK):.3e}")
    print(f"  cond(KKT)  : min={np.nanmin(cKKT):.3e}  max={np.nanmax(cKKT):.3e}  median={np.nanmedian(cKKT):.3e}")
    worst = int(np.nanargmax(cKKT))
    r = _LOG[worst]
    print(f"  Worst KKT  : call #{r['call']} {r['tag']} "
          f"K={r['K_rows']}x{r['K_cols']} cond(K)={r['cond_K']:.2e} "
          f"cond(KKT)={r['cond_KKT']:.2e}")
print("=============================")
