"""Run a Brazilian-disk case end-to-end (notebook equivalent).

    python vcem/cpp/python/run_disk_compression_2.py [case.xlsx]

The default case is vcem/input/disk_compression_data.xlsx.

Mirrors the SimulationsCpp.ipynb engine wiring and cell-6 run logic so the
fix in cpp_dispatch.py can be exercised from the CLI rather than restarting
the Jupyter kernel. Outputs land in vcem/output/<ts>_<output_dir_name>/.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

_env_bin = Path(r"C:/Users/Owner/anaconda3/envs/vcem_4_0/Library/bin")
if _env_bin.exists():
    os.environ["PATH"] = str(_env_bin) + os.pathsep + os.environ.get("PATH", "")

_here = Path(__file__).resolve()
_repo_root = _here.parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_cpp_pkg = _here.parent
if str(_cpp_pkg) not in sys.path:
    sys.path.insert(0, str(_cpp_pkg))

import matplotlib
matplotlib.use("Agg")          # CLI: never open a window or block on show()

import numpy as np
from dataclasses import replace as _dc_replace

# Load case (optional argument: path to a case workbook)
from input.excel_io import load_case
xlsx = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _repo_root / "input" / "disk_compression_data.xlsx"
cfg = load_case(xlsx)
ENGINE = cfg.engine

import bem_cpp  # noqa
print(f"bem_cpp v{bem_cpp.__version__}  threads={bem_cpp.openmp_max_threads()}")
print(f"engine={ENGINE}  coupling={cfg.coupling_method}  outer_cycles={cfg.outer_cycles}")
print(f"network={cfg.vertices.shape[0]} verts, {cfg.connectivity.shape[0]} edges")

# Engine monkey-patches (notebook cell 4 equivalent)
import fracture_utils.Usolver.parametrization as _para
import fracture_utils.Ugenerator.generator as _gen
import fracture_utils.Ubem.brazilian_disk_bem as _bdb
import fracture_utils.Ubem.disk_iterative_coupling as _dic

_orig_solve_dce = _para.DCENetworkStaticV4.solve
def _solve_with_engine(self, *args, **kwargs):
    kwargs.setdefault("engine", ENGINE);  return _orig_solve_dce(self, *args, **kwargs)
_para.DCENetworkStaticV4.solve = _solve_with_engine

_orig_det = _gen.CrackNetworkGenerator.detect_and_split_intersections
def _det_with_engine(self, *args, **kwargs):
    kwargs.setdefault("engine", ENGINE);  return _orig_det(self, *args, **kwargs)
_gen.CrackNetworkGenerator.detect_and_split_intersections = _det_with_engine

_orig_ensure  = _bdb.ensure_bem_field
_orig_compute = _bdb.compute_bem_brazilian_disk_field
def _ensure_with_engine(*a, **k):
    k.setdefault("engine", ENGINE);  return _orig_ensure(*a, **k)
def _compute_with_engine(*a, **k):
    k.setdefault("engine", ENGINE);  return _orig_compute(*a, **k)
_bdb.ensure_bem_field = _ensure_with_engine
_bdb.compute_bem_brazilian_disk_field = _compute_with_engine

_orig_sbwet = _dic.solve_bem_with_extra_boundary_tractions
def _sbwet_with_engine(*a, **k):
    k.setdefault("engine", ENGINE);  return _orig_sbwet(*a, **k)
_dic.solve_bem_with_extra_boundary_tractions = _sbwet_with_engine

# Imports for the run function (notebook cell 6)
from fracture_utils.Ubem.brazilian_disk_bem import (
    ensure_bem_field, compute_bem_brazilian_disk_field,
)
from fracture_utils.Ubem.disk_network_propagation import (
    run_network_growth_uncoupled, run_growth_steps,
)
from input.excel_io import save_supplementary_grid  # noqa
from fracture_utils.Ubem.disk_crack_plotting import (
    make_standard_plot_hook, plot_total_field, plot_step_metrics,
)
from fracture_utils.Ubem.coupling_blocks_v2 import (
    BEMBlockV2, compose_solver_kwargs_v2,
)
from fracture_utils.Ubem.disk_iterative_coupling import (
    compute_crack_boundary_tractions_direct,
    solve_bem_with_extra_boundary_tractions,
)
from fracture_utils.Ubem.boundary_conditions import build_boundary
from fracture_utils.Ugenerator.spanning_cluster import find_spanning_clusters
from input.provenance import write_provenance

def _network_to_arrays(net):
    V = np.array([[int(v.id), float(v.x), float(v.y)] for v in net.vertices], float)
    if len(net.edges):
        C = np.array([[int(e.v0), int(e.v1)] for e in net.edges], int)
    else:
        C = np.zeros((0, 2), int)
    return V, C


def _pick_crack_mode(requested: str, connectivity: np.ndarray) -> str:
    """Resolve cfg.crack_mode='auto' against the current connectivity.

    'auto' returns 'full' only when the network is a single chain
    component with no deg>=3 vertex; otherwise 'half'. Explicit
    'full'/'half' is returned unchanged.

    Why this rule:
    - build_polylines_full (the crack_mode='full' polyline builder) walks
      each connected component as a single chain via order_path_from_component
      and silently drops edges whenever the component branches (deg>=3 at
      an X-intersection or Y-junction). Downstream eval_tip then raises
      "Edge N is not mapped to any polyline".
    - 'full' also breaks under multi-component networks where the
      per-outer-cycle simplify+intersection pass introduces a deg>=4
      crossing vertex (the typical Brazilian-disk fragmentation setup
      with multiple disjoint initial cracks).
    - 'half' uses build_polylines_half_branches, which splits a branched
      component into per-branch polylines and maps every edge correctly,
      at the cost of the per-junction column-rank deficit absorbed by the
      KKT null-space path in [[KKT.py]].
    """
    mode = str(requested).lower().strip()
    if mode != "auto":
        return mode
    if connectivity is None or connectivity.size == 0:
        return "full"
    C = np.asarray(connectivity, int)
    # Already branched -> 'half'.
    if int(np.bincount(C.ravel()).max()) >= 3:
        return "half"
    # Multi-component -> 'half' (intersection vertices will appear after
    # the initial simplify pass).
    parent: dict[int, int] = {}
    def _find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent.setdefault(parent[x], parent[x])
            x = parent[x]
        return x
    for v0, v1 in C:
        a, b = _find(int(v0)), _find(int(v1))
        if a != b:
            parent[a] = b
    n_components = len({_find(int(v)) for pair in C for v in pair})
    return "half" if n_components > 1 else "full"

def _save_total_as_bem_arrays(bem_dir, Sxx, Syy, Sxy):
    np.save(bem_dir / "Sxx.npy", np.asarray(Sxx, float))
    np.save(bem_dir / "Syy.npy", np.asarray(Syy, float))
    np.save(bem_dir / "Sxy.npy", np.asarray(Sxy, float))

# --- run -------------------------------------------------------------
from datetime import datetime as _dt
started_at = _dt.now()
ts = started_at.strftime("%Y%m%d_%H%M%S")
bem_dir = _repo_root / "output" / f"{ts}_{cfg.output_dir_name}"
bem_dir.mkdir(parents=True, exist_ok=True)
print(f"output: {bem_dir}")

# Provenance bookkeeping -- updated as the run progresses; flushed in
# the finally block so a crashed/interrupted run still leaves a record.
run_stats: dict = {
    "engine": ENGINE,
    "openmp_threads": int(bem_cpp.openmp_max_threads()),
    "bem_cpp_version": getattr(bem_cpp, "__version__", ""),
    "coupling_method": cfg.coupling_method,
    "max_steps_requested": int(cfg.max_steps),
    "bem_correction_frequency": int(cfg.bem_correction_frequency),
    "bem_corrections_fired": 0,
    "termination": "in_progress",
    "n_verts_initial": int(cfg.vertices.shape[0]),
    "n_edges_initial": int(cfg.connectivity.shape[0]),
    "n_verts_final": int(cfg.vertices.shape[0]),
    "n_edges_final": int(cfg.connectivity.shape[0]),
}

try:
    # BEM baseline (suppress the legacy rect-grid contour PNGs; the
    # polar overlay below renders the baseline field on the same
    # polar/clustered mesh used by every subsequent STEP_NN plot).
    ensure_bem_field(cfg.disk, bem_dir, recompute=(not cfg.skip_bem_solve),
                     show=False, save_contours=False)

    # Polar overlay so the per-cycle TOTAL plots render on a curvilinear
    # disk-shaped grid (smooth boundary) with contour lines.
    save_supplementary_grid(cfg, bem_dir, always_polar=True, show=True)

    boundary_mesh = build_boundary(cfg.boundary_spec)

    hook = make_standard_plot_hook(
        bem_dir=bem_dir, combined_out_dir=bem_dir, plot_params=cfg.plot,
        show_initial=False, show_cycles=False, show_final=False,
    )

    bem_block = BEMBlockV2(disk=cfg.disk, out_dir=bem_dir)
    aug_payload = bem_block.build_augmented_payload(
        d_mode=cfg.aug_d_mode, gauss_n=cfg.aug_gauss_n,
    )

    def _build_kwargs(crack_mode_effective: str):
        base = {
            "n_crack_elements": cfg.n_crack_elements,
            "parametrization": cfg.parametrization,
            "crack_mode": crack_mode_effective,
            "junction_model": cfg.junction_model,
            "soft_eta": cfg.soft_eta,
        }
        one_way = compose_solver_kwargs_v2("one_way", base_solver_kwargs=base)
        direct = compose_solver_kwargs_v2(
            "full_kkt",
            base_solver_kwargs={
                **base,
                "augmented_enforce_crack_equilibrium": cfg.aug_enforce_crack_equilibrium,
                "augmented_equilibrate": cfg.aug_equilibrate,
                "augmented_equilibrate_cols": cfg.aug_equilibrate_cols,
                "augmented_physical_scaling": cfg.aug_physical_scaling,
                "augmented_scale_traction_rows": cfg.aug_scale_traction_rows,
            },
            augmented_payload=aug_payload,
            augmented_ridge_q=cfg.aug_ridge_q,
            augmented_ridge_y=cfg.aug_ridge_y,
            augmented_keep_bem_applied=cfg.aug_keep_bem_applied,
        )
        return one_way, direct

    # Crack_mode='auto' resolves once from the initial connectivity. The
    # flat STEP driver does not refresh per-step (it would only matter
    # when an intersection during growth flips a single-chain to a
    # branched component, in which case 'half' would be needed -- but
    # such transitions are rare in practice for the disk pipeline).
    _current_mode = _pick_crack_mode(cfg.crack_mode, cfg.connectivity)
    one_way_kwargs, direct_kwargs = _build_kwargs(_current_mode)
    if str(cfg.crack_mode).lower().strip() == "auto":
        print(f"[crack_mode] auto -> {_current_mode}", flush=True)

    original_initial_vids = set(int(v) for v in cfg.vertices[:, 0])
    grow = _dc_replace(cfg.crack_growth, solver_kwargs=one_way_kwargs)

    # BEM correction hook fired by run_growth_steps every K steps. The
    # hook overwrites bem_dir/{Sxx,Syy,Sxy}.npy in place; run_growth_steps
    # rebuilds the propagator from the refreshed arrays for the next STEP.
    def _bem_correction_hook(res, step):
        run_stats["bem_corrections_fired"] += 1
        if cfg.coupling_method == "iterative":
            tx, ty = compute_crack_boundary_tractions_direct(
                res=res, boundary_mesh=boundary_mesh,
            )
            solve_bem_with_extra_boundary_tractions(
                disk_params=cfg.disk, bem_dir=bem_dir,
                tx_extra=-tx, ty_extra=-ty, show=False,
            )
        elif cfg.coupling_method == "direct":
            # Augmented full-KKT one-shot solve on the current network;
            # the resulting TOTAL field is persisted back as the new BEM
            # baseline so the next STEP's grow sees the corrected field.
            corr_dir = bem_dir / f"STEP_{step:02d}_direct_correction"
            corr_dir.mkdir(parents=True, exist_ok=True)
            grow_d = _dc_replace(
                cfg.crack_growth, max_cycles=0,
                solver_kwargs=direct_kwargs,
            )
            Vg, Cg = _network_to_arrays(res.calc.network)
            res_direct = run_network_growth_uncoupled(
                bem_dir=bem_dir, out_dir=corr_dir,
                vertices=Vg, connectivity=Cg,
                params=grow_d, plot_hook=None,
                original_initial_vertex_ids=original_initial_vids,
                plot_params=cfg.plot,
            )
            Sxx, Syy, Sxy = plot_total_field(
                bem_dir=bem_dir, res=res_direct, out_dir=corr_dir,
                tag=f"step_{step:02d}_direct", params=cfg.plot, show=False,
            )
            _save_total_as_bem_arrays(bem_dir, Sxx, Syy, Sxy)
        elif cfg.coupling_method == "no_coupling":
            pass
        else:
            raise ValueError(f"Unknown coupling_method: {cfg.coupling_method!r}")

    bem_freq = int(cfg.bem_correction_frequency)
    if cfg.coupling_method == "no_coupling":
        bem_freq = 0  # ignore freq when correction is explicitly off

    print(
        f"[STEP-driver] max_steps={cfg.max_steps} "
        f"coupling={cfg.coupling_method} bem_correction_frequency={bem_freq}",
        flush=True,
    )

    res_last = run_growth_steps(
        bem_dir=bem_dir, out_dir=bem_dir,
        vertices=cfg.vertices, connectivity=cfg.connectivity,
        params=grow,
        max_steps=int(cfg.max_steps),
        bem_correction_frequency=bem_freq,
        bem_correction_hook=_bem_correction_hook,
        plot_hook=hook,
        plot_params=cfg.plot,
        original_initial_vertex_ids=original_initial_vids,
    )

    V_final, C_final = _network_to_arrays(res_last.calc.network)
    run_stats["n_verts_final"] = int(V_final.shape[0])
    run_stats["n_edges_final"] = int(C_final.shape[0])
    stopped = str(getattr(res_last, "stopped_reason", "") or "")
    run_stats["termination"] = stopped if stopped else "completed"

    if cfg.stop_on_spanning_cluster:
        snapped = getattr(res_last, "snapped_vertex_ids", None) or set()
        clusters = find_spanning_clusters(
            res_last.calc.network,
            disk_R=cfg.disk.R,
            snapped_vertex_ids=snapped,
        )
        if clusters:
            n_clusters = len(clusters)
            n_boundary_pts = sum(len(b) for _, b in clusters)
            print(
                f"[note] spanning cluster present at end of run "
                f"({n_clusters} component(s), {n_boundary_pts} boundary vertex(es)).",
                flush=True,
            )
            if run_stats["termination"] == "completed":
                run_stats["termination"] = "spanning_cluster_at_end"

    print(
        f"\n[done] STEP-driver run completed: "
        f"{V_final.shape[0]} verts, {C_final.shape[0]} edges",
        flush=True,
    )

    # End-of-run alpha / Q / per-tip SIF curves.
    try:
        plot_step_metrics(getattr(res_last, "step_history", []) or [], bem_dir,
                          dpi=int(cfg.plot.dpi))
    except Exception as _pe:
        print(f"[step-metrics] skipped (error: {_pe})", flush=True)

except BaseException as _e:
    run_stats["termination"] = f"error: {type(_e).__name__}: {_e}"
    raise
finally:
    try:
        _prov = write_provenance(
            cfg=cfg, case_xlsx=xlsx, run_dir=bem_dir,
            run_stats=run_stats, started_at=started_at,
        )
        print(f"[provenance] wrote {_prov['json'].name}", flush=True)
    except Exception as _pe:
        print(f"[provenance] skipped (error: {_pe})", flush=True)

# Final video stitch (best-effort)
try:
    _tools_dir = str(_repo_root / "tools")
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from make_run_videos import make_videos as _make_videos
    _make_videos(bem_dir, fps=5)
except Exception as _e:
    print(f"[videos] skipped (error: {_e})", flush=True)

print(f"[summary] output: {bem_dir}")
