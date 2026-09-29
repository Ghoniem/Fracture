"""Panel-layout + J-field mirror-residual diagnostic.

Builds the post-cycle-0-step-1 Z-shape (analytically mirror-symmetric about
the origin), solves it with the same engine the SimulationsCpp notebook uses,
and reports:

  (1) Panel allocation per segment, per polyline edge.
  (2) Panel midpoints and J-mid vectors, with mirror-twin residuals.
  (3) SIF extraction frame (ex, ey) at each tip.
  (4) KI, KII, K_eff at each tip.

If panel-layout is mirror-symmetric and J-mid is mirror-symmetric but KI is
still asymmetric, the bug is in the SIF *fit/frame*. If the J-mid itself is
asymmetric, the bug is upstream (BEM solve or panel allocation).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faulthandler
faulthandler.enable()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vcem"))
sys.path.insert(0, str(REPO / "vcem" / "cpp" / "python"))


def load_bem_sigma_func(bem_dir: Path):
    from scipy.interpolate import RegularGridInterpolator
    xs = np.load(bem_dir / "xs.npy")
    ys = np.load(bem_dir / "ys.npy")
    Sxx = np.nan_to_num(np.load(bem_dir / "Sxx.npy"), nan=0.0)
    Syy = np.nan_to_num(np.load(bem_dir / "Syy.npy"), nan=0.0)
    Sxy = np.nan_to_num(np.load(bem_dir / "Sxy.npy"), nan=0.0)
    Ixx = RegularGridInterpolator((ys, xs), Sxx, bounds_error=False, fill_value=0.0)
    Iyy = RegularGridInterpolator((ys, xs), Syy, bounds_error=False, fill_value=0.0)
    Ixy = RegularGridInterpolator((ys, xs), Sxy, bounds_error=False, fill_value=0.0)

    def sigma_func(X):
        X = np.asarray(X, float)
        pts = np.column_stack([X[:, 1], X[:, 0]])
        S = np.zeros((len(X), 2, 2))
        S[:, 0, 0] = Ixx(pts)
        S[:, 1, 1] = Iyy(pts)
        S[:, 0, 1] = Ixy(pts)
        S[:, 1, 0] = Ixy(pts)
        return S
    return sigma_func


def build_post_step1_zshape():
    """5-vertex mirror-symmetric Z, matching cycle 0 / step 1 output."""
    mm = 1e-3
    da = 0.635e-3
    v0 = (0.0, 0.0)
    v1 = (+2.0 * mm, +2.0 * mm)
    v2 = (-2.0 * mm, -2.0 * mm)
    th = math.radians(126.494)
    v3 = (v1[0] + da * math.cos(th), v1[1] + da * math.sin(th))
    v4 = (v2[0] - da * math.cos(th), v2[1] - da * math.sin(th))
    V = np.array([
        [0, v0[0], v0[1]],
        [1, v1[0], v1[1]],
        [2, v2[0], v2[1]],
        [3, v3[0], v3[1]],
        [4, v4[0], v4[1]],
    ], dtype=float)
    C = np.array([[0, 1], [0, 2], [1, 3], [2, 4]], dtype=int)
    return V, C


def main():
    from preamble import (
        Material, AppliedStress, CrackNetworkV4,
        CandidateEvaluator, MaximumHoopStressLaw,
    )
    from fracture_utils.Upropagation.network_ops import get_netops

    bem_dir = REPO / "vcem" / "output" / "BEM_Brazilian_disk_iterative"
    sigma_func = load_bem_sigma_func(bem_dir)

    V, C = build_post_step1_zshape()
    print("=== Geometry mirror check ===")
    for vid in (3, 4):
        row = V[vid]
        twin = V[7 - vid]  # 3↔4
        print(f"  vid={int(row[0])} xy=({row[1]:+.6e},{row[2]:+.6e})  "
              f"twin xy=({twin[1]:+.6e},{twin[2]:+.6e})  "
              f"sum=({row[1]+twin[1]:+.3e},{row[2]+twin[2]:+.3e})")
    print()

    net = CrackNetworkV4.from_vertices_connectivity(
        vertices=V, connectivity=C, Nv_max=4, validate=True,
    )

    material = Material(E=200e9, nu=0.30, plane_stress=False)
    applied = AppliedStress(sigma_func=sigma_func)
    # Use the same solver_kwargs as the notebook, with engine='cpp'.
    sk = dict(
        n_crack_elements=100,
        representation="cheb_quad",
        node_distribution="tip_dense",
        solver_option="parametrized_crack",
        parametrization="polyline",
        nq_col=12,
        nq_stress=16,
        crack_mode="half",
        junction_model="core",
        engine="cpp",
    )
    evaluator = CandidateEvaluator(
        material=material, applied=applied,
        solver_kwargs=sk,
        direction_law=MaximumHoopStressLaw(),
        enable_n_crack_elements_escalation=False,
    )

    print("=== Solving Z-shape (engine=cpp) ===")
    res = evaluator.solve_results(net)
    print("solve done.")
    print()

    # ----------------------------------------------------------------
    # (1) Panel allocation per segment
    # ----------------------------------------------------------------
    sol_dict = res.sol if hasattr(res, "sol") else {}
    parametrized_polylines = sol_dict.get("parametrized_polylines", []) if isinstance(sol_dict, dict) else []
    polyline_solutions = sol_dict.get("polyline_solutions", []) if isinstance(sol_dict, dict) else []
    print("=== Panel layout per polyline ===")
    print(f"  n_polylines = {len(parametrized_polylines)}")
    for i, (p, sp) in enumerate(zip(parametrized_polylines, polyline_solutions)):
        Np = int(sp.get("Np", -1))
        s_nodes = np.asarray(sp.get("s_nodes", []), float)
        ds_arr = np.asarray(sp.get("ds", []), float)
        segL = np.asarray(p.get("segment_lengths", []), float)
        vids_path = list(p.get("path_vertex_ids", []))
        Ltot = float(p.get("total_length", s_nodes[-1] if s_nodes.size else 0.0))
        print(f"  pid={i} Np={Np} Ltot={Ltot:.4e} vids_path={vids_path}")
        print(f"    segment_lengths = {segL}")
        print(f"    panels-per-segment counted from s_mid:")
        mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
        s_breaks = np.concatenate([[0.0], np.cumsum(segL)])
        for k in range(len(segL)):
            lo, hi = s_breaks[k], s_breaks[k + 1]
            in_seg = np.sum((mid > lo - 1e-12) & (mid < hi + 1e-12))
            print(f"      seg{k}: vid {vids_path[k]} -> {vids_path[k+1]}, "
                  f"L={hi-lo:.4e}, panels={int(in_seg)}")
        print(f"    ds (panel lengths, first 8): {np.array2string(ds_arr[:8], precision=4)}")
        print(f"    ds (panel lengths,  last 8): {np.array2string(ds_arr[-8:], precision=4)}")
        if s_nodes.size:
            mirror_s = (s_nodes + s_nodes[::-1]) - Ltot
            print(f"    s_nodes mirror residual max = {float(np.max(np.abs(mirror_s))):.4e}  "
                  f"(0 = perfect mirror about Ltot/2)")
        if ds_arr.size:
            mirror_ds = ds_arr - ds_arr[::-1]
            print(f"    ds       mirror residual max = {float(np.max(np.abs(mirror_ds))):.4e}")
    print()

    # ----------------------------------------------------------------
    # (2) Per-tip SIF reconstruction with mirror-residual on J_mid
    # ----------------------------------------------------------------
    netops = get_netops()
    deg = netops.degree_map(net)
    polylines = netops.extract_open_polylines(net)
    print(f"polyline vids (canonical): {[list(p.vids) for p in polylines]}")
    tips = netops.extract_deg1_tips(net, polylines, deg)

    edge_data: Dict[int, Dict[str, Any]] = {}
    for t in tips:
        # find edge incident to tip
        v_tip = int(t.v_tip)
        edge_idx = None
        for ei, e in enumerate(net.edges):
            if int(e.v0) == v_tip or int(e.v1) == v_tip:
                edge_idx = ei
                break
        x_mid, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(
            edge_index=int(edge_idx),
            enforce_global_tip_zero=True,
            tip_xy=t.x_tip,
        )
        edge_data[v_tip] = dict(
            tip=t,
            edge_idx=edge_idx,
            x_mid=np.asarray(x_mid, float),
            xy_mid=np.asarray(extra["xy_mid"], float),
            J_mid=np.asarray(extra["J_mid"], float),
            ex=np.asarray(extra["ex"], float),
            ey=np.asarray(extra["ey"], float),
            COD=np.asarray(COD, float),
            CSD=np.asarray(CSD, float),
            bI=np.asarray(extra.get("bI", []), float),
            bII=np.asarray(extra.get("bII", []), float),
        )

    # Pair up tip 3 (top) and tip 4 (bottom)
    top = edge_data.get(3)
    bot = edge_data.get(4)
    if top is None or bot is None:
        print("Could not find both tips vid=3 and vid=4 — got:", list(edge_data.keys()))
        return

    print()
    print("=== Tip-edge frames ===")
    for label, ed in (("top vid=3", top), ("bot vid=4", bot)):
        print(f"  {label}:  edge_idx={ed['edge_idx']}  "
              f"ex=({ed['ex'][0]:+.4f},{ed['ex'][1]:+.4f})  "
              f"ey=({ed['ey'][0]:+.4f},{ed['ey'][1]:+.4f})")
    print("  ex_top + ex_bot (should be 0,0 for mirror):",
          (top["ex"] + bot["ex"]).round(6))
    print("  ey_top - ey_bot (NOTE: code uses polyline ey at both, expect equal):",
          (top["ey"] - bot["ey"]).round(6))
    print()

    # Per-panel mirror residual
    print("=== Panel midpoint + J mirror residuals (top vs bot) ===")
    xy_t = top["xy_mid"]; J_t = top["J_mid"]
    xy_b = bot["xy_mid"]; J_b = bot["J_mid"]
    print(f"  top edge: n_panels={len(xy_t)}")
    print(f"  bot edge: n_panels={len(xy_b)}")
    n = min(len(xy_t), len(xy_b))
    # For mirror panels: bot panel reversed should be -top panel
    # Try both orderings to find which lines up.
    def residual(perm_bot_xy, perm_bot_J):
        dxy = perm_bot_xy + xy_t[:n]   # mirror through origin: sum should be 0
        dJ  = perm_bot_J  + J_t[:n]    # vector J mirrors as -J
        return float(np.max(np.abs(dxy))), float(np.max(np.abs(dJ)))

    # Same order
    res_same = residual(xy_b[:n], J_b[:n])
    # Reversed
    res_rev  = residual(xy_b[:n][::-1], J_b[:n][::-1])
    print(f"  same-order:  max|xy_top + xy_bot|={res_same[0]:.4e}  "
          f"max|J_top + J_bot|={res_same[1]:.4e}")
    print(f"  reversed:    max|xy_top + xy_bot|={res_rev[0]:.4e}  "
          f"max|J_top + J_bot|={res_rev[1]:.4e}")
    use_rev = res_rev[0] < res_same[0]
    print(f"  -> using {'reversed' if use_rev else 'same'} ordering for per-panel dump")
    xy_b_ord = xy_b[:n][::-1] if use_rev else xy_b[:n]
    J_b_ord  = J_b[:n][::-1]  if use_rev else J_b[:n]

    print()
    print("  panel  xy_top              xy_bot            xy_sum    "
          "|J_top|     |J_bot|     |J_top+J_bot|")
    for k in range(min(n, 12)):
        Jt = J_t[k]; Jb = J_b_ord[k]
        xyt = xy_t[k]; xyb = xy_b_ord[k]
        xysum = xyt + xyb
        nrm_Jt = float(np.hypot(*Jt))
        nrm_Jb = float(np.hypot(*Jb))
        nrm_sum = float(np.hypot(*(Jt + Jb)))
        print(f"   {k:>3}  ({xyt[0]:+.3e},{xyt[1]:+.3e})  "
              f"({xyb[0]:+.3e},{xyb[1]:+.3e})  "
              f"({xysum[0]:+.1e},{xysum[1]:+.1e})  "
              f"{nrm_Jt:.3e}  {nrm_Jb:.3e}  {nrm_sum:.3e}")

    print()
    # Mode-wise mirror residual on COD/CSD (these should sign-flip under mirror)
    COD_t = top["COD"]; CSD_t = top["CSD"]
    COD_b = bot["COD"]; CSD_b = bot["CSD"]
    if use_rev:
        COD_b = COD_b[::-1]; CSD_b = CSD_b[::-1]
    n = min(len(COD_t), len(COD_b))
    print("=== Per-panel COD / CSD mirror residuals (sign depends on frame convention) ===")
    # Under perfect mirror symmetry and consistent frame convention, KI > 0 at both
    # tips means COD has the SAME sign (additive residual ≠ 0 expected).
    # KII opposite sign means CSD has OPPOSITE sign (additive residual ≈ 0 expected).
    cod_sum = COD_t[:n] + COD_b[:n]
    cod_diff = COD_t[:n] - COD_b[:n]
    csd_sum = CSD_t[:n] + CSD_b[:n]
    csd_diff = CSD_t[:n] - CSD_b[:n]
    print(f"  max|COD_top + COD_bot| = {np.max(np.abs(cod_sum)):.4e}   "
          f"max|COD_top - COD_bot| = {np.max(np.abs(cod_diff)):.4e}")
    print(f"  max|CSD_top + CSD_bot| = {np.max(np.abs(csd_sum)):.4e}   "
          f"max|CSD_top - CSD_bot| = {np.max(np.abs(csd_diff)):.4e}")
    print()

    # ----------------------------------------------------------------
    # (3) Final SIFs at the two tips
    # ----------------------------------------------------------------
    print("=== SIFs at tips ===")
    for t in tips:
        tev = evaluator.eval_tip(res, t)
        which = t.tip_id.which
        print(f"  vid={t.v_tip:>2} which={which:>5} "
              f"xy=({t.x_tip[0]:+.4e},{t.x_tip[1]:+.4e}) "
              f"KI={tev.KI*1e-6:+.4e} MPa.sqrt(m)  KII={tev.KII*1e-6:+.4e}  "
              f"keff={tev.keff*1e-6:.4e}  theta={math.degrees(tev.theta):+.3f} deg")


if __name__ == "__main__":
    main()
