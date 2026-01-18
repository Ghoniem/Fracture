"""Panel-aware reconstruction used by the plotter for parametrized cracks."""
from __future__ import annotations
import numpy as np
from .plot_smooth import moving_average_nan

def reconstruct_cod_csd_parametrized_smoothed(
    res,
    edge_index: int,
    n_pts: int = 4000,
    enforce_global_tip_zero: bool = True,
    cod_window_panels: int = 0,
    csd_window_panels: int = 0,
):
    edge_index = int(edge_index)
    sol = getattr(res, "sol", None)
    if not isinstance(sol, dict) or str(sol.get("solver_option", "")).lower() != "parametrized_crack":
        raise AttributeError("This reconstruction is only valid for solver_option='parametrized_crack'.")

    e2p = sol.get("parametrized_edge_to_polyline", {})
    pid = int(e2p.get(edge_index, -1))
    polylines = list(sol.get("parametrized_polylines", []))
    poly_solutions = list(sol.get("polyline_solutions", []))
    if pid < 0 or pid >= len(polylines) or pid >= len(poly_solutions):
        raise ValueError("polyline metadata/solution missing or inconsistent.")

    meta = dict(polylines[pid])
    solp = dict(poly_solutions[pid])

    path_vids = [int(v) for v in meta.get("path_vertex_ids", [])]
    path_edges = [int(i) for i in meta.get("path_edge_indices", [])]
    segL = np.asarray(meta.get("segment_lengths", []), float)
    Ltot = float(meta.get("total_length", np.sum(segL)))

    if edge_index not in path_edges:
        raise ValueError(f"Edge {edge_index} is not in polyline path for pid={pid}.")

    s_vert = np.zeros(len(path_vids), float)
    if len(segL) == len(path_vids) - 1:
        s_vert[1:] = np.cumsum(segL)

    k = path_edges.index(edge_index)
    s0 = float(s_vert[k])
    Le = float(segL[k])
    a = 0.5 * Le

    edge = res.calc.network.edges[edge_index]
    v_start = int(path_vids[k])
    v_end = int(path_vids[k + 1])
    if int(edge.v0) == v_start and int(edge.v1) == v_end:
        dir_sign = +1
    elif int(edge.v1) == v_start and int(edge.v0) == v_end:
        dir_sign = -1
    else:
        dir_sign = +1

    n_pts = int(max(200, n_pts))
    x_edge = np.linspace(-a, a, n_pts)

    if dir_sign == +1:
        s = s0 + (x_edge + a)
    else:
        s = s0 + (a - x_edge)

    s_nodes = np.asarray(solp.get("s_nodes"), float)
    bI = np.asarray(solp.get("bI"), float)
    bII = np.asarray(solp.get("bII"), float)
    t_col = np.asarray(solp.get("t_col"), float)
    n_col = np.asarray(solp.get("n_col"), float)
    ds = np.asarray(solp.get("ds"), float)
    Np = int(solp.get("Np", len(bI)))

    bI = bI.reshape(-1,)[:Np]
    bII = bII.reshape(-1,)[:Np]
    ds = ds.reshape(-1,)[:Np]
    t_col = t_col.reshape(-1, 2)[:Np, :]
    n_col = n_col.reshape(-1, 2)[:Np, :]

    if len(s_nodes) != Np + 1:
        s_nodes = np.linspace(0.0, Ltot, Np + 1)

    if int(cod_window_panels) and int(cod_window_panels) > 1:
        bI = moving_average_nan(bI, int(cod_window_panels))
    if int(csd_window_panels) and int(csd_window_panels) > 1:
        bII = moving_average_nan(bII, int(csd_window_panels))

    J_nodes = np.zeros((Np + 1, 2), float)
    for i in range(Np):
        dJ = (bII[i] * t_col[i] + bI[i] * n_col[i]) * float(ds[i])
        J_nodes[i + 1] = J_nodes[i] + dJ

    if enforce_global_tip_zero and Ltot > 0:
        J_end = J_nodes[-1].copy()
        if np.linalg.norm(J_end) > 0:
            alpha = (s_nodes / float(Ltot)).reshape(-1, 1)
            J_nodes = J_nodes - alpha * J_end.reshape(1, 2)

    s_clip = np.clip(s, 0.0, float(Ltot))
    idx = np.searchsorted(s_nodes, s_clip, side="right") - 1
    idx = np.clip(idx, 0, Np - 1)
    sL = s_nodes[idx]
    sR = s_nodes[idx + 1]
    w = np.where(sR > sL, (s_clip - sL) / (sR - sL), 0.0)
    J = (1.0 - w).reshape(-1, 1) * J_nodes[idx] + w.reshape(-1, 1) * J_nodes[idx + 1]

    v0 = res.calc.network.V(edge.v0)
    v1 = res.calc.network.V(edge.v1)
    p0 = np.array([float(v0.x), float(v0.y)], float)
    p1 = np.array([float(v1.x), float(v1.y)], float)
    t = p1 - p0
    L = float(np.hypot(t[0], t[1]))
    if L <= 0:
        R = np.eye(2)
    else:
        ex = t / L
        ey = np.array([-ex[1], ex[0]])
        R = np.column_stack([ex, ey])
    loc = (R.T @ J.T).T
    CSD = loc[:, 0]
    COD = loc[:, 1]
    return x_edge, COD, CSD
