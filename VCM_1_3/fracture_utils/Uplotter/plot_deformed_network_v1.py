"""
Deformed network plotter (v14)

Based on v13 (user-provided canonical behavior for solver_option='parametrized_crack').

Enhancement in v14:
- Optional trimming for "core" junction_model at vertices with degree > 2:
  ensure that, in cyclic order around the vertex, the *lower* face of branch j
  meets exactly at one point with the *upper* face of branch j+1 (mod m). This
  prevents face crossings at the junction and produces a small polygonal gap.

Implementation notes:
- v13 already concatenates polyline chains into a single array per physical branch,
  which removes breaks at internal degree-2 kinks.
- The trimming here operates only on the first point of U/L at the junction for each
  incident polyline endpoint, based on intersection of the first face segments (rays).
  This is a visualization-only postprocess; it does not alter solver fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

# Local helper (provided in this package by the user)
from .reconstruct import reconstruct_cod_csd_parametrized_smoothed
from .smooth import moving_average_nan, resolve_smooth_window


def _deg_map_from_network(net) -> dict[int, int]:
    deg = {int(v.id): 0 for v in getattr(net, "vertices", [])}
    for e in getattr(net, "edges", []):
        v0 = int(getattr(e, "v0"))
        v1 = int(getattr(e, "v1"))
        deg[v0] = deg.get(v0, 0) + 1
        deg[v1] = deg.get(v1, 0) + 1
    return deg


def _infer_ne_half(sol: dict) -> int:
    """Best-effort extraction of ne_half from solver output dict."""
    if not isinstance(sol, dict):
        return 0
    for k in ("ne_half", "nh", "n_half"):
        v = sol.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    params = sol.get("params", None)
    if isinstance(params, dict):
        v = params.get("ne_half", None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    pls = sol.get("polyline_solutions", None)
    if isinstance(pls, list) and pls:
        bI = pls[0].get("bI", None)
        if bI is not None:
            try:
                return max(1, int(len(bI)//2))
            except Exception:
                pass
    return 0


def _unit_frame(p0: np.ndarray, p1: np.ndarray):
    t = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.hypot(t[0], t[1]))
    if L <= 0.0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0
    t_hat = t / L
    n_hat = np.array([-t_hat[1], t_hat[0]])
    return t_hat, n_hat, L


def _enforce_tip_zero_polyline(P: np.ndarray, J: np.ndarray, start_is_tip: bool, end_is_tip: bool):
    """
    Enforce J=0 at selected endpoints of a *continuous polyline* (single array).

    - If start_is_tip: subtract constant J(0)
    - If end_is_tip: remove an affine drift so J(L)=0 without changing interior shape too aggressively
    """
    P = np.asarray(P, float)
    J = np.asarray(J, float)
    if P.shape != J.shape or P.shape[0] < 2:
        return J

    if start_is_tip:
        J = J - J[0].reshape(1, 2)

    if end_is_tip:
        ds = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        Ltot = float(s[-1]) if s.size else 0.0
        if Ltot > 0.0:
            alpha = (s / Ltot).reshape(-1, 1)
            J_end = J[-1].reshape(1, 2)
            J = J - alpha * J_end

    return J


def _sample_arc_xy(meta: dict, n: int = 2000) -> np.ndarray:
    c = np.asarray(meta.get("arc_center", [0.0, 0.0]), float).reshape(2,)
    r = float(meta.get("arc_radius", 0.0))
    th0 = float(meta.get("arc_theta0", 0.0))
    dth = float(meta.get("arc_dtheta", 0.0))
    t = np.linspace(0.0, 1.0, max(2, int(n)))
    th = th0 + dth * t
    xy = c.reshape(1, 2) + r * np.c_[np.cos(th), np.sin(th)]
    return xy


def _reconstruct_jump_from_b(solp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct jump vector J(s) along a parametrized curve from densities.
    Uses the kinematic relation:
        dJ/ds = -(bII * t + bI * n)
    so J(s) = J0 - ∫ (bII t + bI n) ds.
    """
    P = np.asarray(solp.get("x_mid", []), float)
    t = np.asarray(solp.get("t_mid", []), float)
    n = np.asarray(solp.get("n_mid", []), float)
    ds = np.asarray(solp.get("ds", []), float).reshape(-1,)

    if P.ndim != 2 or P.shape[0] < 2:
        return P, np.zeros_like(P), np.zeros((P.shape[0],), float)

    bI  = np.asarray(solp.get("bI", 0.0), float).reshape(-1,)
    bII = np.asarray(solp.get("bII", 0.0), float).reshape(-1,)
    N = P.shape[0]
    if bI.size != N:
        bI = np.resize(bI, N)
    if bII.size != N:
        bII = np.resize(bII, N)

    if ds.size != N:
        dP = np.diff(P, axis=0)
        ds2 = np.hypot(dP[:,0], dP[:,1])
        ds = np.concatenate([[ds2[0]], ds2]) if ds2.size else np.ones((N,), float)

    incr = (bII.reshape(-1,1) * t + bI.reshape(-1,1) * n) * ds.reshape(-1,1)

    J0 = np.asarray(solp.get("J0", [0.0, 0.0]), float).reshape(2,)
    J = np.zeros((N,2), float)
    J[0,:] = J0
    for i in range(1, N):
        J[i,:] = J[i-1,:] - incr[i,:]

    dP = np.diff(P, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(dP[:,0], dP[:,1]))])
    return P, J, s


def _angle_of_vector(v: np.ndarray) -> float:
    v = np.asarray(v, float).reshape(2,)
    return float(math.atan2(v[1], v[0]))


def _line_intersection(p: np.ndarray, r: np.ndarray, q: np.ndarray, s: np.ndarray):
    """
    Intersection of two parametric lines:
        L1: p + t r
        L2: q + u s
    Returns (ok, t, u, x)
    """
    p = np.asarray(p, float).reshape(2,)
    r = np.asarray(r, float).reshape(2,)
    q = np.asarray(q, float).reshape(2,)
    s = np.asarray(s, float).reshape(2,)

    def cross2(a, b):
        return float(a[0]*b[1] - a[1]*b[0])

    denom = cross2(r, s)
    if abs(denom) < 1e-14:
        return False, 0.0, 0.0, 0.5*(p+q)
    qp = q - p
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    x = p + t*r
    return True, float(t), float(u), x


def _trim_core_junction_faces_inplace(poly_data: list[dict], deg: dict[int, int], junction_model: str,
                                      min_degree: int = 3, look_ahead: int = 1):
    """
    In-place postprocess for "core" junction_model:
    At each vertex with degree >= min_degree, enforce cyclic joining:
        L_j(0) == U_{j+1}(0)  (j mod m)

    poly_data entries must have:
      - start_vid, end_vid
      - P, U, L arrays (each (N,2))
    """
    if str(junction_model).lower() != "core":
        return

    # Gather incident polyline endpoints per vertex
    # Each endpoint record: (angle, poly_idx, end_tag, local_U0U1, local_L0L1, tdir)
    incidents: dict[int, list[dict]] = {}

    for idx, d in enumerate(poly_data):
        P = d["P"]; U = d["U"]; L = d["L"]
        if P.shape[0] < 2 or U.shape[0] < 2 or L.shape[0] < 2:
            continue

        # start endpoint
        v = int(d["start_vid"])
        if deg.get(v, 0) >= min_degree:
            tdir = P[1] - P[0]
            ang = _angle_of_vector(tdir)
            incidents.setdefault(v, []).append({
                "angle": ang, "poly_idx": idx, "end": "start",
                "U0": U[0].copy(), "U1": U[min(look_ahead, U.shape[0]-1)].copy(),
                "L0": L[0].copy(), "L1": L[min(look_ahead, L.shape[0]-1)].copy(),
            })

        # end endpoint (reverse local orientation so "outgoing" is away from the vertex)
        v = int(d["end_vid"])
        if deg.get(v, 0) >= min_degree:
            tdir = P[-2] - P[-1]  # outward from end vertex
            ang = _angle_of_vector(tdir)
            incidents.setdefault(v, []).append({
                "angle": ang, "poly_idx": idx, "end": "end",
                "U0": U[-1].copy(), "U1": U[max(-1-look_ahead, -U.shape[0])].copy(),
                "L0": L[-1].copy(), "L1": L[max(-1-look_ahead, -L.shape[0])].copy(),
            })

    # For each high-degree vertex, sort incident branches cyclically and compute intersections
    for vid, items in incidents.items():
        if len(items) < min_degree:
            continue

        items.sort(key=lambda z: z["angle"])
        m = len(items)

        # Intersections X_j between L_j ray and U_{j+1} ray
        X = [None] * m
        for j in range(m):
            a = items[j]
            b = items[(j + 1) % m]

            p = a["L0"]; r = a["L1"] - a["L0"]
            q = b["U0"]; s = b["U1"] - b["U0"]

            ok, t, u, x = _line_intersection(p, r, q, s)

            # Require forward intersection along both rays; otherwise fall back
            if (not ok) or (t < 0.0) or (u < 0.0) or (not np.all(np.isfinite(x))):
                x = 0.5 * (a["L0"] + b["U0"])

            X[j] = np.asarray(x, float).reshape(2,)

        # Apply cyclic join: set L_j at the vertex and U_{j+1} at the vertex to X_j
        for j in range(m):
            a = items[j]
            b = items[(j + 1) % m]
            x = X[j]

            # mutate arrays in poly_data
            da = poly_data[a["poly_idx"]]
            db = poly_data[b["poly_idx"]]

            if a["end"] == "start":
                da["L"][0, :] = x
            else:
                da["L"][-1, :] = x

            if b["end"] == "start":
                db["U"][0, :] = x
            else:
                db["U"][-1, :] = x


@dataclass
class DCEPlotterDeformedV4:
    res: object
    out_dir: Path | None = None

    def __post_init__(self):
        if self.out_dir is None:
            self.out_dir = Path(getattr(self.res, "out_dir", "output"))

    def plot_deformed_network(
        self,
        n_theta: int = 2000,
        scale: float = 5e1,
        show_faces: bool = True,
        units: str = "mm",
        cod_smooth_window: int = 5,
        csd_smooth_window: int = 5,
        stitch_polyline_kinks: bool = True,   # kept for signature compatibility; v13/v14 stitches by construction
        kink_angle_deg: float = 12.0,         # kept for signature compatibility
        trim_core_junction_faces: bool = True,
        show: bool = True,
        save: bool = False,
        debug_counts: bool = True,
    ):
        res = self.res
        sol = getattr(res, "sol", None)
        calc = getattr(res, "calc", None)
        if calc is None or getattr(calc, "network", None) is None:
            raise AttributeError("res.calc.network is required for plotting.")

        net = calc.network
        deg = _deg_map_from_network(net)

        fig, ax = plt.subplots(figsize=(6.5, 6.5))

        n_plotted = 0
        n_skipped = 0

        # ------------------------------------------------------------
        # Preferred path: parametrized cracks with polylines metadata
        # ------------------------------------------------------------
        if isinstance(sol, dict) and str(sol.get("solver_option", "")).lower() == "parametrized_crack":
            polylines = list(sol.get("parametrized_polylines", []))
            if not polylines:
                raise ValueError("solver_option='parametrized_crack' but parametrized_polylines is empty.")

            junction_model = str(sol.get("junction_model", sol.get("constraints", {}).get("junction_model", "parametrized_crack"))).lower()

            # color cycle (robust)
            prop = plt.rcParams.get("axes.prop_cycle", None)
            color_list = prop.by_key().get("color", []) if prop is not None else []
            if not color_list:
                color_list = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

            poly_data: list[dict] = []

            # First pass: reconstruct and concatenate each polyline into one array, build faces
            for pid, meta in enumerate(polylines):
                kind = str(meta.get("kind", "polyline")).lower()
                if kind == "arc":
                    try:
                        sol_list = list(sol.get("polyline_solutions", []))
                        solp = sol_list[pid] if pid < len(sol_list) else None
                        if solp is None:
                            n_skipped += 1
                            continue

                        P_plot = _sample_arc_xy(meta, n=int(n_theta))

                        P_mid, J_mid, s_mid = _reconstruct_jump_from_b(solp)
                        if P_mid.shape[0] < 2:
                            n_skipped += 1
                            continue

                        dP = np.diff(P_plot, axis=0)
                        s_plot = np.concatenate([[0.0], np.cumsum(np.hypot(dP[:,0], dP[:,1]))])
                        Lp = float(s_plot[-1]) if s_plot.size else 0.0
                        if Lp <= 0.0:
                            n_skipped += 1
                            continue

                        s_mid_clip = np.clip(s_mid, 0.0, Lp)
                        Jx = np.interp(s_plot, s_mid_clip, J_mid[:,0])
                        Jy = np.interp(s_plot, s_mid_clip, J_mid[:,1])
                        J_plot = np.c_[Jx, Jy]

                        start_vid = int(meta.get("v_start", -1))
                        end_vid   = int(meta.get("v_end", -1))
                        start_is_tip = (start_vid < 0) or (deg.get(start_vid, 0) == 1)
                        end_is_tip   = (end_vid < 0) or (deg.get(end_vid, 0) == 1)
                        J_plot = _enforce_tip_zero_polyline(P_plot, J_plot, start_is_tip=start_is_tip, end_is_tip=end_is_tip)

                        U = P_plot + 0.5 * float(scale) * J_plot
                        Lw = P_plot - 0.5 * float(scale) * J_plot

                        poly_data.append({
                            "pid": pid,
                            "color": color_list[pid % len(color_list)],
                            "start_vid": start_vid,
                            "end_vid": end_vid,
                            "P": P_plot,
                            "U": U,
                            "L": Lw,
                        })
                        continue
                    except Exception:
                        n_skipped += 1
                        continue
                if kind == "cspline":
                    try:
                        sol_list = list(sol.get("polyline_solutions", []))
                        solp = sol_list[pid] if pid < len(sol_list) else None
                        if solp is None:
                            n_skipped += 1
                            continue

                        # Reconstruct jump at solver midpoints, then resample for smooth plotting
                        P_mid, J_mid, s_mid = _reconstruct_jump_from_b(solp)
                        if P_mid.shape[0] < 2:
                            n_skipped += 1
                            continue

                        nh = _infer_ne_half(sol)
                        w_cod = resolve_smooth_window(cod_smooth_window, nh, C=50, wmin=3)
                        w_csd = resolve_smooth_window(csd_smooth_window, nh, C=50, wmin=3)
                        wJ = max(int(w_cod), int(w_csd))
                        if wJ > 1:
                            Jx = moving_average_nan(J_mid[:, 0], wJ)
                            Jy = moving_average_nan(J_mid[:, 1], wJ)
                            J_mid = np.c_[Jx, Jy]

                        s_plot = np.linspace(float(s_mid[0]), float(s_mid[-1]), max(10, int(n_theta)))
                        Px = np.interp(s_plot, s_mid, P_mid[:, 0])
                        Py = np.interp(s_plot, s_mid, P_mid[:, 1])
                        Jx = np.interp(s_plot, s_mid, J_mid[:, 0])
                        Jy = np.interp(s_plot, s_mid, J_mid[:, 1])
                        P_plot = np.c_[Px, Py]
                        J_plot = np.c_[Jx, Jy]

                        start_vid = int(meta.get("v_start", -1))
                        end_vid   = int(meta.get("v_end", -1))
                        start_is_tip = (start_vid < 0) or (deg.get(start_vid, 0) == 1)
                        end_is_tip   = (end_vid < 0) or (deg.get(end_vid, 0) == 1)
                        J_plot = _enforce_tip_zero_polyline(P_plot, J_plot, start_is_tip=start_is_tip, end_is_tip=end_is_tip)

                        U = P_plot + 0.5 * float(scale) * J_plot
                        Lw = P_plot - 0.5 * float(scale) * J_plot

                        poly_data.append({
                            "pid": pid,
                            "color": color_list[pid % len(color_list)],
                            "start_vid": start_vid,
                            "end_vid": end_vid,
                            "P": P_plot,
                            "U": U,
                            "L": Lw,
                        })
                        continue
                    except Exception:
                        n_skipped += 1
                        continue

                try:
                    path_vids = [int(v) for v in meta.get("path_vertex_ids", [])]
                    path_edges = [int(i) for i in meta.get("path_edge_indices", [])]
                    segL = np.asarray(meta.get("segment_lengths", []), float)
                    if len(path_vids) < 2 or len(path_edges) < 1:
                        n_skipped += 1
                        continue
                    if len(segL) != len(path_vids) - 1:
                        # fall back to geometric length if not provided
                        segL = np.zeros(len(path_vids) - 1, float)
                        for k in range(len(segL)):
                            p0 = np.asarray(net.vertex_coords(path_vids[k]), float)
                            p1 = np.asarray(net.vertex_coords(path_vids[k + 1]), float)
                            segL[k] = float(np.hypot(*(p1 - p0)))
                except Exception:
                    n_skipped += 1
                    continue

                P_all = []
                J_all = []
                ok_any = False

                for k, edge_index in enumerate(path_edges):
                    try:
                        x_edge, COD, CSD = reconstruct_cod_csd_parametrized_smoothed(
                            res,
                            edge_index=edge_index,
                            n_pts=int(n_theta),
                            enforce_global_tip_zero=False,  # endpoint enforcement on concatenated polyline
                            cod_window_panels=int(cod_smooth_window),
                            csd_window_panels=int(csd_smooth_window),
                        )
                        COD = np.asarray(COD, float).reshape(-1,)
                        CSD = np.asarray(CSD, float).reshape(-1,)
                        x_edge = np.asarray(x_edge, float).reshape(-1,)
                        if x_edge.size < 2:
                            raise ValueError("reconstruction returned too few points.")
                    except Exception:
                        n_skipped += 1
                        continue

                    v0 = int(path_vids[k])
                    v1 = int(path_vids[k + 1])
                    p0 = np.asarray(net.vertex_coords(v0), float)
                    p1 = np.asarray(net.vertex_coords(v1), float)
                    t_hat, n_hat, L = _unit_frame(p0, p1)
                    if L <= 0.0:
                        n_skipped += 1
                        continue

                    # Map x_edge in [-a,a] to param in [0,1] along the segment
                    a = 0.5 * float(segL[k]) if k < len(segL) else 0.5 * L
                    if not np.isfinite(a) or a <= 0.0:
                        a = 0.5 * L

                    tpar = (x_edge + a) / (2.0 * a)
                    tpar = np.clip(tpar, 0.0, 1.0)
                    P_seg = p0.reshape(1, 2) + tpar.reshape(-1, 1) * (p1 - p0).reshape(1, 2)

                    # Global jump vector J = CSD*t + COD*n
                    J_seg = CSD.reshape(-1, 1) * t_hat.reshape(1, 2) + COD.reshape(-1, 1) * n_hat.reshape(1, 2)

                    if not P_all:
                        P_all.append(P_seg)
                        J_all.append(J_seg)
                    else:
                        # drop the first point to avoid duplication at the shared vertex
                        P_all.append(P_seg[1:].copy())
                        J_all.append(J_seg[1:].copy())

                    ok_any = True

                if (not ok_any) or (not P_all):
                    continue

                P = np.vstack(P_all)
                J = np.vstack(J_all)

                # Enforce tip closure on the polyline endpoints (deg-1)
                start_vid = int(path_vids[0])
                end_vid   = int(path_vids[-1])
                J = _enforce_tip_zero_polyline(
                    P, J,
                    start_is_tip=(deg.get(start_vid, 0) == 1),
                    end_is_tip=(deg.get(end_vid, 0) == 1),
                )

                U = P + 0.5 * float(scale) * J
                Lw = P - 0.5 * float(scale) * J

                poly_data.append({
                    "pid": pid,
                    "color": color_list[pid % len(color_list)],
                    "start_vid": start_vid,
                    "end_vid": end_vid,
                    "P": P,
                    "U": U,
                    "L": Lw,
                })

            # Second pass: optional core-junction trimming for degree>2 vertices
            if trim_core_junction_faces and poly_data:
                _trim_core_junction_faces_inplace(poly_data, deg=deg, junction_model=junction_model, min_degree=3, look_ahead=1)

            # Third pass: plot
            for d in poly_data:
                P = d["P"]; U = d["U"]; Lw = d["L"]
                col = d["color"]
                ax.plot(P[:, 0], P[:, 1], linestyle=":", linewidth=1.0, color="0.5", alpha=0.8)
                if show_faces:
                    ax.plot(U[:, 0], U[:, 1], linewidth=1.6, color=col)
                    ax.plot(Lw[:, 0], Lw[:, 1], linewidth=1.6, color=col, alpha=0.85)
                n_plotted += 1

            ax.set_title(f"Deformed Crack Network (junction_model={junction_model}, scale={float(scale):.2e})")

        else:
            # ------------------------------------------------------------
            # Fallback: plot undeformed network midlines
            # ------------------------------------------------------------
            ax.set_title(f"Deformed Crack Network (scale={float(scale):.2e})")
            for e in getattr(net, "edges", []):
                try:
                    p0 = np.asarray(net.vertex_coords(int(e.v0)), float)
                    p1 = np.asarray(net.vertex_coords(int(e.v1)), float)
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", linewidth=1.2, alpha=0.7)
                    n_plotted += 1
                except Exception:
                    n_skipped += 1
                    continue

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.grid(True, alpha=0.25)

        if debug_counts:
            ax.text(
                0.02, 0.98,
                f"plotted={n_plotted}, skipped={n_skipped}",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=9
            )

        if save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fpath = self.out_dir / "deformed_network.png"
            fig.savefig(fpath, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax
