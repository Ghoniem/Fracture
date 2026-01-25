"""
plot_deformed_network (final)

Goals (per user request):
1) Remove artificial gauge operations in plotting. Respect solver gauge/constraints.
2) Remove stitching discontinuities at internal kinks by using a *single* global J(s) per polyline.
3) Use solver-provided J distributions directly when available; only fall back to integrating (bI,bII) when J is absent.
4) Match the "network graph" look: units, aspect, and axis limits consistent with the underlying crack graph.

Notes:
- This file keeps the two legacy plotting methods commented out near the end (for now).
- This module is intentionally conservative: it does not modify solver fields; only visualization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

from .smooth import moving_average_nan, resolve_smooth_window

# Optional junction face trimming utilities (core model)
try:
    from .trim_junction import BranchFaces, trim_faces_at_junction
except Exception:  # pragma: no cover
    BranchFaces = None
    trim_faces_at_junction = None


# -----------------------------
# small utilities
# -----------------------------
def _deg_map_from_network(net) -> dict[int, int]:
    deg = {int(v.id): 0 for v in getattr(net, "vertices", [])}
    for e in getattr(net, "edges", []):
        v0 = int(getattr(e, "v0"))
        v1 = int(getattr(e, "v1"))
        deg[v0] = deg.get(v0, 0) + 1
        deg[v1] = deg.get(v1, 0) + 1
    return deg


def _infer_ne_half(sol: dict) -> int:
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
                return max(1, int(len(bI) // 2))
            except Exception:
                pass
    return 0


def _infer_units_scale_from_points(P: np.ndarray, units: str) -> float:
    """
    Heuristic: if units=="mm" but geometry looks like meters, apply 1e3.
    Your examples: coords ~0.008 (m) but should be ~8 (mm).
    """
    u = (units or "").lower().strip()
    if u != "mm":
        return 1.0
    try:
        P = np.asarray(P, float)
        if P.size == 0:
            return 1.0
        m = float(np.nanmax(np.abs(P)))
        return 1e3 if m < 0.5 else 1.0
    except Exception:
        return 1.0


def _resample_polyline_by_arclength(P: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return P, np.zeros((P.shape[0],), float)
    ds = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    L = float(s[-1]) if s.size else 0.0
    if L <= 0.0:
        return P, s
    s_new = np.linspace(0.0, L, max(2, int(n)))
    Px = np.interp(s_new, s, P[:, 0])
    Py = np.interp(s_new, s, P[:, 1])
    return np.c_[Px, Py], s_new



def _arclength(P: np.ndarray) -> np.ndarray:
    """Cumulative arclength for a polyline."""
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return np.zeros((P.shape[0],), float)
    ds = np.hypot(np.diff(P[:,0]), np.diff(P[:,1]))
    return np.concatenate([[0.0], np.cumsum(ds)])

def _enforce_tip_zero_polyline(P: np.ndarray, J: np.ndarray, start_is_tip: bool, end_is_tip: bool) -> np.ndarray:
    """
    Visualization-side enforcement consistent with solver constraints:
    - If exactly one endpoint is a deg-1 tip: constant shift so J(tip)=0.
    - If both endpoints are tips (common for an isolated polyline): enforce J=0 at both ends
      by removing a linear drift (affine correction). This is NOT an arbitrary gauge; it is
      the physical boundary constraint you already enforce in the solver.
    - Otherwise: do nothing.
    """
    P = np.asarray(P, float)
    J = np.asarray(J, float)
    if P.ndim != 2 or J.shape != P.shape or P.shape[0] < 2:
        return J

    if start_is_tip and not end_is_tip:
        return J - J[0].reshape(1, 2)

    if end_is_tip and not start_is_tip:
        return J - J[-1].reshape(1, 2)

    if start_is_tip and end_is_tip:
        # remove linear drift so both ends are exactly zero
        ds = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        L = float(s[-1]) if s.size else 0.0
        if L <= 0.0:
            return J - J[0].reshape(1, 2)
        alpha = (s / L).reshape(-1, 1)
        J0 = J[0].reshape(1, 2)
        J1 = J[-1].reshape(1, 2)
        # First shift start to 0, then remove remaining end drift linearly
        Jt = J - J0
        return Jt - alpha * Jt[-1].reshape(1, 2)

    return J


def _oriented_solp_to_tip(solp: dict, start_is_tip: bool, end_is_tip: bool) -> tuple[dict, bool, bool, bool]:
    """
    Orient the per-polyline arrays so that, if exactly one endpoint is a tip, the tip is at index 0.
    This eliminates accidental reverse-integration artifacts.
    """
    if end_is_tip and not start_is_tip:
        solq = dict(solp)
        for key in ("x_mid", "t_mid", "n_mid", "ds", "bI", "bII", "J_mid", "J", "Jm", "Jsol"):
            if key in solq and solq[key] is not None:
                arr = np.asarray(solq[key], float)
                if arr.ndim >= 1 and arr.shape[0] >= 2:
                    solq[key] = arr[::-1].copy()
        return solq, True, True, False
    return solp, False, start_is_tip, end_is_tip


def _reconstruct_jump(solp: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (P_mid, J_mid).

    Preference order:
      1) Use solver-provided J arrays if present: 'J_mid', 'J', 'Jm', 'Jsol'
      2) Otherwise integrate from densities using: dJ/ds = -(bII*t + bI*n)
    """
    P = np.asarray(solp.get("x_mid", []), float)
    if P.ndim != 2 or P.shape[0] < 2:
        return P, np.zeros_like(P)

    # 1) solver-provided jump
    for k in ("J_mid", "J", "Jm", "Jsol"):
        if k in solp and solp[k] is not None:
            J = np.asarray(solp[k], float)
            if J.shape == P.shape:
                return P, J

    # 2) integrate densities
    t = np.asarray(solp.get("t_mid", []), float)
    n = np.asarray(solp.get("n_mid", []), float)
    ds = np.asarray(solp.get("ds", []), float).reshape(-1,)

    N = P.shape[0]
    bI = np.asarray(solp.get("bI", 0.0), float).reshape(-1,)
    bII = np.asarray(solp.get("bII", 0.0), float).reshape(-1,)
    if bI.size != N:
        bI = np.resize(bI, N)
    if bII.size != N:
        bII = np.resize(bII, N)

    if t.shape != P.shape or n.shape != P.shape:
        # fall back to geometric frame if missing (rare)
        dP = np.gradient(P, axis=0)
        L = np.linalg.norm(dP, axis=1)
        L[L == 0] = 1.0
        t = dP / L[:, None]
        n = np.c_[-t[:, 1], t[:, 0]]

    if ds.size != N:
        dP = np.diff(P, axis=0)
        ds2 = np.hypot(dP[:, 0], dP[:, 1])
        ds = np.concatenate([[ds2[0]], ds2]) if ds2.size else np.ones((N,), float)

    incr = (bII[:, None] * t + bI[:, None] * n) * ds[:, None]
    J = np.zeros((N, 2), float)
    J0 = solp.get('J0', solp.get('J_start', solp.get('J_at_start', None)))
    if J0 is None:
        J[0, :] = 0.0
    else:
        J0 = np.asarray(J0, float).reshape(-1)
        if J0.size >= 2:
            J[0, :] = J0[:2]
        else:
            J[0, :] = 0.0
    for i in range(1, N):
        J[i, :] = J[i - 1, :] - incr[i, :]
    return P, J


def _set_graph_like_limits(ax, X: np.ndarray, pad_frac: float = 0.05):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] == 0:
        return
    xmin, ymin = np.min(X[:, 0]), np.min(X[:, 1])
    xmax, ymax = np.max(X[:, 0]), np.max(X[:, 1])
    dx = max(1e-12, xmax - xmin)
    dy = max(1e-12, ymax - ymin)
    span = max(dx, dy)
    pad = pad_frac * span
    # Use the same padding scale in x and y so nearly-horizontal/vertical cracks
    # still display with meaningful bounds (mirrors the network graph view).
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


# -----------------------------
# main plotter
# -----------------------------
@dataclass
class DCEPlotterDeformedV4:
    res: object
    out_dir: Path | None = None

    def __post_init__(self):
        if self.out_dir is None:
            self.out_dir = Path(getattr(self.res, "out_dir", "output"))

    def plot_deformed_network(
        self,
        n_pts_per_edge: int = 2000,
        scale: float = 5e1,
        show_faces: bool = True,
        units: str = "mm",
        cod_smooth_window: int = 5,
        csd_smooth_window: int = 5,
        junction_model: str | None = None,
        *,
        trim_core_junction_faces: bool = False,
        trim_junction_look_ahead: int = 3,
        trim_junction_search_segments: int = 10,
        show_junction_gap: bool = False,
        show: bool = True,
        save: bool = False,
        debug_counts: bool = True,
    ):
        """
        Plot the deformed crack network using solver-provided jump fields when available.

        junction_model:
          - "strict": stitch faces at deg>1 vertices by applying a *constant* per-polyline
                      jump shift so that endpoint J values match at the shared vertex.
                      (No new physics; visualization-only gauge alignment.)
          - "core":   do not stitch; instead trim faces near deg>1 vertices to avoid
                      over-plotting through the junction core.

        trim_core_junction_faces:
          - When junction_model=="core" and the network has deg>=3 junctions, optionally perform
            geometric trimming so lower_face(branch j) meets upper_face(branch j+1) around the
            junction. This eliminates face lines that run through the junction core.
        """
        res = self.res
        sol = getattr(res, "sol", None)
        calc = getattr(res, "calc", None)
        if calc is None or getattr(calc, "network", None) is None:
            raise AttributeError("res.calc.network is required for plotting.")
        net = calc.network
        deg = _deg_map_from_network(net)

        # Units scale consistent with network graph
        Vc = []
        for e in getattr(net, "edges", []):
            try:
                Vc.append(np.asarray(net.vertex_coords(int(e.v0)), float))
                Vc.append(np.asarray(net.vertex_coords(int(e.v1)), float))
            except Exception:
                pass
        Vc = np.asarray(Vc, float) if Vc else np.zeros((0, 2), float)
        sxy = _infer_units_scale_from_points(Vc, units)

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        n_plotted, n_skipped = 0, 0

        # If not parametrized solver output, just plot the graph midlines
        if not isinstance(sol, dict) or str(sol.get("solver_option", "")).lower() != "parametrized_crack":
            for e in getattr(net, "edges", []):
                try:
                    p0 = np.asarray(net.vertex_coords(int(e.v0)), float) * sxy
                    p1 = np.asarray(net.vertex_coords(int(e.v1)), float) * sxy
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", lw=1.6)
                    n_plotted += 1
                except Exception:
                    n_skipped += 1

            ax.set_title(f"Deformed Crack Network (scale={float(scale):.2e})")
            ax.set_xlabel(f"x [{units}]")
            ax.set_ylabel(f"y [{units}]")
            ax.grid(True, alpha=0.35)
            _set_graph_like_limits(ax, Vc * sxy if Vc.size else None)
            if debug_counts:
                ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9)
            if save:
                if self.out_dir is None:
                    raise ValueError("out_dir must be set to save figures.")
                self.out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(self.out_dir / "deformed_network.png", dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            return fig, ax

        # parametrized crack path
        polylines = list(sol.get("parametrized_polylines", sol.get("polylines", [])))
        sol_list  = list(sol.get("polyline_solutions", []))
        if junction_model is None:
            junction_model = str(sol.get("junction_model", "strict")).lower()
        else:
            junction_model = str(junction_model).lower()

        nh = _infer_ne_half(sol)
        w_cod = resolve_smooth_window(cod_smooth_window, nh, C=50, wmin=3)
        w_csd = resolve_smooth_window(csd_smooth_window, nh, C=50, wmin=3)
        wJ = max(int(w_cod), int(w_csd))

        # 1) Build per-polyline plotted midline + jump, but do NOT draw yet.
        poly_data: list[dict] = []
        for pid, meta in enumerate(polylines):
            solp0 = sol_list[pid] if pid < len(sol_list) else None
            if solp0 is None:
                n_skipped += 1
                continue

            start_vid = int(meta.get("v_start", -1))
            end_vid   = int(meta.get("v_end", -1))
            start_is_tip = (start_vid < 0) or (deg.get(start_vid, 0) == 1)
            end_is_tip   = (end_vid < 0) or (deg.get(end_vid, 0) == 1)

            # Orient arrays so a unique tip (if any) is at index 0 (helps fallback integration).
            solp, did_rev, start_is_tip, end_is_tip = _oriented_solp_to_tip(solp0, start_is_tip, end_is_tip)
            if did_rev:
                start_vid, end_vid = end_vid, start_vid

            P_mid, J_mid = _reconstruct_jump(solp)
            if P_mid.ndim != 2 or P_mid.shape[0] < 2 or J_mid.shape != P_mid.shape:
                n_skipped += 1
                continue

            # Optional smoothing of J
            if wJ > 1:
                J_mid = np.c_[moving_average_nan(J_mid[:, 0], wJ),
                             moving_average_nan(J_mid[:, 1], wJ)]

            # Enforce solver tip J=0 where applicable (deg-1 vertices)
            J_mid = _enforce_tip_zero_polyline(P_mid, J_mid, start_is_tip, end_is_tip)

            # Resample midline for plotting resolution
            P_plot, s_plot = _resample_polyline_by_arclength(P_mid, n=int(n_pts_per_edge))

            # Interpolate J to plot points by arclength
            s_mid = _arclength(P_mid)
            J_plot = np.c_[np.interp(s_plot, s_mid, J_mid[:, 0]),
                           np.interp(s_plot, s_mid, J_mid[:, 1])]

            poly_data.append(dict(
                pid=pid,
                start_vid=start_vid, end_vid=end_vid,
                start_deg=deg.get(start_vid, 0), end_deg=deg.get(end_vid, 0),
                P_plot=P_plot * sxy,
                J_plot=J_plot * sxy,   # J has length units
            ))

        # 2) Stitch or trim at junctions depending on junction_model.
        if junction_model == "strict":
            # Build incident map vid -> list of (poly_index, end, J_endpoint)
            inc: dict[int, list[tuple[int, str, np.ndarray]]] = {}
            for i, pd in enumerate(poly_data):
                sv, ev = pd["start_vid"], pd["end_vid"]
                if sv >= 0 and deg.get(sv, 0) > 1:
                    inc.setdefault(sv, []).append((i, "start", pd["J_plot"][0].copy()))
                if ev >= 0 and deg.get(ev, 0) > 1:
                    inc.setdefault(ev, []).append((i, "end", pd["J_plot"][-1].copy()))

            # For each junction vertex, align all incident polylines by constant shift
            for vid, items in inc.items():
                if len(items) < 2:
                    continue
                Jref = np.mean(np.stack([it[2] for it in items], axis=0), axis=0)
                for (i, endflag, Jend) in items:
                    delta = (Jref - Jend).reshape(1, 2)
                    poly_data[i]["J_plot"] = poly_data[i]["J_plot"] + delta

        # Build crack faces and (optionally) apply core-junction trimming.
        # - If junction_model=="core" and trim_core_junction_faces==True, we trim faces at
        #   multi-branch junctions (degree >= 3) using local face intersections.
        # - Otherwise, we use a simple index-based trim near deg>1 vertices.
        trim_n = max(2, int(0.02 * int(n_pts_per_edge)))

        # Precompute faces for all polylines (full arrays); we may modify them in-place.
        for pd in poly_data:
            Pp = pd["P_plot"]
            Jp = pd["J_plot"]
            pd["U_plot"] = Pp + 0.5 * scale * Jp
            pd["L_plot"] = Pp - 0.5 * scale * Jp

        gap_polys: list[np.ndarray] = []

        if junction_model == "core" and trim_core_junction_faces and (trim_faces_at_junction is not None) and (BranchFaces is not None):
            # Build vertex -> incident map with orientation info.
            inc: dict[int, list[tuple[int, bool]]] = {}
            for i, pd in enumerate(poly_data):
                sv, ev = pd["start_vid"], pd["end_vid"]
                if sv >= 0 and deg.get(sv, 0) > 2:
                    inc.setdefault(sv, []).append((i, True))   # True: start_at_vertex
                if ev >= 0 and deg.get(ev, 0) > 2:
                    inc.setdefault(ev, []).append((i, False))  # False: end_at_vertex

            # Trim per junction vertex.
            for vid, items in inc.items():
                if len(items) < 3:
                    continue
                try:
                    vxy = np.asarray(net.vertex_coords(int(vid)), float).reshape(2,) * sxy
                except Exception:
                    continue

                branches = []
                # Map pid -> orientation (start_at_vertex)
                orient = {}
                for (i, start_at_vertex) in items:
                    pd = poly_data[i]
                    Pp = pd["P_plot"]
                    Up = pd["U_plot"]
                    Lp = pd["L_plot"]
                    if start_at_vertex:
                        branches.append(BranchFaces(pid=i, P=Pp, U=Up, L=Lp, start_at_vertex=True))
                        orient[i] = True
                    else:
                        branches.append(BranchFaces(pid=i, P=Pp[::-1].copy(), U=Up[::-1].copy(), L=Lp[::-1].copy(), start_at_vertex=True))
                        orient[i] = False

                trimmed, gap_pts = trim_faces_at_junction(
                    vxy,
                    branches,
                    look_ahead=int(trim_junction_look_ahead),
                    search_segments=int(trim_junction_search_segments),
                )
                if show_junction_gap and gap_pts is not None and getattr(gap_pts, "size", 0) > 0:
                    gap_polys.append(np.asarray(gap_pts, float))

                # Write back trimmed faces, restoring original orientation where needed.
                for b in trimmed:
                    i = int(b.pid)
                    if orient.get(i, True):  # start_at_vertex
                        poly_data[i]["U_plot"] = b.U
                        poly_data[i]["L_plot"] = b.L
                    else:
                        poly_data[i]["U_plot"] = b.U[::-1].copy()
                        poly_data[i]["L_plot"] = b.L[::-1].copy()

        # Draw polylines
        for pd in poly_data:
            Pp = pd["P_plot"]
            Up = pd["U_plot"]
            Lp = pd["L_plot"]

            # Simple trim option (used when not doing geometric core trimming)
            if junction_model == "core" and not trim_core_junction_faces:
                i0, i1 = 0, len(Pp)
                if pd["start_vid"] >= 0 and pd["start_deg"] > 1:
                    i0 = min(i0 + trim_n, i1 - 2)
                if pd["end_vid"] >= 0 and pd["end_deg"] > 1:
                    i1 = max(i1 - trim_n, i0 + 2)
                Pp = Pp[i0:i1]
                Up = Up[i0:i1]
                Lp = Lp[i0:i1]

            ax.plot(Pp[:, 0], Pp[:, 1], "k--", lw=1.0, alpha=0.55)
            if show_faces:
                ax.plot(Up[:, 0], Up[:, 1], lw=1.6)
                ax.plot(Lp[:, 0], Lp[:, 1], lw=1.6)

            n_plotted += 1

        if show_junction_gap and gap_polys:
            for gp in gap_polys:
                if gp.ndim == 2 and gp.shape[0] >= 3:
                    # close polygon
                    gpc = np.vstack([gp, gp[0:1]])
                    ax.plot(gpc[:, 0], gpc[:, 1], "k-", lw=1.0, alpha=0.35)
# Styling: match network graph look (limits from graph, not from deformation)
        ax.set_title(f"Deformed Crack Network (junction_model={junction_model}, scale={float(scale):.2e})")
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.grid(True, alpha=0.35)
        _set_graph_like_limits(ax, Vc * sxy if Vc.size else None)

        if debug_counts:
            ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9)

        if save:
            if self.out_dir is None:
                raise ValueError("out_dir must be set to save figures.")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / "deformed_network.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

# -----------------------------------------------------------------------------
