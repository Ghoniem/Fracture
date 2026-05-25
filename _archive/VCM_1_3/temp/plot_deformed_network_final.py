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
    padx = pad_frac * dx
    pady = pad_frac * dy
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
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

        # Fallback: if not parametrized solver output, just plot the graph midlines
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
            ax.grid(True, alpha=0.25)
            _set_graph_like_limits(ax, Vc * sxy)

            if debug_counts:
                ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9)
            if save:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(self.out_dir / "deformed_network.png", dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            return fig, ax

        polylines = list(sol.get("parametrized_polylines", []))
        sol_list  = list(sol.get("polyline_solutions", []))
        if junction_model is None:
            junction_model = str(sol.get("junction_model", "strict")).lower()
        else:
            junction_model = str(junction_model).lower()

        # Collect points for global limits (graph-like)
        all_pts = [Vc * sxy] if Vc.size else []

        nh = _infer_ne_half(sol)
        w_cod = resolve_smooth_window(cod_smooth_window, nh, C=50, wmin=3)
        w_csd = resolve_smooth_window(csd_smooth_window, nh, C=50, wmin=3)
        wJ = max(int(w_cod), int(w_csd))

        for pid, meta in enumerate(polylines):
            solp0 = sol_list[pid] if pid < len(sol_list) else None
            if solp0 is None:
                n_skipped += 1
                continue

            # endpoint tip flags from graph degrees
            start_vid = int(meta.get("v_start", -1))
            end_vid   = int(meta.get("v_end", -1))
            start_is_tip = (start_vid < 0) or (deg.get(start_vid, 0) == 1)
            end_is_tip   = (end_vid < 0) or (deg.get(end_vid, 0) == 1)

            # Orient arrays so a unique tip (if any) is at index 0.
            solp, did_rev, start_is_tip, end_is_tip = _oriented_solp_to_tip(solp0, start_is_tip, end_is_tip)

            # Midline and jump along midline
            P_mid, J_mid = _reconstruct_jump(solp)
            if P_mid.ndim != 2 or P_mid.shape[0] < 2 or J_mid.shape != P_mid.shape:
                n_skipped += 1
                continue

            # Smooth J only (optional)
            if wJ > 1:
                J_mid = np.c_[moving_average_nan(J_mid[:, 0], wJ),
                             moving_average_nan(J_mid[:, 1], wJ)]

            # Enforce tip J=0 where applicable (consistent with your solver constraints at deg-1 vertices)
            J_mid = _enforce_tip_zero_polyline(P_mid, J_mid, start_is_tip, end_is_tip)

            # Resample midline for plotting resolution
            P_plot, _ = _resample_polyline_by_arclength(P_mid, n=int(n_pts_per_edge))

            # Interpolate J to plot points by arclength
            ds = np.hypot(np.diff(P_mid[:, 0]), np.diff(P_mid[:, 1]))
            s_mid = np.concatenate([[0.0], np.cumsum(ds)])
            ds2 = np.hypot(np.diff(P_plot[:, 0]), np.diff(P_plot[:, 1]))
            s_plot = np.concatenate([[0.0], np.cumsum(ds2)])
            if s_mid[-1] > 0:
                Jx = np.interp(s_plot, s_mid, J_mid[:, 0])
                Jy = np.interp(s_plot, s_mid, J_mid[:, 1])
                J_plot = np.c_[Jx, Jy]
            else:
                J_plot = np.zeros_like(P_plot)

            # Apply units
            P_plot = P_plot * sxy
            J_plot = J_plot * sxy

            # Deformed faces and midline
            U = P_plot + 0.5 * scale * J_plot
            L = P_plot - 0.5 * scale * J_plot

            # plot
            ax.plot(P_plot[:, 0], P_plot[:, 1], "k--", lw=1.0, alpha=0.55)
            if show_faces:
                ax.plot(U[:, 0], U[:, 1], lw=1.6)
                ax.plot(L[:, 0], L[:, 1], lw=1.6)

            all_pts.append(P_plot)
            all_pts.append(U)
            all_pts.append(L)

            n_plotted += 1

        # Finish styling
        ax.set_title(f"Deformed Crack Network (junction_model={junction_model}, scale={float(scale):.2e})")
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.grid(True, alpha=0.25)

        # Match the network-graph axes/limits (do not auto-scale to deformed faces)
        if Vc is not None and np.asarray(Vc).size:
            _set_graph_like_limits(ax, np.asarray(Vc, float) * sxy)
        elif all_pts:
            _set_graph_like_limits(ax, np.vstack(all_pts))

        if debug_counts:
            ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9)

        if save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / "deformed_network.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax


# -----------------------------------------------------------------------------
