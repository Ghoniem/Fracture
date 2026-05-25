"""
Deformed network plotter (v13)

Key change vs v11/v12:
- For solver_option='parametrized_crack', plot by *polyline chains* (path_edge_indices),
  concatenating all segments into one continuous array before forming faces.
  This removes visual breaks at internal degree-2 kinks that arise when plotting each
  graph edge independently (each edge has its own constant jump gauge).
- Uses reconstruct_cod_csd_parametrized_smoothed() for panel-aware COD/CSD recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

# Local helper (provided in this package by the user)
from .reconstruct import reconstruct_cod_csd_parametrized_smoothed


def _deg_map_from_network(net) -> dict[int, int]:
    deg = {int(v.id): 0 for v in getattr(net, "vertices", [])}
    for e in getattr(net, "edges", []):
        v0 = int(getattr(e, "v0"))
        v1 = int(getattr(e, "v1"))
        deg[v0] = deg.get(v0, 0) + 1
        deg[v1] = deg.get(v1, 0) + 1
    return deg


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
        # cumulative arclength parameter alpha in [0,1]
        ds = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        Ltot = float(s[-1]) if s.size else 0.0
        if Ltot > 0.0:
            alpha = (s / Ltot).reshape(-1, 1)
            J_end = J[-1].reshape(1, 2)
            J = J - alpha * J_end

    return J


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
        cod_smooth_window: int = 0,
        csd_smooth_window: int = 0,
        stitch_polyline_kinks: bool = True,   # kept for signature compatibility; v13 stitches by construction
        kink_angle_deg: float = 12.0,         # kept for signature compatibility
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

# color cycle (robust)
            prop = plt.rcParams.get("axes.prop_cycle", None)
            if prop is not None:
                color_list = prop.by_key().get("color", [])
            else:
                color_list = []

            if not color_list:
                color_list = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

            for pid, meta in enumerate(polylines):
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

                for k, edge_index in enumerate(path_edges):
                    try:
                        # Panel-aware recovery in local frame for this *segment/edge*
                        x_edge, COD, CSD = reconstruct_cod_csd_parametrized_smoothed(
                            res,
                            edge_index=edge_index,
                            n_pts=int(n_theta),
                            enforce_global_tip_zero=False,  # we do endpoint enforcement on the concatenated polyline
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
                    # protect against inconsistent segL
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

                if not P_all:
                    continue

                P = np.vstack(P_all)
                J = np.vstack(J_all)

                # Enforce tip closure on the *polyline* endpoints (deg-1)
                start_vid = int(path_vids[0])
                end_vid   = int(path_vids[-1])
                J = _enforce_tip_zero_polyline(
                    P, J,
                    start_is_tip=(deg.get(start_vid, 0) == 1),
                    end_is_tip=(deg.get(end_vid, 0) == 1),
                )

                # Build faces and plot as ONE continuous polyline (per face)
                U = P + 0.5 * float(scale) * J
                Lw = P - 0.5 * float(scale) * J

                col = color_list[pid % len(color_list)]
                ax.plot(P[:, 0], P[:, 1], linestyle=":", linewidth=1.0, color="0.5", alpha=0.8)
                if show_faces:
                    ax.plot(U[:, 0], U[:, 1], linewidth=1.6, color=col)
                    ax.plot(Lw[:, 0], Lw[:, 1], linewidth=1.6, color=col, alpha=0.85)

                n_plotted += 1

            ax.set_title(f"Deformed Crack Network (parametrized_crack, scale={float(scale):.2e})")
        else:
            # ------------------------------------------------------------
            # Fallback: defer to the older per-edge plotting approach
            # ------------------------------------------------------------
            ax.set_title(f"Deformed Crack Network (scale={float(scale):.2e})")
            # Minimal fallback: plot the undeformed network midlines
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
