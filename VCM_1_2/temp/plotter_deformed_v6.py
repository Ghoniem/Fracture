"""Deformed network plotting using reconstructed geometric jump (COD/CSD)."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Sequence, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .opts import (
    StressPlotOptsV4, ensure_dir, save_fig, apply_clip_percentiles,
    robust_vmin_vmax, symlog_levels, rot_from_tangent, network_extent
)
from .smooth import moving_average_nan, resolve_smooth_window
from ..Uprocessor.PK import PKProcessor


class DCEPlotterDeformedV4:
    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)

    def plot_deformed_network(
        self,
        n_theta: int = 4000,
        scale: float = 1e6,
        show_faces: bool = True,
        units: str = "mm",
        *,
        cod_smooth_window: int = 0,
        csd_smooth_window: int = 0,
        show: bool = True,
        save: bool = True,
        debug_counts: bool = False,
        # plotting-only jump stitching across polyline kinks
        stitch_polyline_kinks: bool = True,
        kink_angle_deg: float = 12.0,
    ):
        """Plot deformed crack network from COD/CSD-derived jump only."""
        pk = PKProcessor(self.res)

        ne_half_eff = int(getattr(self.res, "sol", {}).get("ne_half", 0) or 0)
        if ne_half_eff <= 0:
            ne_half_eff = 10
        cod_win = resolve_smooth_window(cod_smooth_window, ne_half_eff)
        csd_win = resolve_smooth_window(csd_smooth_window, ne_half_eff)

        V = self.calc.network.vertices
        E = self.calc.network.edges
        deg = pk.degree_map()

        sol = getattr(self.res, "sol", {}) if isinstance(getattr(self.res, "sol", {}), dict) else {}
        constraints_meta = sol.get("constraints", {}) if isinstance(sol.get("constraints", {}), dict) else {}
        junction_model = str(constraints_meta.get("junction_model", "strict")).lower().strip()
        if junction_model not in ("strict", "core", "soft"):
            junction_model = "strict"

        Jv_map = {}
        if junction_model == "strict" and getattr(self.res, "is_parametrized", lambda: False)():
            Jv_map = pk.junction_J_map()

        sxy = 1e3 if units.lower() == "mm" else 1.0
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Deformed Crack Network (junction_model={junction_model}, scale={scale:.0e})")

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

        def _maybe_swap_faces(U: np.ndarray, L: np.ndarray, n_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            m = int(min(U.shape[0], L.shape[0]))
            if m < 3:
                return U, L
            i0 = m // 5
            i1 = m - m // 5
            d = np.mean(U[i0:i1, :] - L[i0:i1, :], axis=0)
            if float(np.dot(d, n_hat)) < 0.0:
                return L, U
            return U, L

        def _attach_junction_endpoints(U: np.ndarray, L: np.ndarray, p: np.ndarray, Jv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            p_plus  = p + 0.5 * Jv * float(scale)
            p_minus = p - 0.5 * Jv * float(scale)

            mid = 0.5 * (U + L)
            d0 = float(np.hypot(*(mid[0, :] - p)))
            d1 = float(np.hypot(*(mid[-1, :] - p)))
            end_is_start = d0 <= d1
            U_end = U[0, :] if end_is_start else U[-1, :]
            L_end = L[0, :] if end_is_start else L[-1, :]

            c1 = float(np.hypot(*(U_end - p_plus))) + float(np.hypot(*(L_end - p_minus)))
            c2 = float(np.hypot(*(U_end - p_minus))) + float(np.hypot(*(L_end - p_plus)))
            pU, pL = (p_plus, p_minus) if c1 <= c2 else (p_minus, p_plus)

            if end_is_start:
                U = np.vstack([pU.reshape(1,2), U])
                L = np.vstack([pL.reshape(1,2), L])
            else:
                U = np.vstack([U, pU.reshape(1,2)])
                L = np.vstack([L, pL.reshape(1,2)])
            return U, L

        def _enforce_tip_zero(P: np.ndarray, J: np.ndarray, start_vid: int, end_vid: int) -> np.ndarray:
            """Remove spurious constant/linear offset in the reconstructed jump so that J=0 at degree-1 vertices.

            This is intentionally *only* applied at degree-1 vertices (crack tips). For junction vertices
            (degree >= 2), nonzero COD/CSD is expected and must be preserved.
            """
            tip_start = int(deg.get(int(start_vid), 0)) == 1
            tip_end   = int(deg.get(int(end_vid),   0)) == 1

            if not (tip_start or tip_end):
                return J

            # Parameter s along the edge (used only if both ends are tips)
            if tip_start and tip_end and P.shape[0] >= 2:
                ds = np.sqrt(np.sum(np.diff(P, axis=0)**2, axis=1))
                s = np.concatenate([[0.0], np.cumsum(ds)])
                L = float(s[-1])
                if L > 0.0:
                    s = s / L
                    J0 = J[0, :].copy()
                    J1 = J[-1, :].copy()
                    J = J - ((1.0 - s)[:, None] * J0[None, :] + s[:, None] * J1[None, :])
                    return J

            # Otherwise, enforce at the single tip endpoint
            if tip_start:
                return J - J[0, :][None, :]
            if tip_end:
                return J - J[-1, :][None, :]
            return J

        def _stitch_jump_at_kinks(P: np.ndarray, J: np.ndarray) -> np.ndarray:
            """Stitch piecewise jump reconstructions across polyline kinks.

            In polyline cracks, some reconstruction pipelines treat each straight segment
            independently on a local [0,1] interval. This can introduce *segment-wise constant*
            offsets in the jump J (COD/CSD), which then shows up as gaps between the two
            faces at internal degree-2 kinks.

            Here we detect sharp tangent direction changes and apply a cumulative constant
            shift to subsequent samples so that J is continuous across those kink indices.

            This is a plotting-only correction; it does not modify solver outputs.
            """
            if P.shape[0] < 4:
                return J

            # segment tangents
            dP = np.diff(P, axis=0)
            seg_len = np.sqrt(np.sum(dP * dP, axis=1))
            good = seg_len > (1e-14 * max(1.0, float(np.nanmax(seg_len))))
            if np.count_nonzero(good) < 3:
                return J
            t = np.zeros_like(dP)
            t[good, :] = dP[good, :] / seg_len[good, None]

            # angle change between successive tangents (at interior vertices)
            cosang = np.sum(t[:-1, :] * t[1:, :], axis=1)
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))  # length N-2, corresponds to vertex indices 1..N-2

            thresh = float(kink_angle_deg)
            kink_vertices = np.where(ang >= thresh)[0] + 1  # map to P index
            if kink_vertices.size == 0:
                return J

            J2 = J.copy()
            shift = np.zeros(2, float)
            last_k = 0
            for k in kink_vertices:
                # Apply accumulated shift up to kink
                if k > last_k:
                    J2[last_k:k, :] -= shift[None, :]

                # Compute mismatch at kink and update shift for the remainder
                # Use left value (k-1) as reference.
                mismatch = (J2[k, :] - shift) - (J2[k - 1, :] - shift)
                # If mismatch is large relative to typical J, treat as segment-wise offset.
                # Always apply; the kink detection already gates this.
                shift += mismatch
                last_k = k

            # Apply final accumulated shift
            J2[last_k:, :] -= shift[None, :]
            return J2


        n_plotted = 0
        n_skipped = 0
        vmap = {int(v.id): v for v in V}

        for edge_idx, edge in enumerate(E):
            v0 = vmap[int(edge.v0)]
            v1 = vmap[int(edge.v1)]
            p0 = np.array([float(v0.x), float(v0.y)], float)
            p1 = np.array([float(v1.x), float(v1.y)], float)

            v0_id = int(edge.v0)
            v1_id = int(edge.v1)

            try:
                P, J = pk.jump_along_edge(
edge_index=int(edge_idx),
                    n_theta=int(max(400, n_theta)),
                    enforce_global_tip_zero=False,
                    cod_window_panels=int(cod_win),
                    csd_window_panels=int(csd_win),
                )
            except Exception:
                n_skipped += 1
                continue

            if P is None or J is None:
                n_skipped += 1
                continue

            P = np.asarray(P, float)
            J = np.asarray(J, float)
            if P.ndim != 2 or J.ndim != 2 or P.shape[0] < 2 or P.shape != J.shape:
                n_skipped += 1
                continue

            # If the crack is a polyline, the reconstructed jump may be piecewise with
            # segment-wise constant offsets. Stitch those offsets so the plotted faces
            # follow the midline continuously through internal degree-2 kinks.
            if bool(stitch_polyline_kinks):
                try:
                    J = _stitch_jump_at_kinks(P, J)
                except Exception:
                    pass

            # Map curve endpoints to network vertices (P may be oriented v0->v1 or v1->v0)
            d00 = float(np.hypot(*(P[0, :] - p0)))
            d01 = float(np.hypot(*(P[0, :] - p1)))
            start_vid = v0_id if d00 <= d01 else v1_id
            end_vid   = v1_id if start_vid == v0_id else v0_id

            # Enforce J=0 at *all* degree-1 vertices (tips), independent of junction_model.
            # Do NOT touch degree>=2 vertices (junctions).
            J = _enforce_tip_zero(P, J, start_vid=start_vid, end_vid=end_vid)

            U = P + 0.5 * J * float(scale)
            Lw = P - 0.5 * J * float(scale)

            t = p1 - p0
            Lt = float(np.hypot(t[0], t[1]))
            if Lt > 0:
                t_hat = t / Lt
                n_hat = np.array([-t_hat[1], t_hat[0]], float)
                U, Lw = _maybe_swap_faces(U, Lw, n_hat)

            if junction_model == "strict":
                for vid, p in ((v0_id, p0), (v1_id, p1)):
                    if int(deg.get(int(vid), 0)) >= 2:
                        Jv = Jv_map.get(int(vid), None)
                        if Jv is not None:
                            U, Lw = _attach_junction_endpoints(U, Lw, p, np.asarray(Jv, float).reshape(2,))

            col = colors[int(edge_idx) % len(colors)]
            if show_faces:
                ax.plot(U[:, 0] * sxy, U[:, 1] * sxy, lw=1.5, color=col, alpha=0.95)
                ax.plot(Lw[:, 0] * sxy, Lw[:, 1] * sxy, lw=1.5, color=col, alpha=0.95)
            else:
                ax.plot(P[:, 0] * sxy, P[:, 1] * sxy, lw=2.0, color=col, alpha=0.95)

            ax.plot([p0[0]*sxy, p1[0]*sxy], [p0[1]*sxy, p1[1]*sxy], ls=":", lw=1.0, color="k", alpha=0.35)
            n_plotted += 1

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

        if debug_counts:
            ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9)

        if save:
            try:
                save_fig(fig, self.out_dir, "network_deformed", dpi=150)
            except Exception:
                pass
        if show:
            plt.show()
        return fig
