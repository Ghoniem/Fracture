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
        stitch_polyline_kinks: bool = True,
        kink_angle_deg: float = 12.0,
        show: bool = True,
        save: bool = True,
        debug_counts: bool = False,
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
        def _enforce_tip_zero(P, J, start_vid: int, end_vid: int):
            """Enforce J=0 at degree-1 vertices (tips) only.
            Uses a gauge correction (constant shift, plus optional affine drift if both ends are tips)
            to avoid introducing discontinuities at interior points/junctions.
            """
            J = np.asarray(J, float)
            n = J.shape[0]
            if n < 2:
                return J

            start_is_tip = (int(deg.get(int(start_vid), 0)) == 1)
            end_is_tip   = (int(deg.get(int(end_vid), 0)) == 1)

            if start_is_tip:
                J = J - J[0:1, :]

            if end_is_tip:
                if start_is_tip:
                    # remove affine drift so J(0)=0 and J(1)=0 simultaneously
                    s = np.linspace(0.0, 1.0, n)[:, None]
                    J = J - s * J[-1:, :]
                else:
                    J = J - J[-1:, :]

            return J


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

        def _stitch_faces_through_kinks(P: np.ndarray, U: np.ndarray, L: np.ndarray,
                                        *, kink_angle_deg: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
            """Stitch plotted faces across internal polyline kinks / segment boundaries.

            Primary failure mode for polyline edges is *piecewise reconstruction*: each straight
            segment can carry an independent constant-gauge offset in the reconstructed jump J.
            When forming faces as U,L = P ± 0.5 J, this appears as visible breaks at internal
            degree-2 kinks.

            Robust strategy (plotting-only):
              1) If the arrays contain NaN separators (common when concatenating per-segment
                 samples), split into segments, then *translate* downstream segments so that
                 both faces are C0-continuous at segment boundaries.
              2) Optionally, apply an angle-based pass on the fully concatenated arrays to
                 catch remaining kinks not separated by NaNs.

            This does NOT modify solver data; it only affects visualization geometry.
            """
            P = np.asarray(P, float)
            U = np.asarray(U, float).copy()
            L = np.asarray(L, float).copy()

            if P.ndim != 2 or U.ndim != 2 or L.ndim != 2 or P.shape[1] != 2:
                return U, L
            if P.shape[0] < 2:
                return U, L

            # --- (1) Split by NaN rows and stitch by translation, then concatenate.
            bad = np.any(~np.isfinite(P), axis=1) | np.any(~np.isfinite(U), axis=1) | np.any(~np.isfinite(L), axis=1)
            if np.any(bad):
                # Build contiguous valid runs
                idx = np.arange(P.shape[0])
                good = ~bad
                runs = []
                i0 = None
                for i, ok in enumerate(good):
                    if ok and i0 is None:
                        i0 = i
                    if (not ok or i == len(good)-1) and i0 is not None:
                        i1 = i if not ok else i+1
                        if i1 - i0 >= 2:
                            runs.append((i0, i1))
                        i0 = None

                if len(runs) >= 1:
                    Psegs, Usegs, Lsegs = [], [], []
                    for (a,b) in runs:
                        Psegs.append(P[a:b].copy())
                        Usegs.append(U[a:b].copy())
                        Lsegs.append(L[a:b].copy())

                    # Translate downstream segments so BOTH faces meet at boundaries.
                    for k in range(1, len(Psegs)):
                        U_prev, L_prev = Usegs[k-1], Lsegs[k-1]
                        U_cur,  L_cur  = Usegs[k],   Lsegs[k]

                        du = U_prev[-1] - U_cur[0]
                        dl = L_prev[-1] - L_cur[0]
                        shift = 0.5 * (du + dl)  # common translation keeps opening (U-L) intact
                        Usegs[k] += shift
                        Lsegs[k] += shift
                        Psegs[k] += shift  # keep midline centered with faces

                    P = np.vstack(Psegs)
                    U = np.vstack(Usegs)
                    L = np.vstack(Lsegs)

            # --- (2) Angle-based kink stitching on the concatenated arrays (fallback).
            n = int(P.shape[0])
            if n < 3:
                return U, L

            dP = np.diff(P, axis=0)
            ds = np.linalg.norm(dP, axis=1)
            # avoid zero-length segments (duplicate points)
            eps = 1e-15
            good = ds > eps
            if np.count_nonzero(good) < 2:
                return U, L

            t = np.zeros_like(dP)
            t[good] = dP[good] / ds[good, None]

            # turn angle at interior nodes i: between segment i-1 and i
            ang = np.zeros(n, float)
            for i in range(1, n-1):
                if not good[i-1] or not good[i]:
                    continue
                c = float(np.clip(np.dot(t[i-1], t[i]), -1.0, 1.0))
                ang[i] = math.degrees(math.acos(c))

            kink_idx = np.where(ang >= float(kink_angle_deg))[0]
            if kink_idx.size == 0:
                return U, L

            # Translate downstream portions so both faces are continuous across each kink.
            for k in kink_idx:
                if k <= 0 or k >= n:
                    continue
                du = U[k-1] - U[k]
                dl = L[k-1] - L[k]
                shift = 0.5 * (du + dl)
                U[k:] += shift
                L[k:] += shift
                P[k:] += shift

            return U, L

        
        n_plotted = 0
        n_skipped = 0
        vmap = {int(v.id): v for v in V}

        # ------------------------------------------------------------------
        # PASS 1: reconstruct (P, J) for every edge, but do not plot yet.
        # This allows a global gauge-stitch across internal degree-2 vertices,
        # which is the only reliable way to remove visual breaks at polyline kinks
        # when the polyline is represented as multiple graph edges.
        # ------------------------------------------------------------------
        edge_data = []
        for edge_idx, edge in enumerate(E):
            try:
                v0 = vmap[int(edge.v0)]
                v1 = vmap[int(edge.v1)]
            except Exception:
                n_skipped += 1
                continue

            try:
                P, J, extra = pk.jump_along_edge(
                    edge_index=edge_idx,
                    n_theta=int(n_theta),
                    cod_smooth_window=int(cod_win),
                    csd_smooth_window=int(csd_win),
                    units=str(units),
                )
            except Exception:
                n_skipped += 1
                continue

            # jump_along_edge may return list-of-segments; concatenate.
            if isinstance(P, (list, tuple)) and len(P) > 0:
                try:
                    P = np.vstack([np.asarray(pp, float) for pp in P if pp is not None and len(pp) > 0])
                except Exception:
                    pass
            if isinstance(J, (list, tuple)) and len(J) > 0:
                try:
                    J = np.vstack([np.asarray(jj, float) for jj in J if jj is not None and len(jj) > 0])
                except Exception:
                    pass

            P = np.asarray(P, float)
            J = np.asarray(J, float)
            if P.ndim != 2 or J.ndim != 2 or P.shape[0] < 2 or P.shape != J.shape:
                n_skipped += 1
                continue

            edge_data.append(
                dict(
                    edge_idx=int(edge_idx),
                    v0=int(v0.id),
                    v1=int(v1.id),
                    P=P,
                    J=J,
                    color=colors[edge_idx % len(colors)],
                )
            )

        # ------------------------------------------------------------------
        # PASS 2: gauge-stitch across degree-2 vertices (internal polyline kinks)
        # so that J is consistent across edges meeting at those vertices.
        # This removes segment-wise constant offsets that otherwise create
        # visible breaks in U/L faces.
        # ------------------------------------------------------------------
        if stitch_polyline_kinks and len(edge_data) > 1:

            # Build vertex -> incident edge-ends map
            inc = {}
            for k, ed in enumerate(edge_data):
                inc.setdefault(ed["v0"], []).append((k, "start"))
                inc.setdefault(ed["v1"], []).append((k, "end"))

            # Traverse connected components composed only of deg-2 vertices.
            visited_edges = set()
            from collections import deque

            def J_at(ed, end):
                return ed["J"][0] if end == "start" else ed["J"][-1]

            def shift_edge(ed, delta):
                # delta is 2-vector
                ed["J"] = ed["J"] - np.asarray(delta, float)[None, :]

            for root_ei in range(len(edge_data)):
                if root_ei in visited_edges:
                    continue
                # Start BFS/DFS from this edge; we only propagate constraints through deg-2 vertices.
                dq = deque([root_ei])
                visited_edges.add(root_ei)

                while dq:
                    ei = dq.popleft()
                    ed = edge_data[ei]
                    for vid, end in ((ed["v0"], "start"), (ed["v1"], "end")):
                        if int(deg.get(int(vid), 0)) != 2:
                            continue  # do not stitch through junctions/tips
                        for (ej, endj) in inc.get(int(vid), []):
                            if ej == ei:
                                continue
                            # Compute delta so that edge ej matches edge ei at this shared deg-2 vertex.
                            delta = J_at(edge_data[ej], endj) - J_at(ed, end)
                            # Apply constant shift to entire neighbor edge ej
                            shift_edge(edge_data[ej], delta)
                            if ej not in visited_edges:
                                visited_edges.add(ej)
                                dq.append(ej)

        # ------------------------------------------------------------------
        # PASS 3: plot edges (now gauge-consistent at deg-2 kinks)
        # ------------------------------------------------------------------
        for ed in edge_data:
            P = ed["P"]
            J = ed["J"]
            v0_id = int(ed["v0"])
            v1_id = int(ed["v1"])

            # Determine which vertex corresponds to the first/last samples (orientation)
            start_vid = v0_id
            end_vid   = v1_id

            # Enforce J=0 at degree-1 vertices (tips) only.
            J = _enforce_tip_zero(P, J, start_vid=start_vid, end_vid=end_vid)

            # Optionally attach strict junction end values (rare; only for strict+parametrized)
            if junction_model == "strict" and Jv_map:
                # If a vertex has a prescribed junction jump, attach/overwrite endpoint
                if int(start_vid) in Jv_map:
                    J[0, :] = np.asarray(Jv_map[int(start_vid)], float)
                if int(end_vid) in Jv_map:
                    J[-1, :] = np.asarray(Jv_map[int(end_vid)], float)

            # Build faces
            U = P + 0.5 * J * float(scale)
            Lw = P - 0.5 * J * float(scale)

            # Handle NaN separators by splitting and stitching faces inside each edge.
            good = np.isfinite(U).all(axis=1) & np.isfinite(Lw).all(axis=1)
            if not bool(np.all(good)):
                runs = _split_nan_runs(good)
                U_runs, L_runs, P_runs = [], [], []
                for (a, b) in runs:
                    U_runs.append(U[a:b].copy())
                    L_runs.append(Lw[a:b].copy())
                    P_runs.append(P[a:b].copy())
                # stitch between runs by translation so faces connect
                if len(U_runs) > 1:
                    for r in range(1, len(U_runs)):
                        du = U_runs[r-1][-1] - U_runs[r][0]
                        dl = L_runs[r-1][-1] - L_runs[r][0]
                        shift = 0.5 * (du + dl)
                        U_runs[r] += shift
                        L_runs[r] += shift
                        P_runs[r] += shift
                U = np.vstack(U_runs)
                Lw = np.vstack(L_runs)
                P = np.vstack(P_runs)

            # Final within-edge kink stitching (geometric), applied on the already-concatenated arrays.
            if stitch_polyline_kinks:
                U, Lw = _stitch_faces_at_boundaries(P, U, Lw, kink_angle_deg=float(kink_angle_deg))

            # Ensure consistent face labeling (optional)
            if show_faces:
                ax.plot(U[:, 0] * sxy, U[:, 1] * sxy, lw=1.6, color=ed["color"])
                ax.plot(Lw[:, 0] * sxy, Lw[:, 1] * sxy, lw=1.6, color=ed["color"])
                ax.plot(P[:, 0] * sxy, P[:, 1] * sxy, lw=0.9, color="k", alpha=0.35, ls=":")
            else:
                ax.plot(P[:, 0] * sxy, P[:, 1] * sxy, lw=1.6, color=ed["color"])

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
