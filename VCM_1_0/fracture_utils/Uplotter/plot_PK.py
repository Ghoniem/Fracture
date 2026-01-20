"""plotter_PK.py

Add-on plots for Burgers-content vectors and PK-style force vectors.
This file is meant to live alongside plotter_core.py without changing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .opts import ensure_dir, save_fig
from ..Uprocessor.PK import *  


@dataclass
class VecPlotStyle:
    # Automatic normalization: vectors are scaled so the largest magnitude
    # is drawn with length = user_scale * max_len_factor * network_extent.
    #
    # Typical use: pass user_scale=1.0 and adjust max_len_factor (or user_scale)
    # to make arrows larger/smaller.
    max_len_factor: float = 0.18
    tip_color: str = "k"
    junction_color: str = "C3"
    sum_color: str = "k"
    width: float = 0.004
    headwidth: float = 3.0
    headlength: float = 4.0
    alpha: float = 0.9


class DCEPlotterPK:
    """Additional plots related to B-content and PK-style forces."""

    def __init__(self, results, out_dir: Optional[Path] = None):
        self.res = results
        self.calc = results.calc
        self.out_dir = ensure_dir(out_dir)
        self.pk = PKProcessor(results)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _vertex_positions(self) -> Dict[int, np.ndarray]:
        V = self.calc.network.vertices
        return {int(v.id): np.array([float(v.x), float(v.y)], float) for v in V}

    def _edge_index_for_pid(self, pid: int) -> Optional[int]:
        sol = getattr(self.res, "sol", None)
        if not isinstance(sol, dict):
            return None
        e2p = sol.get("parametrized_edge_to_polyline", {})
        for ei, _pid in e2p.items():
            if int(_pid) == int(pid):
                return int(ei)
        return None

    def _pid_for_edge_index(self, edge_index: int) -> Optional[int]:
        sol = getattr(self.res, "sol", None)
        if not isinstance(sol, dict):
            return None
        e2p = sol.get("parametrized_edge_to_polyline", {})
        pid = e2p.get(int(edge_index), None)
        return int(pid) if pid is not None else None

    def _branch_content_for_edge(self, edge_index: int) -> Optional[np.ndarray]:
        """Return DeltaJ_branch for an edge (parametrized solver), else None."""
        pid = self._pid_for_edge_index(edge_index)
        if pid is None:
            return None
        try:
            return np.asarray(self.pk.deltaJ_branch(int(pid)), float).reshape(2,)
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # (2) Burgers-content vectors (kinematic, not stress-based)
    # ---------------------------------------------------------------------

    def plot_b_content_vectors(
        self,
        *,
        style: VecPlotStyle = VecPlotStyle(),
        user_scale: float = 1.0,
        units: str = "mm",
        show_tips: bool = True,
        show_junctions: bool = True,
        show_sum_at_junction: bool = True,
        annotate: bool = True,
    ):
        """Plot integrated-density ("B-content") vectors.

        Interpretation
        --------------
        We plot the kinematic content
            DeltaJ_branch = ∫ (bII t + bI n) ds
        per incident branch and optionally their sum at each junction.

        This is solver-neutral and does not require stress.
        """
        V = self.calc.network.vertices
        E = self.calc.network.edges
        deg = self.pk.degree_map()
        vpos = self._vertex_positions()

        sxy = 1e3 if units.lower() == "mm" else 1.0

        # --- automatic normalization ---
        # Draw the largest vector with length = user_scale * max_len_factor * network_extent.
        xs = np.array([p[0] for p in vpos.values()], float)
        ys = np.array([p[1] for p in vpos.values()], float)
        # ext = float(max(np.ptp(xs) if xs.size else 1.0, np.ptp(ys) if ys.size else 1.0, 1e-12))
        ext = max(np.ptp(xs), np.ptp(ys))

        max_arrow_len_plot = float(user_scale) * float(style.max_len_factor) * ext * float(sxy)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("B-content vectors (integrated density)")

        # draw network midlines (dotted)
        for e in E:
            p0 = vpos[int(e.v0)]
            p1 = vpos[int(e.v1)]
            ax.plot([p0[0] * sxy, p1[0] * sxy], [p0[1] * sxy, p1[1] * sxy], ls=":", lw=1.0, color="k", alpha=0.4)

        # Junction incidence (parametrized), if available
        v_inc: Dict[int, List[Tuple[int, str]]] = {}
        try:
            jd = self.pk.junction_incidence_parametrized()
            v_inc = jd.v_inc
        except Exception:
            v_inc = {}

        # Collect raw vectors to determine vmax (meters)
        raw_tip = []  # (vid, pos_m, vec_m)
        raw_junc = []  # (pos_m, vec_m, color)
        raw_sum = []   # (pos_m, vec_m)

        if show_tips:
            for ei, e in enumerate(E):
                B = self._branch_content_for_edge(int(ei))
                if B is None:
                    continue
                for vid in (int(e.v0), int(e.v1)):
                    if deg.get(int(vid), 0) != 1:
                        continue
                    raw_tip.append((int(vid), vpos[int(vid)], np.asarray(B, float).reshape(2,)))

        if show_junctions and v_inc:
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C4", "C5"]
            for vid, items in v_inc.items():
                if deg.get(int(vid), 0) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                Bsum = np.zeros(2, float)
                for i, (pid, _which) in enumerate(items):
                    ei = self._edge_index_for_pid(int(pid))
                    if ei is None:
                        continue
                    B = self._branch_content_for_edge(int(ei))
                    if B is None:
                        continue
                    Bsum += B
                    raw_junc.append((p, np.asarray(B, float).reshape(2,), colors[i % len(colors)]))
                if show_sum_at_junction:
                    raw_sum.append((p, Bsum))

        mags = [float(np.hypot(v[0], v[1])) for _, _, v in raw_tip] + \
               [float(np.hypot(v[0], v[1])) for _, v, _ in raw_junc] + \
               [float(np.hypot(v[0], v[1])) for _, v in raw_sum]
        vmax = float(max(mags)) if mags else 0.0
        scale_fac = (max_arrow_len_plot / vmax) if (vmax > 0.0 and max_arrow_len_plot > 0.0) else 0.0

        # Tips: one resultant vector at each degree-1 vertex
        for vid, p_m, B_m in raw_tip:
            p = np.asarray(p_m, float)
            vplot = np.asarray(B_m, float) * float(scale_fac) * float(sxy)
            ax.quiver(
                p[0] * sxy,
                p[1] * sxy,
                vplot[0],
                vplot[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=style.tip_color,
                width=style.width,
                alpha=style.alpha,
                headwidth=style.headwidth,
                headlength=style.headlength,
            )
            ax.scatter([p[0] * sxy], [p[1] * sxy], s=30, color=style.tip_color)
            if annotate:
                ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=8, ha="left", va="bottom")

        # Junctions: per incident branch vectors and sum
        if show_junctions and v_inc:
            # draw junction markers once
            for vid, items in v_inc.items():
                if deg.get(int(vid), 0) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=45, color=style.junction_color)
                if annotate:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=9, ha="left", va="bottom")

            for p_m, B_m, col in raw_junc:
                p = np.asarray(p_m, float)
                vplot = np.asarray(B_m, float) * float(scale_fac) * float(sxy)
                ax.quiver(
                    p[0] * sxy,
                    p[1] * sxy,
                    vplot[0],
                    vplot[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=col,
                    width=style.width,
                    alpha=0.85,
                    headwidth=style.headwidth,
                    headlength=style.headlength,
                )

            for p_m, Bsum_m in raw_sum:
                p = np.asarray(p_m, float)
                vplot = np.asarray(Bsum_m, float) * float(scale_fac) * float(sxy)
                ax.quiver(
                    p[0] * sxy,
                    p[1] * sxy,
                    vplot[0],
                    vplot[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=style.sum_color,
                    width=style.width * 1.3,
                    alpha=0.7,
                    headwidth=style.headwidth * 1.1,
                    headlength=style.headlength * 1.1,
                )

        # Scale key
        if vmax > 0 and max_arrow_len_plot > 0:
            # Put the key near lower-left corner
            x_span = float(np.ptp(xs)) if xs.size else 1.0
            y_span = float(np.ptp(ys)) if ys.size else 1.0

            x0 = float(xs.min() * sxy + 0.05 * (x_span * sxy))
            y0 = float(ys.min() * sxy + 0.05 * (y_span * sxy))

            ax.quiver(
                x0,
                y0,
                max_arrow_len_plot,
                0.0,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="k",
                width=style.width,
                alpha=0.8,
                headwidth=style.headwidth,
                headlength=style.headlength,
            )

            ax.text(x0, y0, f"|B|max={vmax:.2e}", fontsize=9, ha="left", va="bottom")

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

        save_fig(fig, self.out_dir, "b_content_vectors", dpi=150)
        return fig

    # ---------------------------------------------------------------------
    # (3) PK-style force vectors (stress-based)
    # ---------------------------------------------------------------------

    def plot_pk_force_vectors(
        self,
        *,
        style: VecPlotStyle = VecPlotStyle(),
        user_scale: float = 1.0,
        units: str = "mm",
        exclude_self: bool = True,
        show_tips: bool = True,
        show_junctions: bool = True,
        n_samples_tip: int = 32,
        annotate: bool = True,
    ):
        """Plot PK-style force vectors at tips and junctions.

        This wraps `PKProcessor.F_PK_edge_tip(...)`.

        Requirements
        ------------
        For `exclude_self=True`, your results object must implement:
            res.stress_field_global_excluding(edge_index, X, Y, add_remote=True)

        Otherwise, set `exclude_self=False` and implement:
            res.stress_field_global(X, Y, add_remote=True)

        If the required hook is missing, vectors are not fabricated.
        """
        V = self.calc.network.vertices
        E = self.calc.network.edges
        deg = self.pk.degree_map()
        vpos = self._vertex_positions()
        sxy = 1e3 if units.lower() == "mm" else 1.0

        # --- automatic normalization (same convention as B-content plot) ---
        xs = np.array([p[0] for p in vpos.values()], float)
        ys = np.array([p[1] for p in vpos.values()], float)
        ext = float(max(np.ptp(xs) if xs.size else 1.0, np.ptp(ys) if ys.size else 1.0, 1e-12))
        max_arrow_len_plot = float(user_scale) * float(style.max_len_factor) * ext * float(sxy)

        # Junction incidence (parametrized), if available
        v_inc: Dict[int, List[Tuple[int, str]]] = {}
        try:
            jd = self.pk.junction_incidence_parametrized()
            v_inc = jd.v_inc
        except Exception:
            v_inc = {}

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("PK-style force vectors (tips and junctions)")

        # draw network midlines
        for e in E:
            p0 = vpos[int(e.v0)]
            p1 = vpos[int(e.v1)]
            ax.plot([p0[0] * sxy, p1[0] * sxy], [p0[1] * sxy, p1[1] * sxy], ls=":", lw=1.0, color="k", alpha=0.4)

        # Collect raw PK vectors (meters) for normalization.
        raw_tip: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (vid, pos_m, F_m)
        raw_junc: List[Tuple[int, np.ndarray, np.ndarray, str]] = []  # (vid, pos_m, F_m, color)
        raw_sum: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (vid, pos_m, sumF_m)

        if show_tips:
            for ei, e in enumerate(E):
                for at, vid in (("start", int(e.v0)), ("end", int(e.v1))):
                    if deg.get(int(vid), 0) != 1:
                        continue
                    p = vpos[int(vid)]
                    F = self.pk.F_PK_edge_tip(int(ei), at=at, exclude_self=bool(exclude_self), n_samples=int(n_samples_tip))
                    if F is None:
                        continue
                    raw_tip.append((int(vid), p, np.asarray(F, float).reshape(2,)))

        # Junctions (per incident branch, plus sum)
        if show_junctions and v_inc:
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C4", "C5"]
            for vid, items in v_inc.items():
                if deg.get(int(vid), 0) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                Fsum = np.zeros(2, float)
                for i, (pid, which) in enumerate(items):
                    ei = self._edge_index_for_pid(int(pid))
                    if ei is None:
                        continue
                    at = str(which).lower()
                    if at not in ("start", "end"):
                        at = "start"
                    F = self.pk.F_PK_edge_tip(int(ei), at=at, exclude_self=bool(exclude_self), n_samples=int(n_samples_tip))
                    if F is None:
                        continue
                    F = np.asarray(F, float).reshape(2,)
                    Fsum += F
                    raw_junc.append((int(vid), p, F, colors[i % len(colors)]))
                raw_sum.append((int(vid), p, Fsum))

        mags = [float(np.hypot(v[0], v[1])) for _, _, v in raw_tip] + \
               [float(np.hypot(v[0], v[1])) for _, _, v, _ in raw_junc] + \
               [float(np.hypot(v[0], v[1])) for _, _, v in raw_sum]
        vmax = float(max(mags)) if mags else 0.0
        scale_fac = (max_arrow_len_plot / vmax) if (vmax > 0.0 and max_arrow_len_plot > 0.0) else 0.0

        # Draw tip markers/vectors
        for vid, p_m, F_m in raw_tip:
            p = np.asarray(p_m, float)
            vplot = np.asarray(F_m, float) * float(scale_fac) * float(sxy)
            ax.scatter([p[0] * sxy], [p[1] * sxy], s=30, color=style.tip_color)
            if annotate:
                ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=8, ha="left", va="bottom")
            ax.quiver(
                p[0] * sxy,
                p[1] * sxy,
                vplot[0],
                vplot[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="C3",
                width=style.width,
                alpha=style.alpha,
                headwidth=style.headwidth,
                headlength=style.headlength,
            )

        # Junction markers
        if show_junctions and v_inc:
            for vid, items in v_inc.items():
                if deg.get(int(vid), 0) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=45, color=style.junction_color)
                if annotate:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=9, ha="left", va="bottom")

        # Junction per-branch vectors
        for vid, p_m, F_m, col in raw_junc:
            p = np.asarray(p_m, float)
            vplot = np.asarray(F_m, float) * float(scale_fac) * float(sxy)
            ax.quiver(
                p[0] * sxy,
                p[1] * sxy,
                vplot[0],
                vplot[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=col,
                width=style.width,
                alpha=0.85,
                headwidth=style.headwidth,
                headlength=style.headlength,
            )

        # Junction sum vectors
        for vid, p_m, Fsum_m in raw_sum:
            p = np.asarray(p_m, float)
            vplot = np.asarray(Fsum_m, float) * float(scale_fac) * float(sxy)
            ax.quiver(
                p[0] * sxy,
                p[1] * sxy,
                vplot[0],
                vplot[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=style.sum_color,
                width=style.width * 1.3,
                alpha=0.65,
                headwidth=style.headwidth * 1.1,
                headlength=style.headlength * 1.1,
            )

        # Scale key
        if vmax > 0 and max_arrow_len_plot > 0:
            x0 = float(xs.min() * sxy + 0.05 * (xs.ptp() * sxy if xs.size else 1.0))
            y0 = float(ys.min() * sxy + 0.05 * (ys.ptp() * sxy if ys.size else 1.0))
            ax.quiver(
                x0,
                y0,
                max_arrow_len_plot,
                0.0,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="k",
                width=style.width,
                alpha=0.8,
                headwidth=style.headwidth,
                headlength=style.headlength,
            )
            ax.text(x0, y0, f"|F_PK|max={vmax:.2e}", fontsize=9, ha="left", va="bottom")

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

        save_fig(fig, self.out_dir, "pk_force_vectors", dpi=150)
        return fig
