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

from .plot_opts import ensure_dir, save_fig
from .processor_PK import PKProcessor


@dataclass
class VecPlotStyle:
    vec_scale: float = 1.0
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

        def _arrow(p_m: np.ndarray, v_m: np.ndarray, *, color: str, lw: float, alpha: float):
            """Draw a clear begin→end arrow.

            Parameters are in meters; we convert to plot units (mm if requested).
            """
            p_m = np.asarray(p_m, float).reshape(2,)
            v_m = np.asarray(v_m, float).reshape(2,)
            p0 = p_m * sxy
            p1 = (p_m + v_m) * sxy
            ax.annotate(
                "",
                xy=(float(p1[0]), float(p1[1])),
                xytext=(float(p0[0]), float(p0[1])),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha, shrinkA=0.0, shrinkB=0.0),
            )

        def _tip_B_vector(vid: int) -> Optional[np.ndarray]:
            """Return a single resultant B-content vector for a degree-1 vertex.

            Uses the polyline orientation (v_start->v_end) to choose a consistent sign:
              - if tip is v_end:  use +B
              - if tip is v_start: use -B
            """
            if deg.get(int(vid), 0) != 1:
                return None
            # Find the unique incident edge
            ei_inc = None
            for ei, e in enumerate(E):
                if int(e.v0) == int(vid) or int(e.v1) == int(vid):
                    ei_inc = int(ei)
                    break
            if ei_inc is None:
                return None
            B = self._branch_content_for_edge(ei_inc)
            if B is None:
                return None
            # Determine sign from parametrized polyline metadata if available
            sol = getattr(self.res, "sol", None)
            if isinstance(sol, dict):
                pid = self._pid_for_edge_index(ei_inc)
                polys = sol.get("parametrized_polylines", [])
                if pid is not None and 0 <= int(pid) < len(polys):
                    meta = dict(polys[int(pid)])
                    vids = meta.get("path_vertex_ids", [])
                    v_start = int(meta.get("v_start", vids[0] if vids else -1))
                    v_end = int(meta.get("v_end", vids[-1] if vids else -1))
                    if int(vid) == v_start and int(vid) != v_end:
                        return -B
                    if int(vid) == v_end and int(vid) != v_start:
                        return +B
            return B

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

        # Tips: one resultant vector per degree-1 vertex
        if show_tips:
            for vid, d in deg.items():
                if int(d) != 1:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                Btip = _tip_B_vector(int(vid))
                if Btip is None:
                    continue
                _arrow(p, np.asarray(Btip, float).reshape(2,) * float(style.vec_scale), color=style.tip_color, lw=1.8, alpha=style.alpha)
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=34, color=style.tip_color)
                if annotate:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=8, ha="left", va="bottom")

        # Junctions: per incident branch vectors and sum
        if show_junctions and v_inc:
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C4", "C5"]
            for vid, items in v_inc.items():
                if deg.get(int(vid), 0) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue

                ax.scatter([p[0] * sxy], [p[1] * sxy], s=45, color=style.junction_color)
                if annotate:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=9, ha="left", va="bottom")

                Bsum = np.zeros(2, float)
                for i, (pid, _which) in enumerate(items):
                    ei = self._edge_index_for_pid(int(pid))
                    if ei is None:
                        continue
                    B = self._branch_content_for_edge(int(ei))
                    if B is None:
                        continue
                    Bsum += B
                    _arrow(
                        p,
                        np.asarray(B, float).reshape(2,) * float(style.vec_scale),
                        color=colors[i % len(colors)],
                        lw=1.6,
                        alpha=0.85,
                    )

                if show_sum_at_junction:
                    _arrow(
                        p,
                        np.asarray(Bsum, float).reshape(2,) * float(style.vec_scale),
                        color=style.sum_color,
                        lw=2.4,
                        alpha=0.7,
                    )

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

        # Tips
        if show_tips:
            for ei, e in enumerate(E):
                for at, vid in (("start", int(e.v0)), ("end", int(e.v1))):
                    if deg.get(int(vid), 0) != 1:
                        continue
                    p = vpos[int(vid)]
                    F = self.pk.F_PK_edge_tip(int(ei), at=at, exclude_self=bool(exclude_self), n_samples=int(n_samples_tip))
                    ax.scatter([p[0] * sxy], [p[1] * sxy], s=30, color=style.tip_color)
                    if annotate:
                        ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=8, ha="left", va="bottom")
                    if F is None:
                        continue
                    F = np.asarray(F, float).reshape(2,) * float(style.vec_scale)
                    ax.quiver(
                        p[0] * sxy,
                        p[1] * sxy,
                        F[0],
                        F[1],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="C3",
                        width=style.width,
                        alpha=style.alpha,
                        headwidth=style.headwidth,
                        headlength=style.headlength,
                    )

        # Junctions (per incident branch, plus sum)
        if show_junctions and v_inc:
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0", "C1", "C2", "C4", "C5"]
            for vid, items in v_inc.items():
                if deg.get(int(vid), 0) < 2:
                    continue
                p = vpos.get(int(vid), None)
                if p is None:
                    continue
                ax.scatter([p[0] * sxy], [p[1] * sxy], s=45, color=style.junction_color)
                if annotate:
                    ax.text(p[0] * sxy, p[1] * sxy, f"v{int(vid)}", fontsize=9, ha="left", va="bottom")

                Fsum = np.zeros(2, float)
                for i, (pid, which) in enumerate(items):
                    ei = self._edge_index_for_pid(int(pid))
                    if ei is None:
                        continue
                    # Map "which" (start/end on polyline) to edge end. For most cases, pid corresponds to the edge.
                    at = str(which).lower()
                    if at not in ("start", "end"):
                        at = "start"
                    F = self.pk.F_PK_edge_tip(int(ei), at=at, exclude_self=bool(exclude_self), n_samples=int(n_samples_tip))
                    if F is None:
                        continue
                    F = np.asarray(F, float).reshape(2,) * float(style.vec_scale)
                    Fsum += F
                    ax.quiver(
                        p[0] * sxy,
                        p[1] * sxy,
                        F[0],
                        F[1],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color=colors[i % len(colors)],
                        width=style.width,
                        alpha=0.85,
                        headwidth=style.headwidth,
                        headlength=style.headlength,
                    )

                ax.quiver(
                    p[0] * sxy,
                    p[1] * sxy,
                    Fsum[0],
                    Fsum[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=style.sum_color,
                    width=style.width * 1.3,
                    alpha=0.65,
                    headwidth=style.headwidth * 1.1,
                    headlength=style.headlength * 1.1,
                )

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

        save_fig(fig, self.out_dir, "pk_force_vectors", dpi=150)
        return fig
