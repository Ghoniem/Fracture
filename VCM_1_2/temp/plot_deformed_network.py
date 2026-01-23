"""
Deformed network plotter (v16)

Goal:
- Robust plotting for parametrized-crack solutions using the *existing* helper:
    reconstruct_cod_csd_parametrized_smoothed(res, edge_index, ...)
  which returns (x_edge, COD, CSD).

Fixes relative to earlier iterations:
- Do NOT rely on polyline-chain metadata (segL/path_vids) that may not exist.
- Reconstruct per *network edge* (straight segment between two vertex IDs), then
  remove gauge discontinuities at internal degree-2 vertices by constant shifts of J.
- Enforce J=0 at all degree-1 vertices (tips).
- Title uses sol["junction_model"] (not solver_option).

Optional:
- "core junction trimming" is kept as a plotting post-process and can be applied
  regardless of junction_model via trim_junction_apply_mode="always".

This module is intended to be named:
    fracture_utils/Uplotter/plot_deformed_network.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

# Local helper (your package)
from .reconstruct import reconstruct_cod_csd_parametrized_smoothed


# ---------------------------
# Geometry helpers
# ---------------------------
def _angle_of_vector(v: np.ndarray) -> float:
    return float(math.atan2(float(v[1]), float(v[0])))

def _unit_frame(p0: np.ndarray, p1: np.ndarray):
    d = (p1 - p0).astype(float)
    L = float(np.linalg.norm(d))
    if not np.isfinite(L) or L <= 0.0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0
    t = d / L
    n = np.array([-t[1], t[0]], dtype=float)
    return t, n, L

def _vertex_xy(net, vid: int) -> np.ndarray:
    """Try several common APIs to get (x,y) for a vertex id."""
    # Method 1: vertex_coords(vid)->(2,)
    if hasattr(net, "vertex_coords"):
        xy = np.asarray(net.vertex_coords(vid), float).reshape(-1)
        if xy.size >= 2:
            return xy[:2].copy()

    # Method 2: vertices array with first column = id
    verts = getattr(net, "vertices", None)
    if verts is not None:
        arr = np.asarray(verts)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            # assume [id, x, y]
            mask = arr[:, 0].astype(int) == int(vid)
            if np.any(mask):
                row = arr[mask][0]
                return np.asarray([row[1], row[2]], float)

    # Method 3: dict-like storage
    if hasattr(net, "V") and isinstance(net.V, dict) and vid in net.V:
        row = np.asarray(net.V[vid], float).reshape(-1)
        if row.size >= 2:
            return row[:2].copy()

    raise AttributeError(f"Cannot find coordinates for vertex id={vid} on network object.")

def _edge_endpoints(net, edge_index: int):
    """Return (v0_id, v1_id) for a given edge index.

    Supports networks where edges are stored as:
      - net.connectivity: (Ne,2) int array of vertex IDs
      - net.edges: list/tuple of (v0,v1) pairs OR list of EdgeV4-like objects
      - net.edges: dict edge_index -> EdgeV4-like object
      - net.E / net.edge_list: similar forms

    For EdgeV4-like objects, we try common attribute names (v0/v1, v0_id/v1_id,
    start_vid/end_vid, i/j, etc.).
    """
    # Method 1: connectivity array
    conn = getattr(net, "connectivity", None)
    if conn is not None:
        c = np.asarray(conn)
        if c.ndim == 2 and c.shape[1] >= 2 and 0 <= edge_index < c.shape[0]:
            return int(c[edge_index, 0]), int(c[edge_index, 1])

    def _from_edge_obj(eobj):
        # pair / sequence
        try:
            if isinstance(eobj, (list, tuple, np.ndarray)) and len(eobj) >= 2:
                return int(eobj[0]), int(eobj[1])
        except Exception:
            pass
        # common attribute pairs
        attr_pairs = [
            ("v0", "v1"), ("v0_id", "v1_id"), ("start", "end"), ("start_vid", "end_vid"),
            ("i", "j"), ("a", "b"), ("u", "v"), ("n0", "n1"),
        ]
        for a, b in attr_pairs:
            if hasattr(eobj, a) and hasattr(eobj, b):
                try:
                    return int(getattr(eobj, a)), int(getattr(eobj, b))
                except Exception:
                    pass
        # common container attributes
        for attr in ("vertices", "vids", "endpoints", "verts", "v_ids", "node_ids"):
            if hasattr(eobj, attr):
                val = getattr(eobj, attr)
                try:
                    return int(val[0]), int(val[1])
                except Exception:
                    pass
        # method
        for meth in ("endpoints", "get_endpoints", "as_tuple", "to_tuple"):
            if hasattr(eobj, meth) and callable(getattr(eobj, meth)):
                try:
                    val = getattr(eobj, meth)()
                    return int(val[0]), int(val[1])
                except Exception:
                    pass
        raise TypeError(f"Unrecognized edge object type for endpoints: {type(eobj)}")

    # Method 2: net.edges (list/tuple/dict) or other common containers
    for container_name in ("edges", "E", "edge_list"):
        edges = getattr(net, container_name, None)
        if edges is None:
            continue
        # dict: edge_index -> edge object
        if isinstance(edges, dict):
            if edge_index in edges:
                return _from_edge_obj(edges[edge_index])
            # sometimes keys are strings
            if str(edge_index) in edges:
                return _from_edge_obj(edges[str(edge_index)])
            continue
        # sequence
        try:
            if 0 <= edge_index < len(edges):
                return _from_edge_obj(edges[edge_index])
        except Exception:
            pass

    raise AttributeError("Cannot determine edge endpoints from network object.")

def _num_edges(net) -> int:
    for attr in ("Ne", "n_edges", "num_edges"):
        if hasattr(net, attr):
            try:
                return int(getattr(net, attr))
            except Exception:
                pass
    # fallback: from connectivity
    conn = getattr(net, "connectivity", None)
    if conn is not None:
        return int(np.asarray(conn).shape[0])

    for container_name in ("edges", "E", "edge_list"):
        edges = getattr(net, container_name, None)
        if edges is None:
            continue
        if isinstance(edges, dict):
            return int(len(edges))
        try:
            return int(len(edges))
        except Exception:
            pass
        try:
            return int(np.asarray(edges).shape[0])
        except Exception:
            pass

    raise AttributeError("Cannot infer number of edges from network object.")

def _degree_map(net) -> dict[int, int]:
    """Degree map by vertex id."""
    # Method 1: net.degree_map()
    if hasattr(net, "degree_map"):
        dm = net.degree_map()
        return {int(k): int(v) for k, v in dm.items()}

    # Method 2: compute from connectivity
    deg: dict[int, int] = {}
    Ne = _num_edges(net)
    for ei in range(Ne):
        v0, v1 = _edge_endpoints(net, ei)
        deg[v0] = deg.get(v0, 0) + 1
        deg[v1] = deg.get(v1, 0) + 1
    return deg


# ---------------------------
# Gauge stitching (deg-2)
# ---------------------------
def _stitch_J_across_deg2(edges_data: list[dict], deg: dict[int, int]):
    """
    edges_data: list of dict per edge containing:
      - edge_index
      - v0, v1
      - P (N,2)
      - J (N,2)
      - end_tag mapping: J[0] at v0, J[-1] at v1 (after possible reversal)
    We will shift constant offsets so that at every degree-2 vertex, the incident
    edges match endpoint J values.
    """
    # Build incident map: vid -> list of (edge_idx_in_edges_data, which_end)
    inc: dict[int, list[tuple[int, str]]] = {}
    for k, ed in enumerate(edges_data):
        inc.setdefault(int(ed["v0"]), []).append((k, "start"))
        inc.setdefault(int(ed["v1"]), []).append((k, "end"))

    visited = [False] * len(edges_data)

    # Traverse chains starting from any tip/junction (deg!=2) to propagate gauge consistently.
    # For isolated cycles (all deg==2), start arbitrarily.
    start_vertices = [vid for vid, d in deg.items() if d != 2 and vid in inc]
    if not start_vertices:
        start_vertices = [next(iter(inc.keys()))] if inc else []

    def endpoint_J(ed: dict, which: str) -> np.ndarray:
        return ed["J"][0].copy() if which == "start" else ed["J"][-1].copy()

    def shift_edge(ed: dict, delta: np.ndarray):
        ed["J"] = ed["J"] - delta.reshape(1, 2)

    # BFS/DFS along connectivity
    stack: list[int] = []
    for sv in start_vertices:
        for (ek, _) in inc.get(sv, []):
            if not visited[ek]:
                stack.append(ek)
                visited[ek] = True

        while stack:
            ek = stack.pop()
            edk = edges_data[ek]
            for vid, which_here in ((edk["v0"], "start"), (edk["v1"], "end")):
                if deg.get(int(vid), 0) != 2:
                    continue
                items = inc.get(int(vid), [])
                if len(items) != 2:
                    continue
                # find the other edge at this deg-2 vertex
                (eA, endA), (eB, endB) = items
                other = eB if eA == ek else eA
                end_other = endB if eA == ek else endA

                if visited[other]:
                    continue

                # match other edge endpoint to current edge endpoint at this vertex
                J_target = endpoint_J(edk, which_here)
                J_curr_other = endpoint_J(edges_data[other], end_other)
                delta = (J_curr_other - J_target)
                shift_edge(edges_data[other], delta)

                visited[other] = True
                stack.append(other)

    # Handle any unvisited edges (pure cycles)
    for ek in range(len(edges_data)):
        if visited[ek]:
            continue
        # mark and propagate from this edge arbitrarily
        visited[ek] = True
        stack = [ek]
        while stack:
            ecur = stack.pop()
            edc = edges_data[ecur]
            for vid, which_here in ((edc["v0"], "start"), (edc["v1"], "end")):
                if deg.get(int(vid), 0) != 2:
                    continue
                items = inc.get(int(vid), [])
                if len(items) != 2:
                    continue
                (eA, endA), (eB, endB) = items
                other = eB if eA == ecur else eA
                end_other = endB if eA == ecur else endA
                if visited[other]:
                    continue
                J_target = endpoint_J(edc, which_here)
                J_curr_other = endpoint_J(edges_data[other], end_other)
                delta = (J_curr_other - J_target)
                shift_edge(edges_data[other], delta)
                visited[other] = True
                stack.append(other)


def _enforce_tip_zero(edges_data: list[dict], deg: dict[int, int]):
    """Set J=0 at all degree-1 vertices by constant shift per incident edge."""
    for ed in edges_data:
        v0, v1 = int(ed["v0"]), int(ed["v1"])
        if deg.get(v0, 0) == 1:
            ed["J"] = ed["J"] - ed["J"][0:1, :]
        if deg.get(v1, 0) == 1:
            ed["J"] = ed["J"] - ed["J"][-1:, :]


# ---------------------------
# Optional junction trimming
# ---------------------------
def _intersect_rays(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray):
    """Intersection of rays p0->p1 and q0->q1. Returns (ok, x)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    u = p1 - p0
    v = q1 - q0
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]], float)
    b = (q0 - p0).reshape(2,)
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or abs(det) < 1e-14:
        return False, 0.5*(p0+q0)
    t_s = np.linalg.solve(A, b)
    t, s = float(t_s[0]), float(t_s[1])
    if (t < 0.0) or (s < 0.0):
        # intersection behind one of the rays; still return point on closest-forward
        return False, 0.5*(p0+q0)
    x = p0 + t*u
    return True, x

def _trim_core_junction_faces_inplace(edges_data: list[dict], deg: dict[int, int],
                                      junction_model: str, min_degree: int = 3,
                                      look_ahead: int = 2, apply_mode: str = "core"):
    jm = str(junction_model).lower()
    mode = str(apply_mode).lower() if apply_mode is not None else "core"
    if mode in ("core", "core_only") and jm != "core":
        return

    # Build incident face-end data for each vertex
    inc: dict[int, list[dict]] = {}
    for k, ed in enumerate(edges_data):
        P = ed["P"]; U = ed["U"]; L = ed["L"]
        v0, v1 = int(ed["v0"]), int(ed["v1"])

        # start end (v0)
        if P.shape[0] >= 2:
            tdir = (P[min(look_ahead, P.shape[0]-1)] - P[0])
            inc.setdefault(v0, []).append({
                "angle": _angle_of_vector(tdir),
                "edge_k": k,
                "end": "start",
                "U0": U[0].copy(),
                "U1": U[min(look_ahead, U.shape[0]-1)].copy(),
                "L0": L[0].copy(),
                "L1": L[min(look_ahead, L.shape[0]-1)].copy(),
            })
        # end end (v1)
        if P.shape[0] >= 2:
            tdir = (P[-1] - P[max(-1-look_ahead, -P.shape[0])])
            inc.setdefault(v1, []).append({
                "angle": _angle_of_vector(tdir),
                "edge_k": k,
                "end": "end",
                "U0": U[-1].copy(),
                "U1": U[max(-1-look_ahead, -U.shape[0])].copy(),
                "L0": L[-1].copy(),
                "L1": L[max(-1-look_ahead, -L.shape[0])].copy(),
            })

    for vid, items in inc.items():
        if deg.get(int(vid), 0) < min_degree:
            continue
        if len(items) < min_degree:
            continue

        items.sort(key=lambda z: z["angle"])
        m = len(items)

        # For cyclic order, enforce L_j meets U_{j+1}
        for j in range(m):
            A = items[j]
            B = items[(j+1) % m]
            ok, x = _intersect_rays(A["L0"], A["L1"], B["U0"], B["U1"])

            # overwrite the endpoint coordinate (only) to avoid crossings
            eA = edges_data[A["edge_k"]]
            eB = edges_data[B["edge_k"]]

            if A["end"] == "start":
                eA["L"][0, :] = x
            else:
                eA["L"][-1, :] = x

            if B["end"] == "start":
                eB["U"][0, :] = x
            else:
                eB["U"][-1, :] = x


# ---------------------------
# Public plotter
# ---------------------------
@dataclass
class VCMPlotDeformedNetwork:
    res: object
    out_dir: Path | None = None

    def __post_init__(self):
        if self.out_dir is None:
            self.out_dir = Path(getattr(self.res, "out_dir", "output"))

    def plot_deformed_network(
        self,
        n_pts_per_edge: int = 2000,
        scale: float = 2e1,
        show_faces: bool = True,
        units: str = "mm",
        cod_smooth_window: int | str = 0,
        csd_smooth_window: int | str = 0,
        trim_core_junction_faces: bool = False,
        trim_junction_apply_mode: str = "core",
        show: bool = True,
        save: bool = True,
        debug_counts: bool = True,
    ):
        res = self.res
        if not hasattr(res, "calc") or not hasattr(res.calc, "network"):
            raise AttributeError("res.calc.network is required for plotting.")

        net = res.calc.network
        deg = _degree_map(net)

        sol = getattr(res, "sol", {}) or {}
        # Correct label: junction_model from solver kwargs, not solver_option.
        junction_model = sol.get("junction_model", sol.get("junctionModel", ""))
        if junction_model is None or junction_model == "":
            junction_model = "unknown"

        # Unit scaling for axes labels only
        sxy = 1e3 if str(units).lower() == "mm" else 1.0

        # color cycle
        prop = plt.rcParams.get("axes.prop_cycle", None)
        color_list = prop.by_key().get("color", []) if prop is not None else []
        if not color_list:
            color_list = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        Ne = _num_edges(net)
        edges_data: list[dict] = []

        n_plotted = 0
        n_skipped = 0

        for ei in range(Ne):
            try:
                v0, v1 = _edge_endpoints(net, ei)
                p0 = _vertex_xy(net, v0)
                p1 = _vertex_xy(net, v1)
            except Exception:
                n_skipped += 1
                continue

            t_hat, n_hat, L = _unit_frame(p0, p1)
            if L <= 0.0:
                n_skipped += 1
                continue

            # Reconstruction helper (panel-aware)
            try:
                x_edge, COD, CSD = reconstruct_cod_csd_parametrized_smoothed(
                    res,
                    edge_index=int(ei),
                    n_pts=int(n_pts_per_edge),
                    enforce_global_tip_zero=False,
                    cod_window_panels=0 if cod_smooth_window in (0, None) else int(cod_smooth_window),
                    csd_window_panels=0 if csd_smooth_window in (0, None) else int(csd_smooth_window),
                )
            except Exception:
                n_skipped += 1
                continue

            x_edge = np.asarray(x_edge, float).reshape(-1,)
            COD = np.asarray(COD, float).reshape(-1,)
            CSD = np.asarray(CSD, float).reshape(-1,)

            if x_edge.size < 2:
                n_skipped += 1
                continue

            # Map x_edge in [-a,a] to param in [0,1]
            a = 0.5 * L
            tpar = (x_edge + a) / (2.0 * a)
            tpar = np.clip(tpar, 0.0, 1.0)

            P = p0.reshape(1,2) + tpar.reshape(-1,1) * (p1 - p0).reshape(1,2)

            # Vector jump J = CSD*t + COD*n
            J = (CSD.reshape(-1,1) * t_hat.reshape(1,2)) + (COD.reshape(-1,1) * n_hat.reshape(1,2))

            edges_data.append({
                "edge_index": int(ei),
                "v0": int(v0),
                "v1": int(v1),
                "P": P,
                "J": J,
                "color": color_list[int(ei) % len(color_list)],
            })
            n_plotted += 1

        # If nothing reconstructed, still return axes (but show reason)
        if not edges_data:
            ax.set_title(f"Deformed Crack Network (junction_model={junction_model}, scale={scale: .2e})")
            if debug_counts:
                ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9)
            if save:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(self.out_dir / "deformed_network.png", dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            return fig, ax

        # Stitch gauge across internal degree-2 vertices, then enforce tips closed
        _stitch_J_across_deg2(edges_data, deg)
        _enforce_tip_zero(edges_data, deg)

        # Build faces
        for ed in edges_data:
            P = ed["P"]; J = ed["J"]
            ed["U"] = P + 0.5 * float(scale) * J
            ed["L"] = P - 0.5 * float(scale) * J

        # Optional trimming for junction face crossings
        if trim_core_junction_faces:
            _trim_core_junction_faces_inplace(
                edges_data, deg,
                junction_model=str(junction_model),
                min_degree=3,
                look_ahead=2,
                apply_mode=trim_junction_apply_mode,
            )

        # Plot
        for ed in edges_data:
            P = ed["P"]; U = ed["U"]; Lw = ed["L"]
            col = ed["color"]

            # midline
            ax.plot(P[:,0]*sxy, P[:,1]*sxy, color="0.5", lw=1.0, ls=":", alpha=0.9)

            if show_faces:
                ax.plot(U[:,0]*sxy, U[:,1]*sxy, color=col, lw=1.6)
                ax.plot(Lw[:,0]*sxy, Lw[:,1]*sxy, color=col, lw=1.6)

        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.set_title(f"Deformed Crack Network (junction_model={junction_model}, scale={float(scale):.2e})")

        if debug_counts:
            ax.text(0.02, 0.98, f"plotted={n_plotted}, skipped={n_skipped}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9)

        if save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / "deformed_network.png", dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax


# Backward compatibility alias for the older typo'd name
VCMPLotDeformedNetwork = VCMPlotDeformedNetwork

__all__ = ["VCMPlotDeformedNetwork", "VCMPLotDeformedNetwork"]