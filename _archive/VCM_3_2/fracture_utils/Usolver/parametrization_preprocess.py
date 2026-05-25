"""
Pre-processing helpers for the parametrized polyline solver (v2.0).

Scope:
- Normalize representation/node_distribution/collocation_mode (Chebyshev aliases)
- Optional conversion of straight polylines into interpolating cubic splines ('cspline' mode)

This module should remain free of KKT/constraints and operator assembly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from .network import CrackNetworkV4
from .build_curved import make_cspline_polyline_from_path


def normalize_representation(
    representation: str,
    node_distribution: str,
) -> Tuple[str, str, str]:
    """
    Map user-facing representation aliases to internal (rep_in, dist_in, collocation_mode).

    Supported user inputs:
      - 'panel', 'singular'
      - 'cheb_quad', 'cheb_spectral' (+ optional '_singular' suffix)

    Returns
    -------
    rep_in : 'panel'|'singular'
    dist_in : 'uniform'|'tip_dense'
    collocation_mode : 'mid'|'nodes'
    """
    rep_in = str(representation).lower().strip()
    dist_in = str(node_distribution).lower().strip()

    collocation_mode = "mid"
    if rep_in in ("cheb_quad", "cheb_spectral", "cheb_quad_singular", "cheb_spectral_singular"):
        if rep_in.startswith("cheb_spectral"):
            collocation_mode = "nodes"
        dist_in = "tip_dense"
        rep_in = "singular" if rep_in.endswith("_singular") else "panel"

    if rep_in not in ("panel", "singular"):
        raise ValueError(
            f"Unknown representation={representation!r}. Use 'panel', 'singular', 'cheb_quad', or 'cheb_spectral'."
        )
    if dist_in not in ("uniform", "tip_dense"):
        raise ValueError(f"Unknown node_distribution={node_distribution!r}. Use 'uniform' or 'tip_dense'.")
    return rep_in, dist_in, collocation_mode


def convert_polylines_to_csplines(
    network: CrackNetworkV4,
    polylines: List[dict],
    *,
    crack_mode: str,
    parametrization: str,
    branch_param: Dict[Tuple[int, int], str] | None = None,
) -> List[dict]:
    """
    Convert eligible straight polylines to 'cspline' polylines.

    Rules
    -----
    - parametrization in {'cspline','cubic_spline'}: convert all straight polylines with >=3 vertices.
    - parametrization == 'mixed': convert only those with a key present in branch_param mapping to 'cspline'.

    Parameters
    ----------
    branch_param : dict | None
        Mapping from endpoint-key to a mode ('cspline'). Keys may be (min(v0,v1),max(v0,v1))
        or oriented (v0,v1).

    Returns
    -------
    new_polylines : list[dict]
    """
    param_kind = str(parametrization).lower().strip()
    if param_kind not in ("cspline", "cubic_spline", "mixed"):
        return polylines

    new_polylines: List[dict] = []
    for meta in polylines:
        if str(meta.get("kind", "polyline")).lower().strip() != "polyline":
            new_polylines.append(meta)
            continue

        vids_path = list(meta.get("path_vertex_ids", []))
        eidx_path = list(meta.get("path_edge_indices", []))
        if len(vids_path) < 3:
            new_polylines.append(meta)
            continue

        if param_kind == "mixed":
            v0 = int(meta.get("v_start", vids_path[0]))
            v1 = int(meta.get("v_end", vids_path[-1]))
            key = (min(v0, v1), max(v0, v1))
            mode = None
            if isinstance(branch_param, dict):
                mode = branch_param.get(key, branch_param.get((v0, v1), None))
            if str(mode).lower().strip() not in ("cspline", "cubic_spline"):
                new_polylines.append(meta)
                continue

        pts = np.array([network.vertex_coords(v) for v in vids_path], float)
        new_polylines.append(
            make_cspline_polyline_from_path(
                pts,
                vids_path,
                eidx_path,
                crack_mode=crack_mode,
                v_start=int(meta.get("v_start", vids_path[0])),
                v_end=int(meta.get("v_end", vids_path[-1])),
            )
        )

    return new_polylines
