"""Geometry update utilities for crack propagation.

This module is intentionally small: it takes an existing CrackNetworkV4 and
returns a new CrackNetworkV4 with a single tip extended by a straight segment.

v1 assumptions:
- The tip is a degree-1 vertex (open crack tip).
- Extension is a straight segment of length delta_a.
- Direction is given as theta relative to the local tangent (tip.t_hat).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4


@dataclass(frozen=True)
class GeometryUpdater:
    """Apply geometric updates to a crack network.

    The updater is responsible only for updating topology/geometry, not solving.
    """

    def extend_tip(self, network: CrackNetworkV4, tip: Any, *, theta: float, delta_a: float) -> CrackNetworkV4:
        """Extend a degree-1 tip by adding a new vertex and a new line edge.

        Parameters
        ----------
        network:
            Existing network.
        tip:
            TipState-like object with fields:
              - v_tip: int vertex id at the tip
              - x_tip: (2,) tip coordinates
              - t_hat: (2,) unit tangent pointing *outward* from interior to tip
        theta:
            Propagation angle relative to tip tangent (radians). Positive is CCW.
        delta_a:
            Extension length.

        Returns
        -------
        CrackNetworkV4
            New network with the extra segment appended at v_tip.
        """
        v_tip = int(getattr(tip, "v_tip"))
        x0 = np.asarray(getattr(tip, "x_tip"), float).reshape(2,)
        t = np.asarray(getattr(tip, "t_hat"), float).reshape(2,)

        # Rotate tangent by theta
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        d = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]], dtype=float)
        dn = float(np.hypot(d[0], d[1]))
        if dn <= 0.0:
            raise ValueError("Propagation direction has zero length.")
        d /= dn

        x1 = x0 + float(delta_a) * d

        # New ids
        new_vid = int(max(v.id for v in network.vertices) + 1) if network.vertices else 0
        new_eid = int(max(e.id for e in network.edges) + 1) if network.edges else 0

        # Copy vertices (copy adjacency lists)
        V = [VertexV4(v.id, v.x, v.y, list(v.edges), role=v.role) for v in network.vertices]
        V.append(VertexV4(new_vid, float(x1[0]), float(x1[1]), [], role="node"))

        # Copy edges + add new edge
        E = list(network.edges)
        E.append(EdgeV4(new_eid, v_tip, new_vid, kind="line"))

        # Update adjacency lists for the two vertices
        for vv in V:
            if int(vv.id) == v_tip:
                vv.edges.append(new_eid)
            elif int(vv.id) == new_vid:
                vv.edges.append(new_eid)

        return CrackNetworkV4(vertices=V, edges=E, Nv_max=network.Nv_max)
