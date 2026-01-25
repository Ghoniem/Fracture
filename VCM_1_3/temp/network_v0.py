"""Graph/network primitives for v4 polyline solver."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class VertexV4:
    id: int
    x: float
    y: float
    edges: list = field(default_factory=list)

    def xy(self) -> np.ndarray:
        return np.array([float(self.x), float(self.y)], dtype=float)


@dataclass(frozen=True)
class EdgeV4:
    id: int
    v0: int
    v1: int


class CrackNetworkV4:
    def __init__(self, vertices: List[VertexV4], edges: List[EdgeV4], Nv_max: int = 3):
        self.vertices = list(vertices)
        self.edges = list(edges)
        self.Nv_max = int(Nv_max)
        self._vmap = {v.id: v for v in self.vertices}

        deg = {v.id: 0 for v in self.vertices}
        for e in self.edges:
            deg[e.v0] += 1
            deg[e.v1] += 1
        for vid, d in deg.items():
            if d > self.Nv_max:
                raise ValueError(f"Vertex {vid} degree {d} exceeds Nv_max={self.Nv_max}")

    def V(self, vid: int) -> VertexV4:
        return self._vmap[int(vid)]

    def vertex_coords(self, vid: int) -> np.ndarray:
        v = self.V(vid)
        return np.array([float(v.x), float(v.y)], dtype=float)

    @classmethod
    def from_vertices_connectivity(
        cls,
        vertices: np.ndarray,
        connectivity: np.ndarray,
        *,
        Nv_max: int = 3,
        validate: bool = True,
    ) -> "CrackNetworkV4":
        vertices = np.asarray(vertices, float)
        if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
            raise ValueError("vertices must have shape (Nv,2) or (Nv,3) with (id,x,y)")
        if vertices.shape[1] == 3:
            vid = vertices[:, 0].astype(int)
            xy = vertices[:, 1:3]
        else:
            vid = np.arange(vertices.shape[0], dtype=int)
            xy = vertices[:, 0:2]
        id_to_i = {int(v): i for i, v in enumerate(vid)}
        V = [VertexV4(int(vid[i]), float(xy[i, 0]), float(xy[i, 1]), []) for i in range(len(vid))]

        conn = np.asarray(connectivity)
        if conn.ndim != 2 or conn.shape[1] not in (2, 3):
            raise ValueError("connectivity must have shape (Ne,2) or (Ne,3) with (id,v0,v1)")
        if conn.shape[1] == 3:
            eid = conn[:, 0].astype(int)
            v0 = conn[:, 1].astype(int)
            v1 = conn[:, 2].astype(int)
        else:
            eid = np.arange(conn.shape[0], dtype=int)
            v0 = conn[:, 0].astype(int)
            v1 = conn[:, 1].astype(int)

        seen = set()
        edges: List[EdgeV4] = []
        for k in range(len(eid)):
            a = int(v0[k]); b = int(v1[k])
            if a == b:
                if validate:
                    raise ValueError(f"Degenerate edge ({a},{b})")
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            if a not in id_to_i or b not in id_to_i:
                raise ValueError(f"Edge references unknown vertex id: ({a},{b})")
            edges.append(EdgeV4(int(eid[k]), a, b))

        for e in edges:
            V[id_to_i[e.v0]].edges.append(e.id)
            V[id_to_i[e.v1]].edges.append(e.id)

        if validate:
            for v in V:
                if len(v.edges) > int(Nv_max):
                    raise ValueError(f"Vertex {v.id} degree {len(v.edges)} exceeds Nv_max={Nv_max}")
        return cls(vertices=V, edges=edges, Nv_max=Nv_max)
