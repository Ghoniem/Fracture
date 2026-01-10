
"""
dce_solver_parametrized_v4_1.py

v4 parametrized polyline solver (true polyline integral operator).

Key properties:
- Unknowns are Burgers density components (Mode II glide along tangent, Mode I climb along normal)
  represented on each *polyline crack* (not per segment), using a piecewise-constant panel basis
  along arc-length s in [0, L].
- Traction-free boundary conditions are enforced by collocation on the *physical polyline geometry*.
- Multiple disconnected polyline cracks are solved simultaneously in one global constrained LSQ solve
  (crack–crack interaction included).
- Internal degree-2 vertices introduce no extra tips and require no continuity constraints; they are
  just points where the tangent changes.

Notes:
- This is an infinite-medium, isotropic elasticity formulation.
- Plane strain is assumed in the stress kernel (consistent with the classic edge-dislocation solution).
- Closure (Burgers gauge) constraints are enforced per polyline and per mode:
      ∫_0^L bI(s) ds = 0,   ∫_0^L bII(s) ds = 0
  via a small KKT system (2 constraints per polyline).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import math

# from single_crack_utils.dce_solver_v2 import Material, AppliedStress, Crack  # Crack used only for endpoints utilities
@dataclass(frozen=True)
class Material:
    E: float
    nu: float
    plane_stress: bool = False

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self) -> float:
        if self.plane_stress:
            return (3.0 - self.nu) / (1.0 + self.nu)
        return 3.0 - 4.0 * self.nu


@dataclass(frozen=True)
class Crack:
    length: float  # 2a
    angle: float  # radians
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def half_length(self) -> float:
        return 0.5 * self.length


@dataclass(frozen=True)
class AppliedStress:
    sigma_xx: float
    sigma_yy: float
    sigma_xy: float

    def tensor(self) -> np.ndarray:
        return np.array(
            [[self.sigma_xx, self.sigma_xy], [self.sigma_xy, self.sigma_yy]], dtype=float
        )



# ------------------------
# Graph structures (same as v0/v1)
# ------------------------
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
        degrees: bool = True,
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


# ------------------------
# KKT constrained LSQ
# ------------------------
def solve_kkt_lsq(K: np.ndarray, rhs: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    K = np.asarray(K, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    C = np.asarray(C, float)

    A = K.T @ K
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0])
    b = K.T @ rhs

    m = C.shape[0]
    Z = np.zeros((m, m), float)
    KKT = np.block([[A, C.T],
                    [C, Z]])
    bb = np.concatenate([b, np.zeros(m)])
    try:
        sol = np.linalg.solve(KKT, bb)
    except np.linalg.LinAlgError:
        eps = float(1e-12 * (np.max(np.abs(np.diag(A))) if A.size else 1.0))
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-12
        A2 = A + eps * np.eye(A.shape[0])
        KKT = np.block([[A2, C.T],
                        [C, Z]])
        sol = np.linalg.solve(KKT, bb)
    return sol[:A.shape[0]]


# ------------------------
# Polyline utilities
# ------------------------
def _build_adjacency(network: CrackNetworkV4):
    V = [int(v.id) for v in network.vertices]
    E = network.edges
    v2e = {vid: [] for vid in V}
    e2v = []
    for ei, e in enumerate(E):
        a = int(e.v0); b = int(e.v1)
        e2v.append((a, b))
        v2e.setdefault(a, []).append(int(ei))
        v2e.setdefault(b, []).append(int(ei))
    return v2e, e2v


def _find_polyline_components(network: CrackNetworkV4) -> list[dict]:
    v2e, e2v = _build_adjacency(network)
    v_adj = {int(v.id): set() for v in network.vertices}
    for (a, b) in e2v:
        v_adj[a].add(b)
        v_adj[b].add(a)

    unvisited = set(v_adj.keys())
    comps = []
    while unvisited:
        start = next(iter(unvisited))
        stack = [start]
        comp_verts = set()
        comp_edges = set()
        while stack:
            v = stack.pop()
            if v not in unvisited:
                continue
            unvisited.remove(v)
            comp_verts.add(int(v))
            for ei in v2e.get(v, []):
                comp_edges.add(int(ei))
                a, b = e2v[ei]
                w = b if a == v else a
                if w in unvisited:
                    stack.append(w)
        if comp_edges:
            comps.append({"verts": comp_verts, "edges": comp_edges})
    return comps


def _path_order_for_component(network: CrackNetworkV4, comp_edges: set[int]) -> tuple[list[int], list[int]]:
    v2e, e2v = _build_adjacency(network)
    comp_edges = set(int(ei) for ei in comp_edges)

    # restricted degrees
    deg = {}
    comp_verts = set()
    for ei in comp_edges:
        a, b = e2v[ei]
        comp_verts.add(a); comp_verts.add(b)
    for v in comp_verts:
        inc = [ei for ei in v2e.get(v, []) if ei in comp_edges]
        deg[v] = len(inc)

    ends = [v for v, d in deg.items() if d == 1]
    if len(ends) != 2 or any(d not in (1, 2) for d in deg.values()):
        raise ValueError(
            "v4 polyline solver requires each connected component to be a simple chain "
            "(exactly two degree-1 vertices, all others degree-2)."
        )

    start = int(ends[0])
    vids_path = [start]
    eidx_path = []
    cur = start
    used_edges = set()
    for _ in range(len(comp_edges)):
        inc = [ei for ei in v2e.get(cur, []) if (ei in comp_edges and ei not in used_edges)]
        if not inc:
            break
        ei = int(inc[0])
        used_edges.add(ei)
        a, b = e2v[ei]
        nxt = b if a == cur else a
        eidx_path.append(ei)
        vids_path.append(int(nxt))
        cur = int(nxt)

    if len(eidx_path) != len(comp_edges):
        raise ValueError("Could not traverse component as a single path.")
    return vids_path, eidx_path


def _polyline_point_and_frame(network: CrackNetworkV4, vids_path: list[int], seg_lengths: np.ndarray, s: float):
    """Map arc-length s in [0,L] to physical point x(s), and return local tangent and normal."""
    s = float(s)
    pts = np.array([network.vertex_coords(v) for v in vids_path], float)
    Ls = np.asarray(seg_lengths, float)
    Lcum = np.r_[0.0, np.cumsum(Ls)]
    L = float(Lcum[-1])
    s = min(max(s, 0.0), L)
    # find segment
    k = int(np.searchsorted(Lcum, s, side="right") - 1)
    k = max(0, min(k, len(Ls)-1))
    s0 = float(Lcum[k])
    le = float(Ls[k])
    lam = 0.0 if le <= 0 else (s - s0)/le
    p0 = pts[k]; p1 = pts[k+1]
    x = p0 + lam*(p1-p0)
    t = p1 - p0
    nrm = float(np.hypot(t[0], t[1]))
    if nrm <= 0:
        t = np.array([1.0, 0.0])
    else:
        t = t / nrm
    n = np.array([-t[1], t[0]])
    return x, t, n


# ------------------------
# Edge-dislocation stress kernel (plane strain), differential Burgers dB at source
# ------------------------
def _stress_edge_dislocation(dx: np.ndarray, dy: np.ndarray, dBx: float, dBy: float, mu: float, nu: float):
    """
    Return global stress components (sxx, syy, sxy) at field point relative to source at origin,
    due to differential Burgers components (dBx, dBy) along global x,y.

    Implementation uses a rotation construction:
      - start from known formulas for Burgers along +x ("bx-case"),
      - obtain "by-case" by 90-degree rotation of coordinates and tensor transform.

    This is for visualization/validation and intended to be consistent across the solver/results.
    """
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    mu = float(mu); nu = float(nu)

    eps = 1e-30
    r2 = dx*dx + dy*dy + eps
    r4 = r2*r2

    coef = mu / (2.0*math.pi*(1.0 - nu))

    # bx formulas (standard)
    def bx_stress(x, y, b):
        sxx = -coef*b * (y*(3.0*x*x + y*y)) / ( (x*x + y*y + eps)**2 )
        syy =  coef*b * (y*(x*x - y*y))     / ( (x*x + y*y + eps)**2 )
        sxy =  coef*b * (x*(x*x - y*y))     / ( (x*x + y*y + eps)**2 )
        return sxx, syy, sxy

    sxx = np.zeros_like(dx)
    syy = np.zeros_like(dx)
    sxy = np.zeros_like(dx)

    if abs(dBx) > 0:
        a,b,c = bx_stress(dx, dy, dBx)
        sxx += a; syy += b; sxy += c

    if abs(dBy) > 0:
        # by case via 90° CCW rotation: x' = dy, y' = -dx, with bx' = dBy
        x1 = dy
        y1 = -dx
        sxx1, syy1, sxy1 = bx_stress(x1, y1, dBy)
        # rotate tensor back: for 90° CCW, Q = [[0,-1],[1,0]]
        # σ = Q σ' Q^T gives:
        # σ_xx = σ'_yy
        # σ_yy = σ'_xx
        # σ_xy = -σ'_xy
        sxx += syy1
        syy += sxx1
        sxy += -sxy1

    return sxx, syy, sxy


# ------------------------
# Network solver (polyline integral operator)
# ------------------------
class DCENetworkStaticV4:
    def __init__(self, material: Material, network: CrackNetworkV4, applied: AppliedStress):
        self.material = material
        self.network = network
        self.applied = applied

    def solve(
        self,
        ne_half: int,
        representation: str = "panel",
        *,
        solver_option: str = "parametrized_crack",
        parametrization: str = "polyline",
        node_distribution: str = "tip_dense",
        nq_col: int = 3,
        nq_stress: int = 6,
        ridge: float = 0.0,
        add_vertex_constraints: bool = False,
        junction_option: int = 0,
        r0_factor: float = 0.0,
        n_int: int = 0,
    ) -> Dict:
        opt = str(solver_option).lower().strip()
        if opt not in ("parametrized_crack", "parameterized_crack", "param_crack"):
            raise ValueError("v4 solver supports solver_option='parametrized_crack' only.")
        if str(parametrization).lower().strip() not in ("polyline", "segmented", "kinked"):
            raise ValueError("v4 solver currently supports parametrization='polyline' only.")

        # Build polylines (connected components)
        comps = _find_polyline_components(self.network)
        if not comps:
            raise ValueError("No edges in network.")
        polylines = []
        edge_to_polyline = {}

        for pid, comp in enumerate(comps):
            vids_path, eidx_path = _path_order_for_component(self.network, comp["edges"])
            pts = np.array([self.network.vertex_coords(v) for v in vids_path], float)
            seg = pts[1:] - pts[:-1]
            segL = np.sqrt(np.sum(seg*seg, axis=1))
            L = float(np.sum(segL))
            p0 = pts[0]; p1 = pts[-1]
            chord = p1 - p0
            ang = float(np.arctan2(chord[1], chord[0]))
            cen = (float(0.5*(p0[0]+p1[0])), float(0.5*(p0[1]+p1[1])))

            for eidx in eidx_path:
                edge_to_polyline[int(eidx)] = int(pid)

            polylines.append(dict(
                path_vertex_ids=[int(v) for v in vids_path],
                path_edge_indices=[int(i) for i in eidx_path],
                segment_lengths=[float(x) for x in segL.tolist()],
                total_length=float(L),
                equivalent_angle=float(ang),
                equivalent_center=(float(cen[0]), float(cen[1])),
            ))

        # Discretization: panels per polyline
        ne_half = int(ne_half)
        if ne_half < 2:
            ne_half = 2
        # choose number of panels per polyline proportional to total length
        Ls_tot = np.array([p["total_length"] for p in polylines], float)
        Lref = float(np.max(Ls_tot)) if np.max(Ls_tot) > 0 else 1.0

        poly_panels = []
        for pid, p in enumerate(polylines):
            L = float(p["total_length"])
            # base number of panels: 4*ne_half scaled by length ratio
            Np = max(8, int(round(4*ne_half * (L / Lref))))
            # tip_dense via cosine spacing in arc-length
            theta = np.linspace(0.0, math.pi, Np+1)
            s_nodes = 0.5*L*(1.0 - np.cos(theta))  # clusters near s=0 and s=L
            s_mid = 0.5*(s_nodes[:-1] + s_nodes[1:])
            ds = (s_nodes[1:] - s_nodes[:-1])

            # store collocation points (midpoints) and panel weights
            vids_path = p["path_vertex_ids"]
            seg_lengths = np.array(p["segment_lengths"], float)

            x_col = []
            t_col = []
            n_col = []
            for sm in s_mid:
                x, t, n = _polyline_point_and_frame(self.network, vids_path, seg_lengths, float(sm))
                x_col.append(x); t_col.append(t); n_col.append(n)
            x_col = np.array(x_col, float)
            t_col = np.array(t_col, float)
            n_col = np.array(n_col, float)

            # source quadrature points: Gauss points per panel
            nq = max(2, int(nq_stress))
            xg, wg = np.polynomial.legendre.leggauss(nq)
            src_pts = []
            src_t = []
            src_n = []
            src_w = []
            # per panel, integrate over s in [sL,sR]
            for k in range(Np):
                sL = float(s_nodes[k]); sR = float(s_nodes[k+1])
                J = 0.5*(sR-sL)
                sm = 0.5*(sR+sL)
                sq = sm + J*xg
                wq = J*wg
                for sqq, wqq in zip(sq, wq):
                    x, t, n = _polyline_point_and_frame(self.network, vids_path, seg_lengths, float(sqq))
                    src_pts.append(x); src_t.append(t); src_n.append(n); src_w.append(float(wqq))
            src_pts = np.array(src_pts, float)
            src_t = np.array(src_t, float)
            src_n = np.array(src_n, float)
            src_w = np.array(src_w, float)

            poly_panels.append(dict(
                pid=int(pid),
                Np=int(Np),
                s_nodes=s_nodes,
                s_mid=s_mid,
                ds=ds,
                x_col=x_col, t_col=t_col, n_col=n_col,
                src_pts=src_pts, src_t=src_t, src_n=src_n, src_w=src_w,
                nq=int(nq),
            ))

        # Build global collocation list across all polylines (enforce traction free on all)
        # Unknowns: per polyline, piecewise-constant bI and bII on each panel => 2*Np unknowns.
        offsets = []
        nunk = 0
        for pp in poly_panels:
            offsets.append(nunk)
            nunk += 2*int(pp["Np"])

        # collocation rows: tn and ts at each collocation point
        ncol_tot = sum(int(pp["Np"]) for pp in poly_panels)  # one collocation per panel midpoint
        K = np.zeros((2*ncol_tot, nunk), float)
        rhs = np.zeros((2*ncol_tot,), float)

        # material constants
        E = float(self.material.E)
        nu = float(self.material.nu)
        mu = E / (2.0*(1.0+nu))

        # remote traction on each crack face: computed as σ·n projected
        sig = np.array([[float(self.applied.sigma_xx), float(self.applied.sigma_xy)],
                        [float(self.applied.sigma_xy), float(self.applied.sigma_yy)]], float)

        # row assembly
        row0 = 0
        col0 = 0

        # Precompute for speed: list of all source points for all polylines and mapping from unknown index
        # We treat basis as constant per panel: bI_k and bII_k multiply all source quad points within panel k.
        # So for each panel k, we need list of source quad indices belonging to that panel.
        # Build per polyline panel -> source quad slice indices.
        for pid, pp in enumerate(poly_panels):
            Np = int(pp["Np"])
            nq = int(pp["nq"])
            # by construction, we appended nq points per panel in order
            panel_src = []
            idx = 0
            for k in range(Np):
                panel_src.append((idx, idx+nq))
                idx += nq
            pp["panel_src"] = panel_src

        # Collocation mapping
        # Each collocation point belongs to a polyline panel (one per panel midpoint).
        for pid, pp_i in enumerate(poly_panels):
            Npi = int(pp_i["Np"])
            for ic in range(Npi):
                x_i = pp_i["x_col"][ic]
                t_i = pp_i["t_col"][ic]
                n_i = pp_i["n_col"][ic]

                # rhs: - remote traction in local components (tn, ts)
                tr = sig @ n_i.reshape(2,)
                tn0 = float(np.dot(n_i, tr))
                ts0 = float(np.dot(t_i, tr))
                rhs[row0] = -tn0
                rhs[ncol_tot + row0] = -ts0

                # fill columns by looping over all source panels on all polylines
                for pid_j, pp_j in enumerate(poly_panels):
                    offj = offsets[pid_j]
                    Npj = int(pp_j["Np"])
                    # for each panel k, two unknowns: bI_k (normal/climb) and bII_k (tangent/glide)
                    for k in range(Npj):
                        # accumulate traction at x_i due to unit bI on panel k
                        s0, s1 = pp_j["panel_src"][k]
                        src_pts = pp_j["src_pts"][s0:s1]
                        src_t = pp_j["src_t"][s0:s1]
                        src_n = pp_j["src_n"][s0:s1]
                        wq = pp_j["src_w"][s0:s1]

                        # unit bI: dB = n_src * ds  (ds integrated via wq)
                        tn_sum_I = 0.0
                        ts_sum_I = 0.0
                        tn_sum_II = 0.0
                        ts_sum_II = 0.0

                        for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                            dx = float(x_i[0] - xs[0])
                            dy = float(x_i[1] - xs[1])

                            # Mode I (climb): Burgers along ns
                            dB = ns * float(ww)
                            sxx, syy, sxy = _stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
                            # traction on collocation normal
                            tx = float(sxx*n_i[0] + sxy*n_i[1])
                            ty = float(sxy*n_i[0] + syy*n_i[1])
                            tn_sum_I += float(n_i[0]*tx + n_i[1]*ty)
                            ts_sum_I += float(t_i[0]*tx + t_i[1]*ty)

                            # Mode II (glide): Burgers along ts
                            dB2 = ts * float(ww)
                            sxx, syy, sxy = _stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu)
                            tx = float(sxx*n_i[0] + sxy*n_i[1])
                            ty = float(sxy*n_i[0] + syy*n_i[1])
                            tn_sum_II += float(n_i[0]*tx + n_i[1]*ty)
                            ts_sum_II += float(t_i[0]*tx + t_i[1]*ty)

                        # place into K
                        col_I = offj + 2*k + 0
                        col_II = offj + 2*k + 1
                        K[row0, col_I] = tn_sum_I
                        K[ncol_tot + row0, col_I] = ts_sum_I
                        K[row0, col_II] = tn_sum_II
                        K[ncol_tot + row0, col_II] = ts_sum_II

                row0 += 1

        # Closure constraints: per polyline, ∫ bI ds = 0 and ∫ bII ds = 0
        Crows = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[pid]
            Np = int(pp["Np"])
            ds = np.asarray(pp["ds"], float).reshape(-1)
            # b is constant per panel, integral is sum b_k * ds_k
            rI = np.zeros((nunk,), float)
            rII = np.zeros((nunk,), float)
            for k in range(Np):
                rI[off + 2*k + 0] = float(ds[k])
                rII[off + 2*k + 1] = float(ds[k])
            Crows.append(rI)
            Crows.append(rII)
        C = np.vstack(Crows) if Crows else np.zeros((0, nunk), float)

        q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))

        # Unpack per polyline bI/bII per panel
        poly_solutions = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[pid]
            Np = int(pp["Np"])
            bI = np.array([q[off + 2*k + 0] for k in range(Np)], float)
            bII = np.array([q[off + 2*k + 1] for k in range(Np)], float)
            poly_solutions.append(dict(
                pid=int(pid),
                bI=bI,
                bII=bII,
                s_nodes=np.array(pp["s_nodes"], float),
                s_mid=np.array(pp["s_mid"], float),
                x_col=np.array(pp["x_col"], float),
                t_col=np.array(pp["t_col"], float),
                n_col=np.array(pp["n_col"], float),
                ds=np.array(pp["ds"], float),
            ))

        sol = dict(
            representation="panel",
            solver_option="parametrized_crack",
            parametrization="polyline",
            ne_half=int(ne_half),
            n_polylines=int(len(polylines)),
            parametrized_polylines=polylines,
            parametrized_edge_to_polyline=edge_to_polyline,
            displacement_branch_cut="polyline",
            polyline_solutions=poly_solutions,
            meta=dict(nq_stress=int(nq_stress), ridge=float(ridge)),
            constraints=dict(n_constraints=int(C.shape[0])),
        )
        return sol
