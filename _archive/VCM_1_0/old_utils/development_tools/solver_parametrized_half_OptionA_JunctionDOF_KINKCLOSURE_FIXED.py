"""
v4 parametrized polyline solver (true polyline integral operator) — utils_half variant.

Adds crack_mode="half":
- Builds one polyline per BRANCH (starting at deg>=2 vertex) and ending at deg!=2 vertex.
- Each half polyline gets two additional unknowns J0x,J0y (jump offset at s=0).
- Enforces:
  * Jump continuity at shared vertices (deg>=2): J is single-valued.
  * Tip closure at all deg==1 vertices: J(tip)=0.
  * Junction Burgers closure only for true junctions (deg>=3): sum ∫(bII t + bI n) ds = 0.
- Uses one-sided singular weighting at the tip end for half mode when representation="singular".
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import math

from .solver_material import Material, AppliedStress
from .solver_network import CrackNetworkV4
from .solver_kkt import solve_kkt_lsq

# kernels
from .solver_kernels_half import stress_edge_dislocation

# geometry helper (half) – must provide polyline_point_and_frame(network, vids_path, seg_lengths, s)
from .solver_polyline_half import polyline_point_and_frame


def _vertex_degrees(network: CrackNetworkV4) -> dict[int, int]:
    deg = {int(v.id): 0 for v in network.vertices}
    for e in network.edges:
        deg[int(e.v0)] = deg.get(int(e.v0), 0) + 1
        deg[int(e.v1)] = deg.get(int(e.v1), 0) + 1
    return deg


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
        crack_mode: str = "full",
        **_ignored,
    ) -> Dict:
        # accept legacy kwargs; keep signature compatible
        _ = (add_vertex_constraints, junction_option, r0_factor, n_int, _ignored)

        opt = str(solver_option).lower().strip()
        if opt not in ("parametrized_crack", "parameterized_crack", "param_crack"):
            raise ValueError("v4 solver supports solver_option='parametrized_crack' only.")
        if str(parametrization).lower().strip() not in ("polyline", "segmented", "kinked"):
            raise ValueError("v4 solver currently supports parametrization='polyline' only.")

        rep_in = str(representation).lower().strip()
        dist_in = str(node_distribution).lower().strip()

        # Chebyshev aliases (same pattern as your full solver)
        collocation_mode = "mid"
        if rep_in in ("cheb_quad", "cheb_spectral", "cheb_quad_singular", "cheb_spectral_singular"):
            if rep_in.startswith("cheb_spectral"):
                collocation_mode = "nodes"
            dist_in = "tip_dense"
            if rep_in.endswith("_singular"):
                rep_in = "singular"
            else:
                rep_in = "panel"

        if rep_in not in ("panel", "singular"):
            raise ValueError(
                f"Unknown representation={representation!r}. Use 'panel', 'singular', 'cheb_quad', or 'cheb_spectral'."
            )
        if dist_in not in ("uniform", "tip_dense"):
            raise ValueError(f"Unknown node_distribution={node_distribution!r}. Use 'uniform' or 'tip_dense'.")

        use_tip_singular = (rep_in == "singular")
        crack_mode = str(crack_mode).lower().strip()
        if crack_mode not in ("full", "half"):
            raise ValueError("crack_mode must be 'full' or 'half'.")

        ne_half = int(ne_half)
        if ne_half < 2:
            ne_half = 2

        # ------------------------------------------------------------
        # Build polylines + edge_to_polyline
        # ------------------------------------------------------------
        polylines: List[dict] = []
        edge_to_polyline: dict[int, int] = {}

        if crack_mode == "full":
            # Use your existing full polyline traversal utilities.
            # If your full utilities live in utils/ instead of utils_half, adjust imports accordingly.
            from .solver_polyline import find_polyline_components, path_order_for_component

            comps = find_polyline_components(self.network)
            if not comps:
                raise ValueError("No edges in network.")

            for pid, comp in enumerate(comps):
                vids_path, eidx_path = path_order_for_component(self.network, comp["edges"])
                pts = np.array([self.network.vertex_coords(v) for v in vids_path], float)
                seg = pts[1:] - pts[:-1]
                segL = np.sqrt(np.sum(seg * seg, axis=1))
                L = float(np.sum(segL))

                for eidx in eidx_path:
                    edge_to_polyline[int(eidx)] = int(pid)

                polylines.append(dict(
                    path_vertex_ids=[int(v) for v in vids_path],
                    path_edge_indices=[int(i) for i in eidx_path],
                    segment_lengths=[float(x) for x in segL.tolist()],
                    total_length=float(L),
                    crack_mode="full",
                    v_start=int(vids_path[0]),
                    v_end=int(vids_path[-1]),
                ))

        else:
            # HALF mode: branch polylines from deg>=2 vertices
            deg = _vertex_degrees(self.network)

            # vertex -> incident edges
            v2e: dict[int, List[int]] = {}
            for eidx, e in enumerate(self.network.edges):
                a = int(e.v0); b = int(e.v1)
                v2e.setdefault(a, []).append(int(eidx))
                v2e.setdefault(b, []).append(int(eidx))

            def other_vertex(eidx: int, v: int) -> int:
                e = self.network.edges[int(eidx)]
                a = int(e.v0); b = int(e.v1)
                return b if a == v else a

            used_edges = set()
            pid = 0

            start_verts = sorted([int(v) for v, d in deg.items() if int(d) >= 2])

            for v0 in start_verts:
                for e0 in v2e.get(v0, []):
                    if int(e0) in used_edges:
                        continue

                    vids_path = [int(v0)]
                    eidx_path: List[int] = []
                    cur_v = int(v0)
                    cur_e = int(e0)

                    while True:
                        if cur_e in used_edges:
                            break
                        used_edges.add(cur_e)
                        eidx_path.append(int(cur_e))

                        nxt_v = other_vertex(cur_e, cur_v)
                        vids_path.append(int(nxt_v))

                        dnext = int(deg.get(int(nxt_v), 0))
                        # stop if tip (1) or true junction (>=3) or isolated (0)
                        if dnext != 2:
                            break

                        # continue through degree-2 vertex
                        inc = [ei for ei in v2e.get(int(nxt_v), []) if int(ei) not in used_edges]
                        if not inc:
                            break
                        cur_v = int(nxt_v)
                        cur_e = int(inc[0])

                    pts = np.array([self.network.vertex_coords(v) for v in vids_path], float)
                    seg = pts[1:] - pts[:-1]
                    segL = np.sqrt(np.sum(seg * seg, axis=1))
                    L = float(np.sum(segL))
                    if L <= 0:
                        continue

                    # orient so START is open (deg>1) and END is tip (deg==1) when possible
                    v_start = int(vids_path[0])
                    v_end = int(vids_path[-1])
                    if int(deg.get(v_start, 0)) == 1 and int(deg.get(v_end, 0)) > 1:
                        vids_path = list(reversed(vids_path))
                        eidx_path = list(reversed(eidx_path))
                        segL = segL[::-1].copy()
                        v_start = int(vids_path[0])
                        v_end = int(vids_path[-1])

                    for eidx in eidx_path:
                        edge_to_polyline[int(eidx)] = int(pid)

                    polylines.append(dict(
                        path_vertex_ids=[int(v) for v in vids_path],
                        path_edge_indices=[int(i) for i in eidx_path],
                        segment_lengths=[float(x) for x in segL.tolist()],
                        total_length=float(L),
                        crack_mode="half",
                        v_start=int(v_start),
                        v_end=int(v_end),
                        deg_start=int(deg.get(v_start, 0)),
                        deg_end=int(deg.get(v_end, 0)),
                    ))
                    pid += 1

            # handle isolated single-edge component with no deg>=2 vertex
            for eidx, e in enumerate(self.network.edges):
                if int(eidx) in used_edges:
                    continue
                a = int(e.v0); b = int(e.v1)
                pa = np.array(self.network.vertex_coords(a), float)
                pb = np.array(self.network.vertex_coords(b), float)
                L = float(np.linalg.norm(pb - pa))
                if L <= 0:
                    continue
                vids_path = [a, b]
                if int(deg.get(a, 0)) == 1 and int(deg.get(b, 0)) > 1:
                    vids_path = [b, a]
                v_start = int(vids_path[0])
                v_end = int(vids_path[-1])
                edge_to_polyline[int(eidx)] = int(pid)
                polylines.append(dict(
                    path_vertex_ids=[int(v) for v in vids_path],
                    path_edge_indices=[int(eidx)],
                    segment_lengths=[float(L)],
                    total_length=float(L),
                    crack_mode="half",
                    v_start=int(v_start),
                    v_end=int(v_end),
                    deg_start=int(deg.get(v_start, 0)),
                    deg_end=int(deg.get(v_end, 0)),
                ))
                pid += 1

            if not polylines:
                raise ValueError("No valid half polylines were built.")

        # ------------------------------------------------------------
        # Discretize polylines and build operator
        # ------------------------------------------------------------
        poly_panels = []
        for pid, p in enumerate(polylines):
            vids_path = p["path_vertex_ids"]
            segL = np.array(p["segment_lengths"], float)
            L = float(p["total_length"])

            Np = max(8, int(2 * ne_half))

            if dist_in == "uniform":
                s_nodes = np.linspace(0.0, L, Np + 1)
            else:
                theta = np.linspace(0.0, math.pi, Np + 1)
                s_nodes = 0.5 * L * (1.0 - np.cos(theta))

            if collocation_mode == "nodes" and Np >= 2:
                s_col = s_nodes[1:-1].copy()
            else:
                s_col = 0.5 * (s_nodes[:-1] + s_nodes[1:])

            s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
            ds = (s_nodes[1:] - s_nodes[:-1])

            x_mid = []; t_mid = []; n_mid = []
            for sm in s_mid:
                x, t, n = polyline_point_and_frame(self.network, vids_path, segL, float(sm))
                x_mid.append(x); t_mid.append(t); n_mid.append(n)
            x_mid = np.array(x_mid, float)
            t_mid = np.array(t_mid, float)
            n_mid = np.array(n_mid, float)

            x_col = []; t_col = []; n_col = []
            for sc in s_col:
                x, t, n = polyline_point_and_frame(self.network, vids_path, segL, float(sc))
                x_col.append(x); t_col.append(t); n_col.append(n)
            x_col = np.array(x_col, float)
            t_col = np.array(t_col, float)
            n_col = np.array(n_col, float)

            nq = max(2, int(nq_stress))
            xg, wg = np.polynomial.legendre.leggauss(nq)

            src_pts = []; src_t = []; src_n = []; src_w = []
            panel_src = []
            panel_wsing = []
            eps_s = 1e-14 * L if L > 0 else 1e-14

            idx0 = 0
            for k in range(Np):
                sLk = float(s_nodes[k]); sRk = float(s_nodes[k + 1])
                Jk = 0.5 * (sRk - sLk)
                smk = 0.5 * (sRk + sLk)
                sq = smk + Jk * xg
                wq = Jk * wg

                wsing_k = 0.0
                for sqq, wqq in zip(sq, wq):
                    x, t, n = polyline_point_and_frame(self.network, vids_path, segL, float(sqq))
                    src_pts.append(x); src_t.append(t); src_n.append(n)

                    if use_tip_singular:
                        if crack_mode == "full":
                            ss = max(float(sqq), eps_s)
                            tt = max(float(L - float(sqq)), eps_s)
                            sing = 1.0 / math.sqrt(ss * tt)
                        else:
                            # half: tip intended at END (s=L) in our orientation
                            tt = max(float(L - float(sqq)), eps_s)
                            sing = 1.0 / math.sqrt(tt)
                    else:
                        sing = 1.0

                    w_eff = float(wqq) * sing
                    src_w.append(w_eff)
                    wsing_k += w_eff

                panel_wsing.append(wsing_k)
                panel_src.append((idx0, idx0 + nq))
                idx0 += nq

            poly_panels.append(dict(
                pid=int(pid),
                Np=int(Np),
                s_nodes=np.array(s_nodes, float),
                s_mid=np.array(s_mid, float),
                ds=np.array(ds, float),
                x_mid=x_mid, t_mid=t_mid, n_mid=n_mid,
                x_col=x_col, t_col=t_col, n_col=n_col,
                src_pts=np.array(src_pts, float),
                src_t=np.array(src_t, float),
                src_n=np.array(src_n, float),
                src_w=np.array(src_w, float),
                panel_src=panel_src,
                panel_wsing=np.array(panel_wsing, float),
                v_start=int(p.get("v_start", vids_path[0])),
                v_end=int(p.get("v_end", vids_path[-1])),
                L=float(L),
            ))

        # unknown offsets
        offsets = []
        nunk = 0

        # Allocate only density DOFs (bI,bII) per polyline first.
        for pp in poly_panels:
            offsets.append(nunk)
            nunk += 2 * int(pp["Np"])

        # In HALF mode, allocate shared junction jump DOFs J_v at polyline starts.
        # This removes duplicated per-branch J0 modes at junctions and avoids artificial pinning.
        if crack_mode == "half":
            junction_verts = sorted({int(pp["v_start"]) for pp in poly_panels})
            junction_dof = {}
            for i, vj in enumerate(junction_verts):
                junction_dof[int(vj)] = int(nunk + 2 * i)
            nunk += 2 * len(junction_verts)
        else:
            junction_dof = {}

        # operator matrix
        ncol_tot = sum(int(len(pp["x_col"])) for pp in poly_panels)
        K = np.zeros((2 * ncol_tot, nunk), float)
        rhs = np.zeros((2 * ncol_tot,), float)

        E = float(self.material.E)
        nu = float(self.material.nu)
        mu = E / (2.0 * (1.0 + nu))

        sig = np.array([
            [float(self.applied.sigma_xx), float(self.applied.sigma_xy)],
            [float(self.applied.sigma_xy), float(self.applied.sigma_yy)],
        ], float)

        row0 = 0
        for pid_i, pp_i in enumerate(poly_panels):
            Nci = int(len(pp_i["x_col"]))
            for ic in range(Nci):
                x_i = pp_i["x_col"][ic]
                t_i = pp_i["t_col"][ic]
                n_i = pp_i["n_col"][ic]

                tr = sig @ n_i.reshape(2,)
                tn0 = float(np.dot(n_i, tr))
                ts0 = float(np.dot(t_i, tr))
                rhs[row0] = -tn0
                rhs[ncol_tot + row0] = -ts0

                for pid_j, pp_j in enumerate(poly_panels):
                    offj = offsets[pid_j]
                    Npj = int(pp_j["Np"])
                    for k in range(Npj):
                        s0, s1 = pp_j["panel_src"][k]
                        src_pts = pp_j["src_pts"][s0:s1]
                        src_t = pp_j["src_t"][s0:s1]
                        src_n = pp_j["src_n"][s0:s1]
                        wq = pp_j["src_w"][s0:s1]

                        tn_sum_I = 0.0; ts_sum_I = 0.0
                        tn_sum_II = 0.0; ts_sum_II = 0.0

                        for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                            dx = float(x_i[0] - xs[0])
                            dy = float(x_i[1] - xs[1])

                            dB = ns * float(ww)
                            sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
                            tx = float(sxx * n_i[0] + sxy * n_i[1])
                            ty = float(sxy * n_i[0] + syy * n_i[1])
                            tn_sum_I += float(n_i[0] * tx + n_i[1] * ty)
                            ts_sum_I += float(t_i[0] * tx + t_i[1] * ty)

                            dB2 = ts * float(ww)
                            sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu)
                            tx = float(sxx * n_i[0] + sxy * n_i[1])
                            ty = float(sxy * n_i[0] + syy * n_i[1])
                            tn_sum_II += float(n_i[0] * tx + n_i[1] * ty)
                            ts_sum_II += float(t_i[0] * tx + t_i[1] * ty)

                        col_I = offj + 2 * k + 0
                        col_II = offj + 2 * k + 1
                        K[row0, col_I] = tn_sum_I
                        K[ncol_tot + row0, col_I] = ts_sum_I
                        K[row0, col_II] = tn_sum_II
                        K[ncol_tot + row0, col_II] = ts_sum_II

                row0 += 1

        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------
        Crows = []

        if crack_mode == "full":
            # preserve baseline closure per polyline
            for pid, pp in enumerate(poly_panels):
                off = offsets[pid]
                Np = int(pp["Np"])
                ds = np.asarray(pp["ds"], float).reshape(-1)
                rI = np.zeros((nunk,), float)
                rII = np.zeros((nunk,), float)
                for k in range(Np):
                    wC = float(pp["panel_wsing"][k]) if use_tip_singular else float(ds[k])
                    rI[off + 2 * k + 0] = wC
                    rII[off + 2 * k + 1] = wC
                Crows.append(rI); Crows.append(rII)

        else:
            deg = _vertex_degrees(self.network)

            def row_for_J_at(pid: int, which: str, comp: int) -> np.ndarray:
                pp = poly_panels[pid]
                off = offsets[pid]
                Np = int(pp["Np"])
                ds = np.asarray(pp["ds"], float).reshape(-1)
                tmid = np.asarray(pp["t_mid"], float)
                nmid = np.asarray(pp["n_mid"], float)

                r = np.zeros((nunk,), float)

                if crack_mode == "half":
                    joff = junction_dof[int(pp["v_start"])]
                    r[joff + 0] = 1.0 if comp == 0 else 0.0
                    r[joff + 1] = 1.0 if comp == 1 else 0.0

                if which == "start":
                    return r

                for k in range(Np):
                    tx, ty = float(tmid[k, 0]), float(tmid[k, 1])
                    nx, ny = float(nmid[k, 0]), float(nmid[k, 1])
                    w = float(ds[k])
                    r[off + 2 * k + 0] += -(nx if comp == 0 else ny) * w
                    r[off + 2 * k + 1] += -(tx if comp == 0 else ty) * w
                return r

                for k in range(Np):
                    tx, ty = float(tmid[k, 0]), float(tmid[k, 1])
                    nx, ny = float(nmid[k, 0]), float(nmid[k, 1])
                    w = float(ds[k])
                    r[off + 2 * k + 0] += -(nx if comp == 0 else ny) * w
                    r[off + 2 * k + 1] += -(tx if comp == 0 else ty) * w
                return r

            # vertex incidence: v -> list[(pid, "start"/"end")]
            v_inc: dict[int, List[tuple[int, str]]] = {}
            for pid, pp in enumerate(poly_panels):
                vs = int(pp["v_start"]); ve = int(pp["v_end"])
                v_inc.setdefault(vs, []).append((pid, "start"))
                v_inc.setdefault(ve, []).append((pid, "end"))

            # continuity at all deg>=2 vertices
            for v, items in v_inc.items():
                if int(deg.get(int(v), 0)) < 2 or len(items) < 2:
                    continue
                pid_ref, w_ref = items[0]
                for (pid_j, w_j) in items[1:]:
                    for comp in (0, 1):
                        rr = row_for_J_at(pid_j, w_j, comp) - row_for_J_at(pid_ref, w_ref, comp)
                        if np.linalg.norm(rr) > 0:
                            Crows.append(rr)

            # Dipole neutrality (junction Burgers closure).
            # - For a true junction (deg>=3): enforce vector closure  Σ_j ∫ (bII*t + bI*n) ds = 0  (x and y components).
            # - For a degree-2 vertex: enforcing full vector closure with continuity+tip-closure can algebraically pin J(v).
            #   We enforce ONE projected closure along the bisector normal at deg==2 (skips near-collinear cases).
            theta_min_flat = float(_ignored.get("theta_min_flat", 5.0 * math.pi / 180.0))
            if not (0.0 < theta_min_flat < math.pi):
                theta_min_flat = 5.0 * math.pi / 180.0

            def _outgoing_tangent_at_vertex(pid: int, which: str) -> np.ndarray:
                pp = poly_panels[pid]
                tmid = np.asarray(pp["t_mid"], float)
                if tmid.size == 0:
                    return np.array([1.0, 0.0], float)
                if which == "start":
                    t = tmid[0].copy()
                else:
                    t = -tmid[-1].copy()
                nrm = float(np.hypot(t[0], t[1]))
                if nrm <= 0:
                    return np.array([1.0, 0.0], float)
                return t / nrm

            def _add_vector_closure(items):
                for comp in (0, 1):
                    r = np.zeros((nunk,), float)
                    for (pid_j, _which) in items:
                        ppj = poly_panels[pid_j]
                        offj = offsets[pid_j]
                        Npj = int(ppj["Np"])
                        dsj = np.asarray(ppj["ds"], float).reshape(-1)
                        tmidj = np.asarray(ppj["t_mid"], float)
                        nmidj = np.asarray(ppj["n_mid"], float)
                        for k in range(Npj):
                            w = float(dsj[k])
                            tx, ty = float(tmidj[k, 0]), float(tmidj[k, 1])
                            nx, ny = float(nmidj[k, 0]), float(nmidj[k, 1])
                            r[offj + 2 * k + 0] += (nx if comp == 0 else ny) * w
                            r[offj + 2 * k + 1] += (tx if comp == 0 else ty) * w
                    Crows.append(r)

            def _add_projected_closure(items, proj: np.ndarray):
                proj = np.asarray(proj, float).reshape(2,)
                r = np.zeros((nunk,), float)
                for (pid_j, _which) in items:
                    ppj = poly_panels[pid_j]
                    offj = offsets[pid_j]
                    Npj = int(ppj["Np"])
                    dsj = np.asarray(ppj["ds"], float).reshape(-1)
                    tmidj = np.asarray(ppj["t_mid"], float)
                    nmidj = np.asarray(ppj["n_mid"], float)
                    for k in range(Npj):
                        w = float(dsj[k])
                        tx, ty = float(tmidj[k, 0]), float(tmidj[k, 1])
                        nx, ny = float(nmidj[k, 0]), float(nmidj[k, 1])
                        r[offj + 2 * k + 0] += float(proj[0] * nx + proj[1] * ny) * w
                        r[offj + 2 * k + 1] += float(proj[0] * tx + proj[1] * ty) * w
                Crows.append(r)

            for v, items in v_inc.items():
                dv = int(deg.get(int(v), 0))
                if dv < 2 or len(items) < 2:
                    continue


            if dv >= 3:
                # True junction: enforce vector dipole neutrality (x and y components)
                _add_vector_closure(items)
            elif dv == 2 and len(items) == 2:
                # Degree-2 "junction" (kink): enforce ONE projected closure to remove the remaining monopole mode
                # without algebraically pinning the kink. Projection is the bisector normal.
                theta_min_flat = float(_ignored.get("theta_min_flat", 5.0 * math.pi / 180.0))
                if not (0.0 < theta_min_flat < math.pi):
                    theta_min_flat = 5.0 * math.pi / 180.0

                (pid0, w0), (pid1, w1) = items[0], items[1]
                t0 = _outgoing_tangent_at_vertex(pid0, w0)
                t1 = _outgoing_tangent_at_vertex(pid1, w1)
                c = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
                theta = float(np.arccos(c))
                # skip near-collinear (flat split) cases
                if theta < theta_min_flat or abs(theta - math.pi) < theta_min_flat:
                    pass
                else:
                    b = t0 + t1
                    nb = float(np.hypot(b[0], b[1]))
                    if nb > 0:
                        b = b / nb
                        proj = np.array([-b[1], b[0]], float)  # bisector normal
                        _add_projected_closure(items, proj)

                    _add_vector_closure(items)
                else:
                    (pid0, w0), (pid1, w1) = items[0], items[1]
                    t0 = _outgoing_tangent_at_vertex(pid0, w0)
                    t1 = _outgoing_tangent_at_vertex(pid1, w1)
                    c = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
                    theta = float(np.arccos(c))
                    if theta < theta_min_flat or abs(theta - math.pi) < theta_min_flat:
                        continue  # flat split: no closure at this vertex
                    b = t0 + t1
                    nb = float(np.hypot(b[0], b[1]))
                    if nb <= 0:
                        continue
                    b = b / nb
                    proj = np.array([-b[1], b[0]], float)  # bisector normal
                    _add_projected_closure(items, proj)

            # Global gauge (Option A): fix jump reference at ONE degree-1 tip.
            # This removes the rigid jump mode without overconstraining multi-tip networks.
            ref_tip = None
            for v_tip, d in deg.items():
                if int(d) == 1:
                    ref_tip = int(v_tip)
                    break
            if ref_tip is not None:
                items = v_inc.get(int(ref_tip), [])
                if items:
                    pid_t, which_t = items[0]
                    for comp in (0, 1):
                        Crows.append(row_for_J_at(pid_t, which_t, comp))

        C = np.vstack(Crows) if Crows else np.zeros((0, nunk), float)

        q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))
        # NOTE: if your solve_kkt_lsq signature is solve_kkt_lsq(K, rhs, C, ridge), replace line above with:
        # q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))

        # ------------------------------------------------------------
        # Pack solution
        # ------------------------------------------------------------
        poly_solutions = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[pid]
            Np = int(pp["Np"])
            bI_hat = np.array([q[off + 2 * k + 0] for k in range(Np)], float)
            bII_hat = np.array([q[off + 2 * k + 1] for k in range(Np)], float)

            if use_tip_singular:
                s_mid = np.array(pp["s_mid"], float)
                Lp = float(pp["s_nodes"][-1])
                eps = 1e-14 * Lp if Lp > 0 else 1e-14
                if crack_mode == "full":
                    sing_mid = 1.0 / np.sqrt(np.maximum(s_mid, eps) * np.maximum(Lp - s_mid, eps))
                else:
                    sing_mid = 1.0 / np.sqrt(np.maximum(Lp - s_mid, eps))
                bI = bI_hat * sing_mid
                bII = bII_hat * sing_mid
            else:
                bI = bI_hat.copy()
                bII = bII_hat.copy()

            solp = dict(
                pid=int(pid),
                bI=bI, bII=bII,
                bI_hat=bI_hat, bII_hat=bII_hat,
                s_nodes=np.array(pp["s_nodes"], float),
                s_mid=np.array(pp["s_mid"], float),
                x_mid=np.array(pp["x_mid"], float),
                t_mid=np.array(pp["t_mid"], float),
                n_mid=np.array(pp["n_mid"], float),
                x_col=np.array(pp["x_col"], float),
                t_col=np.array(pp["t_col"], float),
                n_col=np.array(pp["n_col"], float),
                ds=np.array(pp["ds"], float),
                v_start=int(pp["v_start"]),
                v_end=int(pp["v_end"]),
            )

            if crack_mode == "half":
                joff = junction_dof[int(pp["v_start"])]
                solp["J0"] = np.array([float(q[joff + 0]), float(q[joff + 1])], float)

            poly_solutions.append(solp)

        sol = dict(
            representation=rep_in,
            solver_option="parametrized_crack",
            parametrization="polyline",
            ne_half=int(ne_half),
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            crack_mode=crack_mode,
            n_polylines=int(len(polylines)),
            parametrized_polylines=polylines,
            parametrized_edge_to_polyline=edge_to_polyline,
            polyline_solutions=poly_solutions,
            meta=dict(nq_stress=int(nq_stress), ridge=float(ridge), ndof=int(nunk)),
            constraints=dict(n_constraints=int(C.shape[0])),
        )
        return sol