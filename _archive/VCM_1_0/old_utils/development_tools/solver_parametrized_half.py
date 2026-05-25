"""v4 parametrized polyline solver (true polyline integral operator)."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import math

from .solver_material import Material, AppliedStress
from .solver_network import CrackNetworkV4
from .solver_kkt import solve_kkt_lsq
from .solver_polyline_half import find_polyline_components, path_order_for_component, polyline_point_and_frame, vertex_degrees, half_polylines_from_edges
from .solver_kernels_half import stress_edge_dislocation


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
        crack_mode: str = "full",  # 'full' (default) or 'half'
        add_gauge: bool = True,
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

        mode = str(crack_mode).lower().strip()
        if mode not in ("full", "half"):
            raise ValueError("crack_mode must be 'full' or 'half'.")

        # --- decouple density representation from node distribution
        rep_in = str(representation).lower().strip()
        dist_in = str(node_distribution).lower().strip()

        # Backward/alias support for legacy Chebyshev naming
        # cheb_quad / cheb_spectral select Chebyshev-Lobatto (tip-dense) spacing.
        # cheb_spectral additionally uses interior nodes for collocation.
        collocation_mode = "mid"  # "mid" (panel midpoints) or "nodes" (interior nodes)

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

        # Polyline components:
        # - full: existing behavior (each connected component must be a simple chain)
        # - half: treat each edge as an independent branch polyline (junctions supported via constraints)
        if mode == "half":
            comps = half_polylines_from_edges(self.network)
        else:
            comps = find_polyline_components(self.network)
        if not comps:
            raise ValueError("No edges in network.")

        deg_global = vertex_degrees(self.network)

        polylines = []
        edge_to_polyline = {}

        for pid, comp in enumerate(comps):
            vids_path, eidx_path = path_order_for_component(self.network, comp["edges"])
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

            # Determine which end(s) are physical tips (for half-crack one-sided singular weighting)
            v0_id = int(vids_path[0]); v1_id = int(vids_path[-1])
            d0 = int(deg_global.get(v0_id, 0)); d1 = int(deg_global.get(v1_id, 0))
            if mode == "half":
                if d0 == 1 and d1 != 1:
                    singular_end = "start"
                elif d1 == 1 and d0 != 1:
                    singular_end = "end"
                elif d0 == 1 and d1 == 1:
                    singular_end = "both"  # isolated edge
                else:
                    singular_end = "none"  # between junctions
            else:
                singular_end = "both"

            polylines.append(dict(
                singular_end=str(singular_end),
                end_vertex_ids=(int(v0_id), int(v1_id)),
                path_vertex_ids=[int(v) for v in vids_path],
                path_edge_indices=[int(i) for i in eidx_path],
                segment_lengths=[float(x) for x in segL.tolist()],
                total_length=float(L),
                equivalent_angle=float(ang),
                equivalent_center=(float(cen[0]), float(cen[1])),
            ))

        ne_half = int(ne_half)
        if ne_half < 2:
            ne_half = 2

        Ls_tot = np.array([p["total_length"] for p in polylines], float)
        Lref = float(np.max(Ls_tot)) if np.max(Ls_tot) > 0 else 1.0

        poly_panels = []
        for pid, p in enumerate(polylines):
            L = float(p["total_length"])
            Np = max(8, int(2*ne_half))

            if dist_in == "uniform":

                s_nodes = np.linspace(0.0, L, Np+1)

            else:

                if mode == "half":
                    # One-sided clustering toward the free tip end.
                    # Use a simple polynomial grading that concentrates nodes near eta=1.
                    eta = np.linspace(0.0, 1.0, Np+1)
                    pgrade = 3.0
                    u = 1.0 - np.power(1.0 - eta, pgrade)
                    s_nodes = L * u
                else:
                    # Chebyshev-Lobatto / cosine clustering (dense near both tips)
                    theta = np.linspace(0.0, math.pi, Np+1)
                    s_nodes = 0.5*L*(1.0 - np.cos(theta))

            

            # Collocation points along the polyline parameter s

            if collocation_mode == "nodes" and Np >= 2:

                s_col = s_nodes[1:-1].copy()

            else:

                s_col = 0.5*(s_nodes[:-1] + s_nodes[1:])

            

            s_mid = 0.5*(s_nodes[:-1] + s_nodes[1:])

            ds = (s_nodes[1:] - s_nodes[:-1])

            vids_path = p["path_vertex_ids"]
            seg_lengths = np.array(p["segment_lengths"], float)
            x_mid = []
            t_mid = []
            n_mid = []
            for sm in s_mid:
                x, t, n = polyline_point_and_frame(self.network, vids_path, seg_lengths, float(sm))
                x_mid.append(x); t_mid.append(t); n_mid.append(n)
            x_mid = np.array(x_mid, float)
            t_mid = np.array(t_mid, float)
            n_mid = np.array(n_mid, float)


            x_col = []
            t_col = []
            n_col = []
            for sm in s_col:
                x, t, n = polyline_point_and_frame(self.network, vids_path, seg_lengths, float(sm))
                x_col.append(x); t_col.append(t); n_col.append(n)
            x_col = np.array(x_col, float)
            t_col = np.array(t_col, float)
            n_col = np.array(n_col, float)

            nq = max(2, int(nq_stress))
            xg, wg = np.polynomial.legendre.leggauss(nq)
            src_pts = []
            src_t = []
            src_n = []
            src_w = []
            panel_wsing = []  # per-panel integral weights (for constraints when singular)
            eps_s = 1e-14 * L
            for k in range(Np):
                sL = float(s_nodes[k]); sR = float(s_nodes[k+1])
                J = 0.5*(sR-sL)
                sm = 0.5*(sR+sL)
                sq = sm + J*xg
                wq = J*wg
                wsing_k = 0.0
                for sqq, wqq in zip(sq, wq):
                    x, t, n = polyline_point_and_frame(self.network, vids_path, seg_lengths, float(sqq))
                    src_pts.append(x); src_t.append(t); src_n.append(n)
                    if use_tip_singular:
                        ss = max(float(sqq), eps_s)
                        tt = max(float(L - float(sqq)), eps_s)
                        if mode == "half":
                            se = str(p.get('singular_end','end'))
                            if se == "end":
                                sing = 1.0 / math.sqrt(tt)
                            elif se == "start":
                                sing = 1.0 / math.sqrt(ss)
                            elif se == "both":
                                sing = 1.0 / math.sqrt(ss * tt)
                            else:
                                sing = 1.0
                        else:
                            sing = 1.0 / math.sqrt(ss * tt)
                    else:
                        sing = 1.0
                    w_eff = float(wqq) * sing
                    src_w.append(w_eff)
                    wsing_k += w_eff
                panel_wsing.append(wsing_k)
            src_pts = np.array(src_pts, float)
            src_t = np.array(src_t, float)
            src_n = np.array(src_n, float)
            src_w = np.array(src_w, float)

            panel_src = []
            idx = 0
            for k in range(Np):
                panel_src.append((idx, idx+nq))
                idx += nq

            poly_panels.append(dict(
                singular_end=str(p.get('singular_end','both')),
                end_vertex_ids=tuple(p.get('end_vertex_ids',(int(vids_path[0]), int(vids_path[-1])))),
                pid=int(pid),
                Np=int(Np),
                s_nodes=s_nodes,
                s_mid=s_mid,
                ds=ds,
                x_mid=x_mid, t_mid=t_mid, n_mid=n_mid,
                x_col=x_col, t_col=t_col, n_col=n_col,
                src_pts=src_pts, src_t=src_t, src_n=src_n, src_w=src_w,
                nq=int(nq),
                panel_src=panel_src,
                panel_wsing=np.array(panel_wsing, float),
            ))

        offsets = []
        nunk = 0
        for pp in poly_panels:
            offsets.append(nunk)
            nunk += 2*int(pp["Np"])

        ncol_tot = sum(int(len(pp["x_col"])) for pp in poly_panels)
        K = np.zeros((2*ncol_tot, nunk), float)
        rhs = np.zeros((2*ncol_tot,), float)

        E = float(self.material.E)
        nu = float(self.material.nu)
        mu = E / (2.0*(1.0+nu))

        sig = np.array([[float(self.applied.sigma_xx), float(self.applied.sigma_xy)],
                        [float(self.applied.sigma_xy), float(self.applied.sigma_yy)]], float)

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

                        tn_sum_I = 0.0
                        ts_sum_I = 0.0
                        tn_sum_II = 0.0
                        ts_sum_II = 0.0

                        for (xs, ts, ns, ww) in zip(src_pts, src_t, src_n, wq):
                            dx = float(x_i[0] - xs[0])
                            dy = float(x_i[1] - xs[1])

                            dB = ns * float(ww)
                            sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
                            tx = float(sxx*n_i[0] + sxy*n_i[1])
                            ty = float(sxy*n_i[0] + syy*n_i[1])
                            tn_sum_I += float(n_i[0]*tx + n_i[1]*ty)
                            ts_sum_I += float(t_i[0]*tx + t_i[1]*ty)

                            dB2 = ts * float(ww)
                            sxx, syy, sxy = stress_edge_dislocation(dx, dy, float(dB2[0]), float(dB2[1]), mu, nu)
                            tx = float(sxx*n_i[0] + sxy*n_i[1])
                            ty = float(sxy*n_i[0] + syy*n_i[1])
                            tn_sum_II += float(n_i[0]*tx + n_i[1]*ty)
                            ts_sum_II += float(t_i[0]*tx + t_i[1]*ty)

                        col_I = offj + 2*k + 0
                        col_II = offj + 2*k + 1
                        K[row0, col_I] = tn_sum_I
                        K[ncol_tot + row0, col_I] = ts_sum_I
                        K[row0, col_II] = tn_sum_II
                        K[ncol_tot + row0, col_II] = ts_sum_II

                row0 += 1


        # -------------------------
        # Constraints
        # -------------------------
        Crows = []

        if mode == "half":
            # (1) Junction closure: for each junction vertex, enforce zero net displacement-jump injection:
            #     sum_{branches incident to v} ∫ (bII * t + bI * n) ds = 0  (vector; 2 scalar constraints).
            # Use panel-midpoint frames (t_mid/n_mid) and panel measures (ds or panel_wsing) consistently.
            # A junction is identified as a vertex with degree >= 3 in the global network.
            junction_vids = sorted([int(v) for v, d in deg_global.items() if int(d) >= 3])

            # Map junction -> incident polyline ids
            v2p = {vid: [] for vid in junction_vids}
            for pid, pp in enumerate(poly_panels):
                v0, v1 = tuple(pp.get("end_vertex_ids", (None, None)))
                if v0 in v2p: v2p[v0].append(pid)
                if v1 in v2p: v2p[v1].append(pid)

            for vid in junction_vids:
                pids = v2p.get(vid, [])
                if not pids:
                    continue

                row_x = np.zeros((nunk,), float)
                row_y = np.zeros((nunk,), float)

                for pid in pids:
                    pp = poly_panels[pid]
                    off = offsets[pid]
                    Np = int(pp["Np"])
                    ds = np.asarray(pp["ds"], float).reshape(-1)
                    t_mid = np.asarray(pp["t_mid"], float)
                    n_mid = np.asarray(pp["n_mid"], float)

                    # choose weights consistent with singular option
                    if use_tip_singular:
                        w_panel = np.asarray(pp["panel_wsing"], float).reshape(-1)
                    else:
                        w_panel = ds

                    for k in range(Np):
                        wC = float(w_panel[k])
                        # dJ = -(bII t + bI n) ds  -> closure wants sum(dJ)=0, sign irrelevant
                        # Coefficients for bI and bII in global components:
                        col_I = off + 2*k + 0
                        col_II = off + 2*k + 1
                        row_x[col_I]  += float(n_mid[k,0]) * wC
                        row_x[col_II] += float(t_mid[k,0]) * wC
                        row_y[col_I]  += float(n_mid[k,1]) * wC
                        row_y[col_II] += float(t_mid[k,1]) * wC

                Crows.append(row_x)
                Crows.append(row_y)

            # (2) One global gauge (optional): enforce zero normal-opening at one reference free tip.
            if add_gauge:
                # pick first degree-1 vertex and its incident polyline
                tip_vids = [int(v) for v, d in deg_global.items() if int(d) == 1]
                if tip_vids:
                    vref = int(tip_vids[0])
                    # find first polyline touching this vertex
                    pref = None
                    tip_at_end = True
                    for pid, pp in enumerate(poly_panels):
                        v0, v1 = tuple(pp.get("end_vertex_ids", (None, None)))
                        if v0 == vref:
                            pref = pid; tip_at_end = False; break
                        if v1 == vref:
                            pref = pid; tip_at_end = True; break
                    if pref is not None:
                        pmeta = polylines[pref]
                        Lp = float(pmeta.get("total_length", 0.0))
                        # normal at the tip point
                        vids_path = list(pmeta.get("path_vertex_ids", []))
                        seg_lengths = np.array(pmeta.get("segment_lengths", []), float)
                        s_tip = float(Lp if tip_at_end else 0.0)
                        _, _, n_tip = polyline_point_and_frame(self.network, vids_path, seg_lengths, s_tip)

                        pp = poly_panels[pref]
                        off = offsets[pref]
                        Np = int(pp["Np"])
                        ds = np.asarray(pp["ds"], float).reshape(-1)
                        t_mid = np.asarray(pp["t_mid"], float)
                        n_mid = np.asarray(pp["n_mid"], float)
                        if use_tip_singular:
                            w_panel = np.asarray(pp["panel_wsing"], float).reshape(-1)
                        else:
                            w_panel = ds

                        row_g = np.zeros((nunk,), float)
                        for k in range(Np):
                            wC = float(w_panel[k])
                            col_I = off + 2*k + 0
                            col_II = off + 2*k + 1
                            row_g[col_I]  += float(np.dot(n_mid[k], n_tip)) * wC
                            row_g[col_II] += float(np.dot(t_mid[k], n_tip)) * wC
                        Crows.append(row_g)

        else:
            # Existing full-crack constraints (unchanged behavior): per-polyline integrals of bI and bII.
            for pid, pp in enumerate(poly_panels):
                off = offsets[pid]
                Np = int(pp["Np"])
                ds = np.asarray(pp["ds"], float).reshape(-1)
                rI = np.zeros((nunk,), float)
                rII = np.zeros((nunk,), float)
                for k in range(Np):
                    wC = float(pp["panel_wsing"][k]) if use_tip_singular else float(ds[k])
                    rI[off + 2*k + 0] = wC
                    rII[off + 2*k + 1] = wC
                Crows.append(rI)
                Crows.append(rII)

        C = np.vstack(Crows) if Crows else np.zeros((0, nunk), float)

        q = solve_kkt_lsq(K, rhs, C, ridge=float(ridge))

        poly_solutions = []
        for pid, pp in enumerate(poly_panels):
            off = offsets[pid]
            Np = int(pp["Np"])
            bI_hat = np.array([q[off + 2*k + 0] for k in range(Np)], float)
            bII_hat = np.array([q[off + 2*k + 1] for k in range(Np)], float)

            if use_tip_singular:
                s_mid = np.array(pp["s_mid"], float)
                Lp = float(np.asarray(pp["s_nodes"], float)[-1])   # last node is the polyline length L
                eps = 1e-14 * Lp
                ss = np.maximum(s_mid, eps)
                tt = np.maximum(Lp - s_mid, eps)
                if mode == "half":
                    se = str(pp.get('singular_end','end'))
                    if se == 'end':
                        sing_mid = 1.0 / np.sqrt(tt)
                    elif se == 'start':
                        sing_mid = 1.0 / np.sqrt(ss)
                    elif se == 'both':
                        sing_mid = 1.0 / np.sqrt(ss * tt)
                    else:
                        sing_mid = np.ones_like(s_mid)
                else:
                    sing_mid = 1.0 / np.sqrt(ss * tt)
                bI = bI_hat * sing_mid
                bII = bII_hat * sing_mid
            else:
                bI = bI_hat.copy()
                bII = bII_hat.copy()

            poly_solutions.append(dict(
                pid=int(pid),
                bI=bI,
                bII=bII,
                bI_hat=bI_hat,
                bII_hat=bII_hat,
                s_nodes=np.array(pp["s_nodes"], float),
                s_mid=np.array(pp["s_mid"], float),
                x_mid=np.array(pp.get("x_mid", pp.get("x_col")), float),
                t_mid=np.array(pp.get("t_mid", pp.get("t_col")), float),
                n_mid=np.array(pp.get("n_mid", pp.get("n_col")), float),
                x_col=np.array(pp["x_col"], float),
                t_col=np.array(pp["t_col"], float),
                n_col=np.array(pp["n_col"], float),
                ds=np.array(pp["ds"], float),
            ))

        sol = dict(
            representation=rep_in,
            solver_option="parametrized_crack",
            parametrization="polyline",
            ne_half=int(ne_half),
            node_distribution=dist_in,
            collocation_mode=collocation_mode,
            n_polylines=int(len(polylines)),
            parametrized_polylines=polylines,
            parametrized_edge_to_polyline=edge_to_polyline,
            displacement_branch_cut="polyline",
            polyline_solutions=poly_solutions,
            meta=dict(nq_stress=int(nq_stress), ridge=float(ridge), ndof=int(nunk), n_panels_per_polyline=int(max(8, int(2*ne_half))), use_tip_singular=bool(use_tip_singular), node_distribution=dist_in, collocation_mode=collocation_mode),
            constraints=dict(n_constraints=int(C.shape[0])),
        )
        return sol