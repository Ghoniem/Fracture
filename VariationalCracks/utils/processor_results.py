"""Results object for v4 polyline-panel solver."""
from __future__ import annotations
import numpy as np
import math

from .processor_kernels import applied_tensor, edge_dislocation_u, stress_edge_dislocation


class DCEResultsNetworkV4:
    def __init__(self, calc, sol: dict):
        self.calc = calc
        self.sol = sol

    # -------------------------
    # Compatibility hooks used by plotter
    # -------------------------
    def is_parametrized(self) -> bool:
        return str(self.sol.get("solver_option", "")).lower() == "parametrized_crack"

    def reconstruct_cod_csd_parametrized(
        self,
        edge_index: int,
        n_pts: int = 4000,
        enforce_global_tip_zero: bool = True,
    ):
        edge_index = int(edge_index)
        if not self.is_parametrized():
            raise AttributeError("This results object is intended for solver_option='parametrized_crack'.")

        e2p = dict(self.sol.get("parametrized_edge_to_polyline", {}))
        if edge_index not in e2p:
            raise ValueError(f"Edge {edge_index} is not mapped to any polyline in solution metadata.")
        pid = int(e2p[edge_index])

        polylines = list(self.sol.get("parametrized_polylines", []))
        poly_solutions = list(self.sol.get("polyline_solutions", []))
        if pid < 0 or pid >= len(polylines) or pid >= len(poly_solutions):
            raise ValueError("polyline metadata/solution missing or inconsistent.")

        meta = dict(polylines[pid])
        solp = dict(poly_solutions[pid])

        path_vids = [int(v) for v in meta.get("path_vertex_ids", [])]
        path_edges = [int(i) for i in meta.get("path_edge_indices", [])]
        segL = np.asarray(meta.get("segment_lengths", []), float)
        Ltot = float(meta.get("total_length", np.sum(segL)))

        if edge_index not in path_edges:
            raise ValueError(f"Edge {edge_index} is not in polyline path for pid={pid}.")

        s_vert = np.zeros(len(path_vids), float)
        if len(segL) == len(path_vids) - 1:
            s_vert[1:] = np.cumsum(segL)

        k = path_edges.index(edge_index)
        s0 = float(s_vert[k])
        Le = float(segL[k])
        a = 0.5 * Le

        edge = self.calc.network.edges[edge_index]
        v_start = int(path_vids[k])
        v_end = int(path_vids[k + 1])
        if int(edge.v0) == v_start and int(edge.v1) == v_end:
            dir_sign = +1
        elif int(edge.v1) == v_start and int(edge.v0) == v_end:
            dir_sign = -1
        else:
            dir_sign = +1

        n_pts = int(max(200, n_pts))
        x_edge = np.linspace(-a, a, n_pts)

        if dir_sign == +1:
            s = s0 + (x_edge + a)
        else:
            s = s0 + (a - x_edge)

        s_nodes = np.asarray(solp.get("s_nodes"), float)
        bI  = np.asarray(solp.get("bI"), float)
        bII = np.asarray(solp.get("bII"), float)

        # Collocation frames (may be Nc=Np or Nc=Np-1 for cheb_spectral)
        t_col = np.asarray(solp.get("t_col"), float)
        n_col = np.asarray(solp.get("n_col"), float)

        # Panel-midpoint frames (length Np) — required for panel-midpoint reconstruction
        t_mid = np.asarray(solp.get("t_mid"), float) if solp.get("t_mid") is not None else None
        n_mid = np.asarray(solp.get("n_mid"), float) if solp.get("n_mid") is not None else None

        ds = np.asarray(solp.get("ds"), float)
        Np = int(solp.get("Np", len(bI)))

        # Prefer midpoint frames; fallback to collocation frames only if they match panel count
        if t_mid is not None and n_mid is not None and len(t_mid) == Np and len(n_mid) == Np:
            t_use, n_use = t_mid, n_mid
        elif len(t_col) == Np and len(n_col) == Np:
            t_use, n_use = t_col, n_col
        else:
            raise ValueError(
                f"Panel-midpoint reconstruction requires frames of length Np={Np}. "
                f"Got len(t_mid)={None if t_mid is None else len(t_mid)}, len(n_mid)={None if n_mid is None else len(n_mid)}, "
                f"len(t_col)={len(t_col)}, len(n_col)={len(n_col)}. "
                "For cheb_spectral, ensure solver returns t_mid/n_mid."
            )

        if t_use.shape[0] != Np or n_use.shape[0] != Np:
            raise ValueError(
                f"Panel-midpoint reconstruction requires frames of length Np={Np}; "
                f"got len(t)={t_use.shape[0]}, len(n)={n_use.shape[0]}. "
                "For cheb_spectral, ensure solver returns t_mid/n_mid arrays."
            )


        if len(s_nodes) != Np + 1:
            s_nodes = np.linspace(0.0, Ltot, Np + 1)

        J_nodes = np.zeros((Np + 1, 2), float)
        for i in range(Np):
            dJ = -(bII[i] * t_use[i] + bI[i] * n_use[i]) * float(ds[i])
            J_nodes[i + 1] = J_nodes[i] + dJ

        if enforce_global_tip_zero and Ltot > 0:
            J_end = J_nodes[-1].copy()
            if np.linalg.norm(J_end) > 0:
                alpha = (s_nodes / float(Ltot)).reshape(-1, 1)
                J_nodes = J_nodes - alpha * J_end.reshape(1, 2)

        s_clip = np.clip(s, 0.0, float(Ltot))
        idx = np.searchsorted(s_nodes, s_clip, side="right") - 1
        idx = np.clip(idx, 0, Np - 1)
        sL = s_nodes[idx]
        sR = s_nodes[idx + 1]
        w = np.where(sR > sL, (s_clip - sL) / (sR - sL), 0.0)
        J = (1.0 - w).reshape(-1, 1) * J_nodes[idx] + w.reshape(-1, 1) * J_nodes[idx + 1]

        v0 = self.calc.network.V(edge.v0)
        v1 = self.calc.network.V(edge.v1)
        t = np.array([float(v1.x - v0.x), float(v1.y - v0.y)], float)
        L = float(np.hypot(t[0], t[1]))
        if L <= 0:
            R = np.eye(2)
        else:
            ex = t / L
            ey = np.array([-ex[1], ex[0]])
            R = np.column_stack([ex, ey])
        loc = (R.T @ J.T).T
        CSD = loc[:, 0]
        COD = loc[:, 1]
        return x_edge, COD, CSD

    def reconstruct_cod_csd_panel_midpoints(
        self,
        edge_index: int,
        enforce_global_tip_zero: bool = True,
    ):
        edge_index = int(edge_index)
        if not self.is_parametrized():
            raise AttributeError("This results object is intended for solver_option='parametrized_crack'.")

        e2p = dict(self.sol.get("parametrized_edge_to_polyline", {}))
        if edge_index not in e2p:
            raise ValueError(f"Edge {edge_index} is not mapped to any polyline in solution metadata.")
        pid = int(e2p[edge_index])

        polylines = list(self.sol.get("parametrized_polylines", []))
        poly_solutions = list(self.sol.get("polyline_solutions", []))
        if pid < 0 or pid >= len(polylines) or pid >= len(poly_solutions):
            raise ValueError("polyline metadata/solution missing or inconsistent.")

        meta = dict(polylines[pid])
        solp = dict(poly_solutions[pid])

        path_vids = [int(v) for v in meta.get("path_vertex_ids", [])]
        path_edges = [int(i) for i in meta.get("path_edge_indices", [])]
        segL = np.asarray(meta.get("segment_lengths", []), float)
        Ltot = float(meta.get("total_length", np.sum(segL)))

        if edge_index not in path_edges:
            raise ValueError(f"Edge {edge_index} is not in polyline path for pid={pid}.")

        s_vert = np.zeros(len(path_vids), float)
        if len(segL) == len(path_vids) - 1:
            s_vert[1:] = np.cumsum(segL)

        k = path_edges.index(edge_index)
        s0 = float(s_vert[k])
        Le = float(segL[k])
        a_edge = 0.5 * Le

        edge = self.calc.network.edges[edge_index]
        v_start = int(path_vids[k]); v_end = int(path_vids[k + 1])
        if int(edge.v0) == v_start and int(edge.v1) == v_end:
            dir_sign = +1
        elif int(edge.v1) == v_start and int(edge.v0) == v_end:
            dir_sign = -1
        else:
            dir_sign = +1

        bI  = np.asarray(solp.get("bI"), float)
        bII = np.asarray(solp.get("bII"), float)

        # Collocation frames (may be Nc=Np or Nc=Np-1 for cheb_spectral)
        t_col = np.asarray(solp.get("t_col"), float)
        n_col = np.asarray(solp.get("n_col"), float)

        # Panel-midpoint frames (length Np) — required for panel-midpoint reconstruction
        t_mid = np.asarray(solp.get("t_mid"), float) if solp.get("t_mid") is not None else None
        n_mid = np.asarray(solp.get("n_mid"), float) if solp.get("n_mid") is not None else None

        ds = np.asarray(solp.get("ds"), float)
        Np = int(solp.get("Np", len(bI)))

        # Prefer midpoint frames; fallback to collocation frames only if they match panel count
        if t_mid is not None and n_mid is not None and len(t_mid) == Np and len(n_mid) == Np:
            t_use, n_use = t_mid, n_mid
        elif len(t_col) == Np and len(n_col) == Np:
            t_use, n_use = t_col, n_col
        else:
            raise ValueError(
                f"Panel-midpoint reconstruction requires frames of length Np={Np}. "
                f"Got len(t_mid)={None if t_mid is None else len(t_mid)}, len(n_mid)={None if n_mid is None else len(n_mid)}, "
                f"len(t_col)={len(t_col)}, len(n_col)={len(n_col)}. "
                "For cheb_spectral, ensure solver returns t_mid/n_mid."
            )

        s_nodes = np.asarray(solp.get("s_nodes"), float)
        if s_nodes.size != Np + 1:
            s_nodes = np.linspace(0.0, float(Ltot), Np + 1)

        s_mid = np.asarray(solp.get("s_mid"), float)
        if s_mid.size != Np:
            s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])

        J_nodes = np.zeros((Np + 1, 2), float)
        J_mid = np.zeros((Np, 2), float)
        for i in range(Np):
            dJ = -(bII[i] * t_use[i] + bI[i] * n_use[i]) * float(ds[i])
            J_mid[i] = J_nodes[i] + 0.5 * dJ
            J_nodes[i + 1] = J_nodes[i] + dJ

        if enforce_global_tip_zero and Ltot > 0:
            J_end = J_nodes[-1].copy()
            if np.linalg.norm(J_end) > 0:
                alpha_nodes = (s_nodes / float(Ltot)).reshape(-1, 1)
                J_nodes = J_nodes - alpha_nodes * J_end.reshape(1, 2)

                alpha_mid = (s_mid / float(Ltot)).reshape(-1, 1)
                J_mid = J_mid - alpha_mid * J_end.reshape(1, 2)

        def _interp_J(s_query: float) -> np.ndarray:
            s_query = float(np.clip(s_query, float(s_nodes[0]), float(s_nodes[-1])))
            j = int(np.searchsorted(s_nodes, s_query, side="right") - 1)
            j = max(0, min(j, len(s_nodes) - 2))
            sL = float(s_nodes[j]); sR = float(s_nodes[j + 1])
            if sR <= sL:
                return J_nodes[j].copy()
            w = (s_query - sL) / (sR - sL)
            return (1.0 - w) * J_nodes[j] + w * J_nodes[j + 1]

        J_s0 = _interp_J(s0)
        J_s1 = _interp_J(s0 + Le)

        m = (s_mid >= s0) & (s_mid <= s0 + Le + 1e-15)
        s_mid_e = s_mid[m]
        J_mid_e = J_mid[m]
        bI_e = bI[m]; bII_e = bII[m]

        x_mid_edge = (s_mid_e - (s0 + 0.5 * Le)) * float(dir_sign)

        v0_obj = self.calc.network.V(edge.v0)
        v1_obj = self.calc.network.V(edge.v1)
        p0 = np.array([float(v0_obj.x), float(v0_obj.y)], dtype=float)
        p1 = np.array([float(v1_obj.x), float(v1_obj.y)], dtype=float)
        t = p1 - p0
        L = float(np.linalg.norm(t))
        if L <= 0:
            ex = np.array([1.0, 0.0])
        else:
            ex = t / L
        ey = np.array([-ex[1], ex[0]])

        CSD_mid = J_mid_e @ ex
        COD_mid = J_mid_e @ ey

        u = (x_mid_edge + a_edge) / (2.0 * a_edge) if a_edge > 0 else np.zeros_like(x_mid_edge)
        xy_mid = p0.reshape(1,2) + u.reshape(-1,1) * (p1 - p0).reshape(1,2)

        if dir_sign > 0:
            J_v0 = J_s0
            J_v1 = J_s1
        else:
            J_v0 = J_s1
            J_v1 = J_s0

        extra = dict(
            pid=pid,
            dir_sign=dir_sign,
            s_mid=s_mid_e,
            J_mid=J_mid_e,
            xy_mid=xy_mid,
            J_v0=J_v0,
            J_v1=J_v1,
            bI=bI_e,
            bII=bII_e,
            ex=ex,
            ey=ey,
            a_edge=a_edge,
        )
        return x_mid_edge, COD_mid, CSD_mid, extra

    def crack_face_coords_panel_midpoints(
        self,
        edge_index: int,
        *,
        scale: float = 1.0,
        enforce_global_tip_zero: bool = True,
    ):
        x_mid, COD_mid, CSD_mid, extra = self.reconstruct_cod_csd_panel_midpoints(
            edge_index=edge_index,
            enforce_global_tip_zero=enforce_global_tip_zero,
        )
        xy = extra["xy_mid"]
        J = extra["J_mid"]
        xy_upper = xy + 0.5 * float(scale) * J
        xy_lower = xy - 0.5 * float(scale) * J

        edge = self.calc.network.edges[edge_index]
        v0_obj = self.calc.network.V(edge.v0)
        v1_obj = self.calc.network.V(edge.v1)
        p0 = np.array([float(v0_obj.x), float(v0_obj.y)], dtype=float)
        p1 = np.array([float(v1_obj.x), float(v1_obj.y)], dtype=float)
        J_v0 = np.asarray(extra.get("J_v0", [0.0, 0.0]), float)
        J_v1 = np.asarray(extra.get("J_v1", [0.0, 0.0]), float)

        extra["xyU0"] = p0 + 0.5 * float(scale) * J_v0
        extra["xyL0"] = p0 - 0.5 * float(scale) * J_v0
        extra["xyU1"] = p1 + 0.5 * float(scale) * J_v1
        extra["xyL1"] = p1 - 0.5 * float(scale) * J_v1

        return xy_upper, xy_lower, extra

    def edge(self, i: int):
        raise AttributeError(
            "DCEResultsNetworkV4 does not expose per-edge DCEResultsV2; "
            "use reconstruct_cod_csd_parametrized for deformed plots."
        )

    def stress_field_global(self, Xg, Yg, add_remote: bool = True):
        Xg = np.asarray(Xg, float)
        Yg = np.asarray(Yg, float)
        sxx = np.zeros_like(Xg)
        syy = np.zeros_like(Xg)
        sxy = np.zeros_like(Xg)

        E = float(self.calc.material.E)
        nu = float(self.calc.material.nu)
        mu = E / (2.0*(1.0+nu))

        for poly in self.sol.get("polyline_solutions", []):
            bI = np.asarray(poly["bI"], float)
            bII = np.asarray(poly["bII"], float)
            xmid = np.asarray(poly["x_col"], float)
            tmid = np.asarray(poly["t_col"], float)
            nmid = np.asarray(poly["n_col"], float)
            ds = np.asarray(poly["ds"], float)

            for x0, t0, n0, bi, bii, w in zip(xmid, tmid, nmid, bI, bII, ds):
                dB = (bii * t0 + bi * n0) * float(w)
                dx = Xg - float(x0[0])
                dy = Yg - float(x0[1])
                a,b,c = stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
                sxx += a; syy += b; sxy += c

        if add_remote:
            sig = applied_tensor(self.calc.applied)
            sxx += sig[0,0]; syy += sig[1,1]; sxy += sig[0,1]
        return sxx, syy, sxy

    def displacement_field_global(self, Xg, Yg, add_remote: bool = False):
        Xg = np.asarray(Xg, float)
        Yg = np.asarray(Yg, float)
        ux = np.zeros_like(Xg)
        uy = np.zeros_like(Xg)

        nu = float(self.calc.material.nu)

        for poly in self.sol.get("polyline_solutions", []):
            bI = np.asarray(poly["bI"], float)
            bII = np.asarray(poly["bII"], float)
            xmid = np.asarray(poly["x_col"], float)
            tmid = np.asarray(poly["t_col"], float)
            nmid = np.asarray(poly["n_col"], float)
            ds = np.asarray(poly["ds"], float)

            for x0, t0, n0, bi, bii, w in zip(xmid, tmid, nmid, bI, bII, ds):
                dB = (bii * t0 + bi * n0) * float(w)
                dx = Xg - float(x0[0])
                dy = Yg - float(x0[1])
                dux, duy = edge_dislocation_u(dx, dy, float(dB[0]), float(dB[1]), nu)
                ux += dux; uy += duy

        return ux, uy