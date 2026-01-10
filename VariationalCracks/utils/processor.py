
"""
dce_results_parametrized_v4_1.py

Results for v4 polyline-panel solver.

Computes stress and displacement fields by integrating differential edge-dislocation kernels
against the solved panel Burgers densities on each polyline.

This avoids artifacts associated with treating each segment as a finite crack with tips.
"""

from __future__ import annotations
import numpy as np
import math

def applied_tensor(applied) -> np.ndarray:
    if hasattr(applied, "tensor") and callable(applied.tensor):
        return np.asarray(applied.tensor(), float)
    return np.array([[float(applied.sigma_xx), float(applied.sigma_xy)],
                     [float(applied.sigma_xy), float(applied.sigma_yy)]], float)

def _edge_dislocation_u(dx: np.ndarray, dy: np.ndarray, dBx: float, dBy: float, nu: float):
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    nu = float(nu)

    eps = 1e-30
    r2 = dx*dx + dy*dy + eps
    inv = 1.0 / r2

    atan = np.arctan2(dy, dx)
    ln = np.log(r2)

    c1 = 1.0 / (2.0*np.pi)
    c2 = 1.0 / (4.0*np.pi*(1.0 - nu))

    ux_bx = c1 * atan + c2 * (dx*dy) * inv
    uy_bx = -c2 * ((1.0 - 2.0*nu)*0.5*ln + 0.5*(dx*dx - dy*dy)*inv)

    ux_by = -c2 * ((1.0 - 2.0*nu)*0.5*ln - 0.5*(dx*dx - dy*dy)*inv)
    uy_by = c1 * atan - c2 * (dx*dy) * inv

    ux = dBx * ux_bx + dBy * ux_by
    uy = dBx * uy_bx + dBy * uy_by
    return ux, uy

def _stress_edge_dislocation(dx: np.ndarray, dy: np.ndarray, dBx: float, dBy: float, mu: float, nu: float):
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    mu = float(mu); nu = float(nu)

    eps = 1e-30
    coef = mu / (2.0*math.pi*(1.0 - nu))

    def bx_stress(x, y, b):
        r2 = x*x + y*y + eps
        r4 = r2*r2
        sxx = -coef*b * (y*(3.0*x*x + y*y)) / r4
        syy =  coef*b * (y*(x*x - y*y))     / r4
        sxy =  coef*b * (x*(x*x - y*y))     / r4
        return sxx, syy, sxy

    sxx = np.zeros_like(dx)
    syy = np.zeros_like(dx)
    sxy = np.zeros_like(dx)

    if abs(dBx) > 0:
        a,b,c = bx_stress(dx, dy, dBx)
        sxx += a; syy += b; sxy += c
    if abs(dBy) > 0:
        x1 = dy
        y1 = -dx
        sxx1, syy1, sxy1 = bx_stress(x1, y1, dBy)
        sxx += syy1
        syy += sxx1
        sxy += -sxy1
    return sxx, syy, sxy

class DCEResultsNetworkV4:
    def __init__(self, calc, sol: dict):
        self.calc = calc
        self.sol = sol

    # -------------------------
    # Compatibility hooks used by dce_plots_parametrized_v1
    # -------------------------
    def is_parametrized(self) -> bool:
        return str(self.sol.get("solver_option", "")).lower() == "parametrized_crack"

    def reconstruct_cod_csd_parametrized(
        self,
        edge_index: int,
        n_pts: int = 4000,
        enforce_global_tip_zero: bool = True,
    ):
        """Return (x_edge, COD, CSD) for an edge that belongs to a polyline crack.

        This is required by dce_plots_parametrized_v1. We reconstruct the displacement-jump
        vector along the *polyline* by integrating the solved panel Burgers densities, then
        express it in the local frame of the requested edge.
        """
        edge_index = int(edge_index)
        if not self.is_parametrized():
            raise AttributeError("This results object is intended for solver_option='parametrized_crack'.")

        # Map edge -> polyline id
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

        # Build cumulative arc-length at path vertices
        s_vert = np.zeros(len(path_vids), float)
        if len(segL) == len(path_vids) - 1:
            s_vert[1:] = np.cumsum(segL)

        # Determine this edge's position in the path
        k = path_edges.index(edge_index)
        s0 = float(s_vert[k])
        Le = float(segL[k])
        a = 0.5 * Le

        # Determine path direction relative to stored edge orientation
        edge = self.calc.network.edges[edge_index]
        v_start = int(path_vids[k])
        v_end = int(path_vids[k + 1])
        if int(edge.v0) == v_start and int(edge.v1) == v_end:
            dir_sign = +1
        elif int(edge.v1) == v_start and int(edge.v0) == v_end:
            dir_sign = -1
        else:
            dir_sign = +1  # fallback

        # Sample points along the edge in its local x in [-a,a]
        n_pts = int(max(200, n_pts))
        x_edge = np.linspace(-a, a, n_pts)

        # Map x_edge -> arc-length s along the polyline
        if dir_sign == +1:
            s = s0 + (x_edge + a)
        else:
            s = s0 + (a - x_edge)

        # Reconstruct global jump vector J(s) by integrating b over panels
        # Panel data
        s_nodes = np.asarray(solp.get("s_nodes"), float)
        bI = np.asarray(solp.get("bI"), float)
        bII = np.asarray(solp.get("bII"), float)
        t_col = np.asarray(solp.get("t_col"), float)
        n_col = np.asarray(solp.get("n_col"), float)
        ds = np.asarray(solp.get("ds"), float)
        Np = int(solp.get("Np", len(bI)))

        if len(s_nodes) != Np + 1:
            # Fall back to cosine grid stored in solver v4 panel structure
            s_nodes = np.linspace(0.0, Ltot, Np + 1)

        # cumulative jump at panel nodes
        J_nodes = np.zeros((Np + 1, 2), float)
        for i in range(Np):
            dJ = -(bII[i] * t_col[i] + bI[i] * n_col[i]) * float(ds[i])  # sign convention: opening COD>0 under +sigma_yy
            J_nodes[i + 1] = J_nodes[i] + dJ

        if enforce_global_tip_zero and Ltot > 0:
            # Remove affine gauge so J(0)=J(L)=0
            J_end = J_nodes[-1].copy()
            if np.linalg.norm(J_end) > 0:
                alpha = (s_nodes / float(Ltot)).reshape(-1, 1)
                J_nodes = J_nodes - alpha * J_end.reshape(1, 2)

        # interpolate J at sample s
        s_clip = np.clip(s, 0.0, float(Ltot))
        # find panel index
        idx = np.searchsorted(s_nodes, s_clip, side="right") - 1
        idx = np.clip(idx, 0, Np - 1)
        sL = s_nodes[idx]
        sR = s_nodes[idx + 1]
        w = np.where(sR > sL, (s_clip - sL) / (sR - sL), 0.0)
        J = (1.0 - w).reshape(-1, 1) * J_nodes[idx] + w.reshape(-1, 1) * J_nodes[idx + 1]

        # Convert global jump to edge-local (CSD,COD)
        # Edge rotation local->global
        v0 = self.calc.network.V(edge.v0)
        v1 = self.calc.network.V(edge.v1)
        t = np.array([float(v1.x - v0.x), float(v1.y - v0.y)], float)
        L = float(np.hypot(t[0], t[1]))
        if L <= 0:
            R = np.eye(2)
        else:
            ex = t / L
            ey = np.array([-ex[1], ex[0]])
            R = np.column_stack([ex, ey])  # local (t,n) -> global
        loc = (R.T @ J.T).T
        CSD = loc[:, 0]
        COD = loc[:, 1]
        return x_edge, COD, CSD


    def reconstruct_cod_csd_panel_midpoints(
        self,
        edge_index: int,
        enforce_global_tip_zero: bool = True,
    ):
        """Return COD/CSD sampled at *polyline panel midpoints* that lie on a given edge.

        Motivation
        ----------
        The parametrized-crack solve stores Burgers densities (bI, bII) per panel. Reconstructing
        COD/CSD on an arbitrarily fine grid (n_pts >> Np) can reveal deterministic panel-scale
        ripple. Sampling at panel midpoints provides a representation-consistent, unsmoothed
        COD/CSD that typically converges cleanly with ne_half.

        Returns
        -------
        x_mid_edge : (M,) array
            Edge-local coordinate in [-a_edge, a_edge] at panel midpoints.
        COD_mid : (M,) array
            Opening jump at panel midpoints in the edge-local normal direction.
        CSD_mid : (M,) array
            Sliding jump at panel midpoints in the edge-local tangent direction.
        extra : dict
            Useful diagnostics: global midpoint coords, bI/bII on those panels, and global jump J.
        """
        edge_index = int(edge_index)
        if not self.is_parametrized():
            raise AttributeError("This results object is intended for solver_option='parametrized_crack'.")

        # Map edge -> polyline id
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

        # cumulative arc-length at path vertices
        s_vert = np.zeros(len(path_vids), float)
        if len(segL) == len(path_vids) - 1:
            s_vert[1:] = np.cumsum(segL)

        # this edge's arc-length window along the polyline
        k = path_edges.index(edge_index)
        s0 = float(s_vert[k])
        Le = float(segL[k])
        a_edge = 0.5 * Le

        # Edge orientation relative to path
        edge = self.calc.network.edges[edge_index]
        v_start = int(path_vids[k]); v_end = int(path_vids[k + 1])
        if int(edge.v0) == v_start and int(edge.v1) == v_end:
            dir_sign = +1
        elif int(edge.v1) == v_start and int(edge.v0) == v_end:
            dir_sign = -1
        else:
            dir_sign = +1

        # Panel data (polyline-global)
        bI  = np.asarray(solp.get("bI"), float)
        bII = np.asarray(solp.get("bII"), float)
        t_col = np.asarray(solp.get("t_col"), float)
        n_col = np.asarray(solp.get("n_col"), float)
        ds = np.asarray(solp.get("ds"), float)
        Np = int(solp.get("Np", len(bI)))

        # Panel-node arc-length grid
        s_nodes = np.asarray(solp.get("s_nodes"), float)
        if s_nodes.size != Np + 1:
            s_nodes = np.linspace(0.0, float(Ltot), Np + 1)

        # Panel midpoints in arc-length
        s_mid = np.asarray(solp.get("s_mid"), float)
        if s_mid.size != Np:
            s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])

        # Reconstruct global jump at panel nodes and midpoints
        J_nodes = np.zeros((Np + 1, 2), float)
        J_mid = np.zeros((Np, 2), float)
        for i in range(Np):
            dJ = -(bII[i] * t_col[i] + bI[i] * n_col[i]) * float(ds[i])  # sign convention
            J_mid[i] = J_nodes[i] + 0.5 * dJ
            J_nodes[i + 1] = J_nodes[i] + dJ

        if enforce_global_tip_zero and Ltot > 0:
            # Remove affine gauge so J(0)=J(L)=0
            J_end = J_nodes[-1].copy()
            if np.linalg.norm(J_end) > 0:
                alpha_nodes = (s_nodes / float(Ltot)).reshape(-1, 1)
                J_nodes = J_nodes - alpha_nodes * J_end.reshape(1, 2)

                alpha_mid = (s_mid / float(Ltot)).reshape(-1, 1)
                J_mid = J_mid - alpha_mid * J_end.reshape(1, 2)

        # Helper: interpolate jump at an arbitrary arc-length position s_query
        def _interp_J(s_query: float) -> np.ndarray:
            s_query = float(np.clip(s_query, float(s_nodes[0]), float(s_nodes[-1])))
            # Find right index j such that s_nodes[j] <= s_query <= s_nodes[j+1]
            j = int(np.searchsorted(s_nodes, s_query, side="right") - 1)
            j = max(0, min(j, len(s_nodes) - 2))
            sL = float(s_nodes[j]); sR = float(s_nodes[j + 1])
            if sR <= sL:
                return J_nodes[j].copy()
            w = (s_query - sL) / (sR - sL)
            return (1.0 - w) * J_nodes[j] + w * J_nodes[j + 1]

        # Jump at this edge's endpoints in *polyline* orientation (at path vertices)
        J_s0 = _interp_J(s0)
        J_s1 = _interp_J(s0 + Le)

        # Select panels whose midpoints lie on this edge window
        m = (s_mid >= s0) & (s_mid <= s0 + Le + 1e-15)
        s_mid_e = s_mid[m]
        J_mid_e = J_mid[m]
        bI_e = bI[m]; bII_e = bII[m]

        # Map to edge-local x in [-a_edge, a_edge] consistent with edge orientation
        x_mid_edge = (s_mid_e - (s0 + 0.5 * Le)) * float(dir_sign)

        # Edge local frame (based on edge orientation v0->v1)
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
        # Project global jump into edge local (t,n)
        CSD_mid = J_mid_e @ ex
        COD_mid = J_mid_e @ ey

        # Physical midpoint coordinates on the (undeformed) edge centerline
        # param u in [0,1] based on edge-local x
        u = (x_mid_edge + a_edge) / (2.0 * a_edge) if a_edge > 0 else np.zeros_like(x_mid_edge)
        xy_mid = p0.reshape(1,2) + u.reshape(-1,1) * (p1 - p0).reshape(1,2)

        # Map endpoint jumps to the edge's (v0, v1) ordering
        # J_s0/J_s1 correspond to path vertex order (v_start -> v_end). If the edge is
        # traversed opposite to the path, swap them so J_v0 always corresponds to edge.v0.
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
        """Return upper/lower crack-face coordinates at panel midpoints (no smoothing).

        The crack centerline is the original edge geometry. Upper/lower faces are obtained by
        adding/subtracting half the displacement-jump vector J at each midpoint.

        Returns
        -------
        xy_upper : (M,2), xy_lower : (M,2), extra : dict
        """
        x_mid, COD_mid, CSD_mid, extra = self.reconstruct_cod_csd_panel_midpoints(
            edge_index=edge_index,
            enforce_global_tip_zero=enforce_global_tip_zero,
        )
        xy = extra["xy_mid"]
        J = extra["J_mid"]
        xy_upper = xy + 0.5 * float(scale) * J
        xy_lower = xy - 0.5 * float(scale) * J

        # Also compute displaced endpoints (needed to connect faces cleanly at junctions)
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

    # Optional: provide an 'edge' method for compatibility with older code paths.
    def edge(self, i: int):
        raise AttributeError(
            "DCEResultsNetworkV4 (v4) does not expose per-edge DCEResultsV2; "
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

        # integrate per polyline panel (piecewise constant), using midpoints and ds weights
        for poly in self.sol.get("polyline_solutions", []):
            bI = np.asarray(poly["bI"], float)
            bII = np.asarray(poly["bII"], float)
            xmid = np.asarray(poly["x_col"], float)   # one per panel
            tmid = np.asarray(poly["t_col"], float)
            nmid = np.asarray(poly["n_col"], float)
            ds = np.asarray(poly["ds"], float)

            for x0, t0, n0, bi, bii, w in zip(xmid, tmid, nmid, bI, bII, ds):
                # Burgers differential in global coordinates
                dB = (bii * t0 + bi * n0) * float(w)
                dx = Xg - float(x0[0])
                dy = Yg - float(x0[1])
                a,b,c = _stress_edge_dislocation(dx, dy, float(dB[0]), float(dB[1]), mu, nu)
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
                dux, duy = _edge_dislocation_u(dx, dy, float(dB[0]), float(dB[1]), nu)
                ux += dux; uy += duy

        return ux, uy



# -----------------------------------------------------------------------------
# SIF extraction utilities
# -----------------------------------------------------------------------------

def sif_from_cod_fit(
    x: np.ndarray,
    COD: np.ndarray,
    CSD: np.ndarray,
    *,
    a: float,
    mu: float,
    kappa: float,
    tip: str = "right",
    window: str = "auto",   # "auto" or "fixed"
    rho_min: float = 5e-4,
    rho_max: float = 5e-2,
    c1: float = 10.0,
    c2: float = 0.20,
    n_fit: int = 120,
    two_term: bool = True,
    dx_tip: float | None = None,
    min_pts: int = 16,
):
    """Robust near-tip SIF extraction from COD/CSD using the v2-style asymptotic fit.

    Model
    -----
    COD(r) ≈ A * sqrt(r/(2π)) + B * sqrt(r/(2π)) * (r/(2π))     (optional 2nd term)
    CSD(r) ≈ AII * sqrt(r/(2π)) + BII * sqrt(r/(2π)) * (r/(2π))

    Mapping to SIFs (same convention as v2):
        K_I  = (mu / (kappa + 1)) * A
        K_II = (mu / (kappa + 1)) * AII

    The primary failure mode for coarse discretizations is a too-narrow fit window.
    This implementation adaptively expands r_max (and mildly relaxes r_min) until
    at least `min_pts` samples are available.
    """
    x = np.asarray(x, float)
    COD = np.asarray(COD, float)
    CSD = np.asarray(CSD, float)
    a = float(a)

    tip_l = str(tip).lower()
    if tip_l == "right":
        r = a - x
    elif tip_l == "left":
        r = a + x
    else:
        raise ValueError("tip must be 'left' or 'right'")

    # Estimate dx_tip from sampling if not provided
    if dx_tip is None:
        xx = np.sort(x[np.isfinite(x)])
        if xx.size < 3:
            dx_tip = 0.05 * a
        else:
            if tip_l == "right":
                slab = xx[-min(25, xx.size):]
            else:
                slab = xx[:min(25, xx.size)]
            dxx = np.diff(slab)
            dxx = dxx[np.isfinite(dxx) & (dxx > 0)]
            dx_tip = float(np.min(dxx)) if dxx.size else 0.05 * a

    dx_tip = float(dx_tip)

    # Initial window
    wmode = str(window).lower()
    if wmode == "auto":
        r_min = max(float(rho_min) * a, float(c1) * dx_tip)
        r_max = min(float(rho_max) * a, float(c2) * math.sqrt(max(a * dx_tip, 1e-30)))
        if r_max <= 1.05 * r_min:
            r_max = 2.5 * r_min
    elif wmode == "fixed":
        r_min = float(rho_min) * a
        r_max = float(rho_max) * a
    else:
        raise ValueError("window must be 'auto' or 'fixed'")

    # Adaptive expansion if too few points
    max_expand_iters = 12
    expand = 1.6
    relax = 0.8
    r_max_cap = float(rho_max) * a

    for _ in range(max_expand_iters):
        m = np.isfinite(r) & np.isfinite(COD) & np.isfinite(CSD) & (r > r_min) & (r < r_max)
        if np.count_nonzero(m) >= int(min_pts):
            break
        r_max = min(r_max * expand, r_max_cap)
        if r_max >= 0.999 * r_max_cap:
            r_min = max(r_min * relax, 1e-10 * a)

    m = np.isfinite(r) & np.isfinite(COD) & np.isfinite(CSD) & (r > r_min) & (r < r_max)
    rr = r[m]
    CODw = COD[m]
    CSDw = CSD[m]

    if rr.size < 8:
        raise ValueError(
            f"Fit window too small after expansion: rr.size={rr.size}. "
            f"Try increasing rho_max (currently {rho_max}) or increasing n_pts, "
            f"or pass dx_tip explicitly."
        )

    # Use closest n_fit points (smallest r) within the window
    order = np.argsort(rr)
    rr = rr[order][: int(n_fit)]
    CODw = CODw[order][: int(n_fit)]
    CSDw = CSDw[order][: int(n_fit)]

    phi = np.sqrt(rr / (2.0 * np.pi))

    if two_term:
        rbar = rr / (2.0 * np.pi)
        X = np.column_stack([phi, phi * rbar])
        AI, BI = np.linalg.lstsq(X, CODw, rcond=None)[0]
        AII, BII = np.linalg.lstsq(X, CSDw, rcond=None)[0]
    else:
        AI = float(np.dot(phi, CODw) / np.dot(phi, phi))
        AII = float(np.dot(phi, CSDw) / np.dot(phi, phi))
        BI = 0.0
        BII = 0.0

    KI = (float(mu) / (float(kappa) + 1.0)) * float(AI)
    KII = (float(mu) / (float(kappa) + 1.0)) * float(AII)

    meta = dict(
        window=wmode,
        r_min=float(r_min),
        r_max=float(r_max),
        dx_tip=float(dx_tip),
        n_fit=int(len(rr)),
        two_term=bool(two_term),
        A_I=float(AI),
        B_I=float(BI),
        A_II=float(AII),
        B_II=float(BII),
    )
    return KI, KII, meta


def estimate_K_from_jump_near_tip(x, jump, a, Eprime, side="right", frac_window=0.08):
    """One-term sqrt(r) slope estimator (quick diagnostic)."""
    x = np.asarray(x, float)
    jump = np.asarray(jump, float)
    a = float(a)

    if str(side).lower().startswith("r"):
        r = a - x
    else:
        r = a + x

    rmax = float(frac_window) * a
    m = np.isfinite(r) & np.isfinite(jump) & (r > 0) & (r < rmax)
    if np.count_nonzero(m) < 20:
        return np.nan

    X = np.sqrt(r[m])
    Y = jump[m]
    A = float(np.dot(X, Y) / np.dot(X, X))
    K = A * float(Eprime) * np.sqrt(2.0 * np.pi) / 8.0
    return K
