"""Post-processing utilities for jump (J) and configurational/PK-style forces.

This module is intentionally solver-neutral: it only consumes objects already
available on the `results` instance (typically named `res`) and `res.sol`.

Key distinction (to prevent recurring mix-ups)
---------------------------------------------
J(s):
    Displacement jump vector across crack faces (kinematics). This is what
    must be used to build *deformed crack geometry*.

DeltaJ_branch (historically printed as "B" in a diagnostics cell):
    The integral of the solved density along a branch,
        DeltaJ_branch = ∫ (bII t + bI n) ds
    This is a *kinematic* content (jump accumulation), not a PK force.

F_PK:
    A true Peach–Koehler/configurational force requires stress (or an energy
    derivative). This module provides an optional helper to compute a PK-style
    force density from an externally-supplied stress field evaluator, but it is
    not required for plotting deformed geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from ..Uprocessor import *  
from ..Uplotter.reconstruct import *

import numpy as np, math
from typing import Tuple, Optional


def _as2(v) -> np.ndarray:
    return np.asarray(v, float).reshape(2,)


@dataclass
class JunctionDiagnostics:
    """Structured junction data extracted from `res.sol` (parametrized option)."""

    degree: Dict[int, int]
    v_inc: Dict[int, List[Tuple[int, str]]]


class PKProcessor:
    """Compute jump vectors and (optional) PK-style forces for plotting."""

    def __init__(self, res):
        self.res = res
        self.calc = getattr(res, "calc", None)
        self.sol = getattr(res, "sol", None) if hasattr(res, "sol") else None

    # ---------------------------------------------------------------------
    # Graph helpers
    # ---------------------------------------------------------------------

    def degree_map(self) -> Dict[int, int]:
        V = self.calc.network.vertices
        E = self.calc.network.edges
        deg = {int(v.id): 0 for v in V}
        for e in E:
            deg[int(e.v0)] = deg.get(int(e.v0), 0) + 1
            deg[int(e.v1)] = deg.get(int(e.v1), 0) + 1
        return deg

    def junction_incidence_parametrized(self) -> JunctionDiagnostics:
        """Build vertex -> incident (pid, which_end) map for parametrized polyline solver."""
        sol = self.sol
        if not isinstance(sol, dict):
            raise ValueError("res.sol is missing; cannot build parametrized incidence map.")

        e2p = sol.get("parametrized_edge_to_polyline", {})
        polys = sol.get("parametrized_polylines", [])
        deg = self.degree_map()

        v_inc: Dict[int, List[Tuple[int, str]]] = {}
        for _edge_index, pid in e2p.items():
            pid = int(pid)
            if pid < 0 or pid >= len(polys):
                continue
            pmeta = dict(polys[pid])
            v0 = int(pmeta.get("v_start", pmeta.get("path_vertex_ids", [0])[0]))
            v1 = int(pmeta.get("v_end", pmeta.get("path_vertex_ids", [0])[-1]))
            v_inc.setdefault(v0, []).append((pid, "start"))
            v_inc.setdefault(v1, []).append((pid, "end"))

        return JunctionDiagnostics(degree=deg, v_inc=v_inc)

    # ---------------------------------------------------------------------
    # Kinematic jump reconstruction using the user's diagnostic formulas
    # ---------------------------------------------------------------------

    def deltaJ_branch(self, pid: int) -> np.ndarray:
        """Integral of density along polyline branch (historically printed as B)."""
        solps = self.sol.get("polyline_solutions", []) if isinstance(self.sol, dict) else []
        s = dict(solps[int(pid)])
        bI = np.asarray(s["bI"], float)
        bII = np.asarray(s["bII"], float)
        ds = np.asarray(s["ds"], float)
        t = np.asarray(s.get("t_mid", s.get("t_col")), float)
        n = np.asarray(s.get("n_mid", s.get("n_col")), float)
        B = np.zeros(2, float)
        for k in range(len(ds)):
            B += (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])
        return B

    def J_at_vertex_parametrized(self, pid: int, which: str) -> np.ndarray:
        """Endpoint jump J at polyline end (start=endpoints use J0 gauge, end integrates)."""
        solps = self.sol.get("polyline_solutions", []) if isinstance(self.sol, dict) else []
        s = dict(solps[int(pid)])
        bI = np.asarray(s["bI"], float)
        bII = np.asarray(s["bII"], float)
        ds = np.asarray(s["ds"], float)
        t = np.asarray(s.get("t_mid", s.get("t_col")), float)
        n = np.asarray(s.get("n_mid", s.get("n_col")), float)
        J0 = np.asarray(s.get("J0", [0.0, 0.0]), float)
        J = J0.copy()
        if str(which).lower() == "end":
            for k in range(len(ds)):
                J -= (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])
        return J

    def junction_J_map(self) -> Dict[int, np.ndarray]:
        """Return common junction jump vectors J(v) keyed by vertex id."""
        out: Dict[int, np.ndarray] = {}
        try:
            jd = self.junction_incidence_parametrized()
        except Exception:
            return out

        for vid, items in jd.v_inc.items():
            if jd.degree.get(int(vid), 0) < 2:
                continue
            # Use the first incident branch as reference
            pid0, which0 = items[0]
            out[int(vid)] = self.J_at_vertex_parametrized(pid0, which0)
        return out

    # ---------------------------------------------------------------------
    # Jump along an edge for plotting deformed geometry
    # ---------------------------------------------------------------------

    def jump_along_edge(
        self,
        edge_index: int,
        n_theta: int = 4000,
        enforce_global_tip_zero: bool = False,
        tip_point: Optional[np.ndarray] = None,
        cod_window_panels: int = 0,
        csd_window_panels: int = 0,
        **_ignored: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (P, J) arrays along an edge.

        P: (N,2) points on the *midline* of the edge in global coordinates.
        J: (N,2) jump vectors in global coordinates at the same param points.

        This uses COD/CSD reconstructions (kinematics). It never uses deltaJ_branch.

        Notes
        -----
        `tip_point` and any additional keyword arguments are accepted for
        backward/forward compatibility with plotter code, but are not required
        for the geometric jump reconstruction in this module.
        """
        edge_index = int(edge_index)
        # Prefer panel-midpoint geometric COD/CSD if available.
        # This avoids polyline-global gauge drift and produces a local geometric jump.
        if hasattr(self.res, "reconstruct_cod_csd_panel_midpoints"):
            # Signature varies across versions; only pass universally-supported args.
            x, COD, CSD, extra = self.res.reconstruct_cod_csd_panel_midpoints(
                edge_index=edge_index,
                enforce_global_tip_zero=bool(enforce_global_tip_zero),
            )
            extra = dict(extra) if isinstance(extra, dict) else {}
        elif getattr(self.res, "is_parametrized", lambda: False)():
            # Fallback to parametrized reconstruction helper (may carry a gauge constant).
            from ..Uplotter.reconstruct import reconstruct_cod_csd_parametrized_smoothed

            x, COD, CSD = reconstruct_cod_csd_parametrized_smoothed(
                self.res,
                edge_index=edge_index,
                n_pts=int(max(400, n_theta)),
                enforce_global_tip_zero=bool(enforce_global_tip_zero),
                cod_window_panels=int(cod_window_panels),
                csd_window_panels=int(csd_window_panels),
            )
            extra = {}
        else:
            edge_res = self.res.edge(edge_index)
            x, COD, CSD = edge_res.reconstruct_cod_csd(
                n_theta=int(n_theta),
                enforce_tip_zero=bool(enforce_global_tip_zero),
            )
            extra = {}

        x = np.asarray(x, float).reshape(-1,)
        COD = np.asarray(COD, float).reshape(-1,)
        CSD = np.asarray(CSD, float).reshape(-1,)

        # Geometry frame for this edge
        edge = self.calc.network.edges[edge_index]
        v0 = self.calc.network.V(edge.v0)
        v1 = self.calc.network.V(edge.v1)
        p0 = np.array([float(v0.x), float(v0.y)], float)
        p1 = np.array([float(v1.x), float(v1.y)], float)
        t = p1 - p0
        L = float(np.hypot(t[0], t[1]))
        if ("ex" in extra) and ("ey" in extra):
            ex = np.asarray(extra["ex"], float).reshape(2,)
            ey = np.asarray(extra["ey"], float).reshape(2,)
        elif L > 0:
            ex = t / L
            ey = np.array([-ex[1], ex[0]], float)
        else:
            ex = np.array([1.0, 0.0], float)
            ey = np.array([0.0, 1.0], float)

        # Midline points
        if ("xy_mid" in extra) and np.asarray(extra["xy_mid"]).ndim == 2:
            P = np.asarray(extra["xy_mid"], float)
        else:
            c = 0.5 * (p0 + p1)
            P = c.reshape(1, 2) + x.reshape(-1, 1) * ex.reshape(1, 2)

        # Jump in global coords from local (CSD,COD)
        J = CSD.reshape(-1, 1) * ex.reshape(1, 2) + COD.reshape(-1, 1) * ey.reshape(1, 2)

        # Gauge fix for half-crack visualization: if a physical tip point is provided,
        # shift the reconstructed jump by a constant so that J(tip)=0 at the nearest
        # reconstructed point to that tip. This does NOT change gradients/densities;
        # it only fixes the additive integration constant for geometric reconstruction.
        if tip_point is not None and P.size and J.size:
            tp = np.asarray(tip_point, float).reshape(2,)
            d2 = np.sum((P - tp.reshape(1, 2)) ** 2, axis=1)
            k = int(np.argmin(d2))
            J = J - J[k].reshape(1, 2)

        return P, J



    # ---------------------------------------------------------------------
    # B-content (integrated density) vectors
    # ---------------------------------------------------------------------

    def edge_to_pid(self, edge_index: int) -> Optional[int]:
        """Map a network edge index to a polyline id (parametrized solver)."""
        sol = self.sol
        if not isinstance(sol, dict):
            return None
        e2p = sol.get("parametrized_edge_to_polyline", {})
        pid = e2p.get(int(edge_index), None)
        return int(pid) if pid is not None else None

    def B_content_edge(self, edge_index: int) -> Optional[np.ndarray]:
        """Return the B-content vector for this edge (parametrized), as a 2-vector."""
        pid = self.edge_to_pid(edge_index)
        if pid is None:
            return None
        return self.deltaJ_branch(pid)

    def B_content_at_vertex(self, vertex_id: int) -> Dict[tuple[int, str], np.ndarray]:
        """Return per-incident-branch B-content vectors at a given vertex.

        Returns a dict keyed by (pid, which_end) for parametrized incidence.
        """
        out: Dict[tuple[int, str], np.ndarray] = {}
        try:
            jd = self.junction_incidence_parametrized()
        except Exception:
            return out
        items = jd.v_inc.get(int(vertex_id), [])
        for pid, which in items:
            out[(int(pid), str(which))] = self.deltaJ_branch(int(pid))
        return out

    # ---------------------------------------------------------------------
    # Optional PK-style forces from stress excluding self (best-effort)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------

    def pk_force_density_from_stress(self, sigma: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        """2D PK-style force density for line direction ez: f = (sigma·b)×ez."""
        sigma = np.asarray(sigma, float).reshape(2, 2)
        bvec = _as2(bvec)
        a = sigma @ bvec
        return np.array([a[1], -a[0]], float)

    def F_PK_edge_tip(
        self,
        edge_index: int,
        *,
        at: str = "end",
        exclude_self: bool = True,
        n_samples: int = 32,
    ) -> Optional[np.ndarray]:
        """Best-effort PK-like force near a tip by integrating PK density along edge.

        Requires one of the following hooks on `res`:
          - res.stress_field_global(X, Y, add_remote=True) -> (sxx, syy, sxy)
          - res.stress_field_global_excluding(edge_index, X, Y, add_remote=True)
            (preferred for exclude_self)

        If exclusion is not available, returns None when exclude_self=True.
        """
        edge_index = int(edge_index)
        # Need the solved density along this edge. For parametrized, we can map edge->pid.
        sol = self.sol
        if not isinstance(sol, dict):
            return None

        # Determine pid for this edge (parametrized) and extract bvec along panels
        e2p = sol.get("parametrized_edge_to_polyline", {})
        pid = e2p.get(edge_index, None)
        if pid is None:
            return None
        pid = int(pid)
        solps = sol.get("polyline_solutions", [])
        if pid < 0 or pid >= len(solps):
            return None
        s = dict(solps[pid])
        bI = np.asarray(s["bI"], float)
        bII = np.asarray(s["bII"], float)
        ds = np.asarray(s["ds"], float)
        t = np.asarray(s.get("t_mid", s.get("t_col")), float)
        n = np.asarray(s.get("n_mid", s.get("n_col")), float)

        # Panel midpoint positions if available
        xy = np.asarray(s.get("xy_mid", s.get("xy_col", [])), float)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != len(ds):
            # fallback: approximate by linear interpolation along the edge geometry
            edge = self.calc.network.edges[edge_index]
            v0 = self.calc.network.V(edge.v0)
            v1 = self.calc.network.V(edge.v1)
            p0 = np.array([float(v0.x), float(v0.y)], float)
            p1 = np.array([float(v1.x), float(v1.y)], float)
            lam = (np.arange(len(ds)) + 0.5) / max(1, len(ds))
            xy = p0.reshape(1, 2) * (1 - lam).reshape(-1, 1) + p1.reshape(1, 2) * lam.reshape(-1, 1)

        # Select samples near chosen tip end
        at = str(at).lower()
        if at not in ("start", "end"):
            at = "end"
        idxs = np.arange(len(ds))
        if at == "start":
            idxs = idxs[: max(1, int(n_samples))]
        else:
            idxs = idxs[-max(1, int(n_samples)) :]

        # Stress evaluator
        stress_excl = getattr(self.res, "stress_field_global_excluding", None)
        stress_tot = getattr(self.res, "stress_field_global", None)
        if exclude_self:
            if callable(stress_excl):
                def _sigma_at(X, Y):
                    sxx, syy, sxy = stress_excl(edge_index, X, Y, add_remote=True)
                    return sxx, syy, sxy
            else:
                return None
        else:
            if not callable(stress_tot):
                return None
            def _sigma_at(X, Y):
                sxx, syy, sxy = stress_tot(X, Y, add_remote=True)
                return sxx, syy, sxy

        F = np.zeros(2, float)
        for k in idxs:
            # Build bvec in global
            bvec = (bII[k] * t[k] + bI[k] * n[k]).reshape(2,)
            X = np.array([xy[k, 0]], float)
            Y = np.array([xy[k, 1]], float)
            sxx, syy, sxy = _sigma_at(X, Y)
            sig = np.array([[float(sxx[0]), float(sxy[0])], [float(sxy[0]), float(syy[0])]], float)
            f = self.pk_force_density_from_stress(sig, bvec)
            F += f * float(ds[k])
        return F
# Tip proxy========================================================================
    def F_PK_tip(
        self,
        edge_index: int,
        *,
        at: str = "end",
        exclude_self: bool = True,
        window_frac: float = 0.1,
        n_eval_panels: int = 1,
    ) -> Optional[np.ndarray]:
        """Method (1): tip PK estimate using one σ evaluation and B_edge content.

        F_PK_tip = (σ_tot(x_eval) · B_edge) × e_z

        - x_eval is the midpoint of the panel closest to the requested tip (default),
        optionally averaged over the first `n_eval_panels` panels near the tip.
        - B_edge = Σ_k ΔB_k over the *edge segment* panels,
        with ΔB_k = (bII_k t_k + bI_k n_k) ds_k.

        Default window_frac=0.1 is accepted for interface consistency (not used in B_edge integration).
        """
        edge_index = int(edge_index)
        at = str(at).lower().strip()
        if at not in ("start", "end"):
            at = "end"

        if not hasattr(self.res, "reconstruct_cod_csd_panel_midpoints"):
            return None

        try:
            x_mid, COD_mid, CSD_mid, extra = self.res.reconstruct_cod_csd_panel_midpoints(
                edge_index=edge_index,
                enforce_global_tip_zero=False,
            )
        except Exception:
            return None

        extra = dict(extra) if isinstance(extra, dict) else {}
        bI = np.asarray(extra.get("bI", []), float).reshape(-1,)
        bII = np.asarray(extra.get("bII", []), float).reshape(-1,)
        xy = np.asarray(extra.get("xy_mid", []), float)
        if bI.size == 0 or xy.ndim != 2 or xy.shape[0] != bI.size:
            return None

        pid = int(extra.get("pid", -1))
        sol = self.sol if isinstance(self.sol, dict) else {}
        solps = sol.get("polyline_solutions", [])
        if pid < 0 or pid >= len(solps):
            return None
        solp = dict(solps[pid])

        # Mask ds/t/n to match edge-segment panels using s_mid alignment
        s_mid_e = np.asarray(extra.get("s_mid", []), float).reshape(-1,)
        s_mid_all = np.asarray(solp.get("s_mid", []), float).reshape(-1,)
        if s_mid_all.size and s_mid_e.size:
            ds_guess = np.nanmedian(np.diff(np.sort(s_mid_all))) if s_mid_all.size > 3 else None
            tol = float(ds_guess * 0.51) if ds_guess and np.isfinite(ds_guess) else 1e-12
            mask = np.zeros_like(s_mid_all, dtype=bool)
            for sv in s_mid_e:
                mask |= np.abs(s_mid_all - float(sv)) <= tol
        else:
            mask = slice(None)

        ds_all = np.asarray(solp.get("ds", []), float).reshape(-1,)
        t_all = np.asarray(solp.get("t_mid", solp.get("t_col", [])), float)
        n_all = np.asarray(solp.get("n_mid", solp.get("n_col", [])), float)

        if isinstance(mask, slice):
            ds = ds_all; t = t_all; n = n_all
        else:
            ds = ds_all[mask]; t = t_all[mask]; n = n_all[mask]

        if ds.size != bI.size or t.shape[0] != bI.size or n.shape[0] != bI.size:
            return None

        # Integrated B-content on this edge segment
        B_edge = np.zeros(2, float)
        for k in range(bI.size):
            B_edge += (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])

        a_edge = float(extra.get("a_edge", 0.0))
        xloc = np.asarray(x_mid, float).reshape(-1,)
        if xloc.size != bI.size:
            xloc = np.linspace(-a_edge, a_edge, bI.size)

        # distance from selected tip
        if at == "end":
            r = a_edge - xloc
        else:
            r = a_edge + xloc

        order = np.argsort(r)
        m_eval = max(1, int(n_eval_panels))
        idx = order[:m_eval]
        x_eval = np.mean(xy[idx, :], axis=0).reshape(2,)

        stress_excl = getattr(self.res, "stress_field_global_excluding", None)
        stress_tot = getattr(self.res, "stress_field_global", None)

        if exclude_self:
            if not callable(stress_excl):
                return None
            sxx, syy, sxy = stress_excl(edge_index, np.array([x_eval[0]]), np.array([x_eval[1]]), add_remote=True)
        else:
            if not callable(stress_tot):
                return None
            sxx, syy, sxy = stress_tot(np.array([x_eval[0]]), np.array([x_eval[1]]), add_remote=True)

        sig = np.array([[float(sxx[0]), float(sxy[0])], [float(sxy[0]), float(syy[0])]], float)
        return self.pk_force_density_from_stress(sig, B_edge)
## Window integrated =======================================================================

    def F_PK_window(
        self,
        edge_index: int,
        at: str = "end",
        exclude_self: bool = True,
        window_panels: int = 0,
        window_frac: float = 0.15,
        min_panels: int = 6,
        normal_offset_frac: float = 0.0,
        **_ignored,
    ):
        """Window-integrated PK force near a selected tip.

        This is a *local* estimator of the configurational/PK force vector F near a tip.
        For a straight crack in pure Mode I, the component along the crack tangent satisfies:
            J1 ≈ F·ex,  and  KI = sqrt(E' * J1)
        where E' = E (plane stress) or E/(1-ν^2) (plane strain).

        Features
        --------
        - Panel-safe for cheb_spectral: indexes panel arrays only (length Np).
        - window selection can be:
            * window_panels > 0 (last/first N panels)
            * otherwise window_frac (arclength fraction near tip), with min_panels fallback
        - optional normal_offset_frac shifts the stress evaluation point off the midline:
            x_eval = x_mid + (normal_offset_frac * L_edge) * n_mid
          This reduces residual near-singular sensitivity.

        Returns
        -------
        F : (2,) np.ndarray or None
        """
        edge_index = int(edge_index)
        at = str(at).lower().strip()
        if at not in ("start", "end"):
            at = "end"

        sol = self.sol if isinstance(self.sol, dict) else {}
        e2p = sol.get("parametrized_edge_to_polyline", {})
        pid = e2p.get(edge_index, None)
        if pid is None:
            return None
        pid = int(pid)

        solps = sol.get("polyline_solutions", [])
        if pid < 0 or pid >= len(solps):
            return None
        poly = dict(solps[pid])

        # Physical densities (already singular-weighted in solver when representation='singular')
        bI  = np.asarray(poly.get("bI", []), float).reshape(-1,)
        bII = np.asarray(poly.get("bII", []), float).reshape(-1,)
        ds  = np.asarray(poly.get("ds", []), float).reshape(-1,)

        # Use panel-midpoint frames if present; else fall back to collocation and truncate to panels.
        if poly.get("t_mid") is not None and poly.get("n_mid") is not None:
            t = np.asarray(poly.get("t_mid"), float)
            n = np.asarray(poly.get("n_mid"), float)
        else:
            t = np.asarray(poly.get("t_col", []), float)
            n = np.asarray(poly.get("n_col", []), float)

        # Evaluation points: prefer x_mid (panel midpoints); else x_col (truncate)
        if poly.get("x_mid") is not None:
            xeval = np.asarray(poly.get("x_mid"), float)
        else:
            xeval = np.asarray(poly.get("x_col", []), float)

        # Common panel count
        N = min(len(ds), len(bI), len(bII), len(t), len(n), len(xeval))
        if N <= 0:
            return None

        ds = ds[:N]; bI = bI[:N]; bII = bII[:N]
        t = t[:N]; n = n[:N]; xeval = xeval[:N]

        # Window indices
        wp = int(window_panels) if window_panels is not None else 0
        if wp > 0:
            wp = min(wp, N)
            idxs = np.arange(N - wp, N, dtype=int) if at == "end" else np.arange(0, wp, dtype=int)
        else:
            # arclength-based window
            L = float(np.sum(ds))
            if not np.isfinite(L) or L <= 0.0:
                wp = min(max(int(min_panels), 1), N)
                idxs = np.arange(N - wp, N, dtype=int) if at == "end" else np.arange(0, wp, dtype=int)
            else:
                s_mid = np.cumsum(ds) - 0.5 * ds  # panel mid-arc from start
                r = (L - s_mid) if at == "end" else s_mid
                rmax = float(max(window_frac, 0.0)) * L
                m = (r >= 0.0) & (r <= rmax + 1e-15)
                if int(np.count_nonzero(m)) < int(min_panels):
                    wp = min(max(int(min_panels), 1), N)
                    idxs = np.arange(N - wp, N, dtype=int) if at == "end" else np.arange(0, wp, dtype=int)
                else:
                    idxs = np.where(m)[0].astype(int)

        # Stress evaluators
        stress_excl_panels = getattr(self.res, "stress_field_global_excluding_panels", None)
        stress_tot = getattr(self.res, "stress_field_global", None)
        if exclude_self and not callable(stress_excl_panels):
            return None
        if (not exclude_self) and (not callable(stress_tot)):
            return None

        # Normal offset (scaled by polyline length)
        L_edge = float(np.sum(ds)) if np.isfinite(np.sum(ds)) else 0.0
        off = float(normal_offset_frac) * L_edge

        F = np.zeros(2, float)
        for k in np.asarray(idxs, int):
            dB = (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])

            xk = xeval[k, :] + (off * n[k, :]) if off != 0.0 else xeval[k, :]
            X = np.array([float(xk[0])], float)
            Y = np.array([float(xk[1])], float)

            if exclude_self:
                skip = {int(k)}
                if k - 1 >= 0:
                    skip.add(int(k - 1))
                if k + 1 < N:
                    skip.add(int(k + 1))
                sxx, syy, sxy = stress_excl_panels(pid, skip, X, Y, add_remote=True)
            else:
                sxx, syy, sxy = stress_tot(X, Y, add_remote=True)

            sig = np.array([[float(sxx[0]), float(sxy[0])],
                            [float(sxy[0]), float(syy[0])]], float)
            F += self.pk_force_density_from_stress(sig, dB)

        return F
    def kink_angle_from_K(
        KI: float,
        KII: float,
        *,
        criterion: str = "max_hoop",
        return_degrees: bool = True,
        eps: float = 1e-30,
    ) -> float:
        """Compute crack propagation (kink) angle theta from (K_I, K_II).

        Parameters
        ----------
        KI, KII : float
            Mode I and Mode II stress intensity factors at the tip.
        criterion : str
            - "max_hoop" : Maximum hoop stress (Erdogan–Sih). Recommended default.
            - "sed"      : Minimum strain energy density (optional; uses a numeric minimization fallback).
        return_degrees : bool
            If True, return angle in degrees; otherwise radians.
        eps : float
            Small number to avoid division-by-zero.

        Returns
        -------
        theta : float
            Kink angle measured from the current crack tangent direction.
            Positive theta corresponds to rotation toward +normal direction.
        """
        KI = float(KI)
        KII = float(KII)
        crit = str(criterion).strip().lower()

        # --- Trivial / near-trivial cases ---
        if abs(KII) <= eps:
            # Pure mode I => straight ahead
            theta = 0.0
            return math.degrees(theta) if return_degrees else theta

        if abs(KI) <= eps and abs(KII) > eps:
            # Pure mode II: MTS gives theta = ±70.53° (≈ arctan(±1/√2)*2)
            theta = -math.pi / 2.0  # placeholder, will be overwritten below for max_hoop
            # We'll let the criterion formula handle it robustly.

        if crit in ("max_hoop", "mts", "max_circ", "max_circumferential"):
            # Maximum hoop stress (Erdogan–Sih)
            #
            # Closed-form:
            #   theta = 2 * atan( (KI ± sqrt(KI^2 + 8 KII^2)) / (4 KII) )
            #
            # We evaluate both roots and choose the smaller-magnitude angle
            # to avoid very sharp propagation kinks.
            #
            disc = KI * KI + 8.0 * KII * KII
            root = math.sqrt(disc)

            denom = 4.0 * KII
            if abs(denom) <= eps:
                denom = eps if denom >= 0 else -eps

            theta_plus = 2.0 * math.atan((KI + root) / denom)
            theta_minus = 2.0 * math.atan((KI - root) / denom)

            # Map to principal range [-pi, pi] for stability
            if theta_plus > math.pi:
                theta_plus -= 2.0 * math.pi
            if theta_plus < -math.pi:
                theta_plus += 2.0 * math.pi
            if theta_minus > math.pi:
                theta_minus -= 2.0 * math.pi
            if theta_minus < -math.pi:
                theta_minus += 2.0 * math.pi

            # Choose smallest-magnitude root to avoid sharp kink jumps.
            theta = theta_plus if abs(theta_plus) <= abs(theta_minus) else theta_minus

            return math.degrees(theta) if return_degrees else theta

        elif crit in ("sed", "min_sed", "min_energy_density"):
            # Minimum strain energy density criterion.
            # There are closed-form expressions in the literature, but they are more error-prone
            # with sign conventions. A robust implementation is a 1D numerical minimization
            # of the SED function over theta in a bounded interval.
            #
            # We minimize S(θ) over θ ∈ [-pi/2, pi/2] and return argmin.
            #
            # NOTE: This is an optional path. If you want, we can replace with a closed form
            # after you confirm sign conventions vs your local frame.
            def S(theta: float) -> float:
                c = math.cos(theta / 2.0)
                s = math.sin(theta / 2.0)
                # One common proportional form for SED near tip (up to constants) is:
                # S ∝ KI^2 * f1(theta) + KI*KII * f2(theta) + KII^2 * f3(theta)
                # Here we use a standard normalized form sufficient for minimization.
                # (This does not require E' because it cancels in argmin.)
                f1 = c * c * (1.0 - math.sin(theta) * math.sin(theta / 2.0))
                f2 = 2.0 * s * c * (2.0 * math.cos(theta) + 1.0)
                f3 = s * s * (9.0 - 7.0 * math.cos(theta))
                return (KI * KI) * f1 + (KI * KII) * f2 + (KII * KII) * f3

            # coarse-to-fine search
            a, b = -0.5 * math.pi, 0.5 * math.pi
            thetas = np.linspace(a, b, 2001)
            vals = np.array([S(float(t)) for t in thetas], float)
            j = int(np.nanargmin(vals))
            theta = float(thetas[j])

            return math.degrees(theta) if return_degrees else theta

        else:
            raise ValueError(f"Unknown criterion='{criterion}'. Use 'max_hoop' or 'sed'.")

    def kink_angle_from_PK_window(
        self,
        edge_index: int,
        *,
        at: str = "end",
        window_panels: int = 6,
        window_frac: float = 0.1,
        exclude_self: bool = True,
        plane_strain: bool = False,          # backward compatibility (ignored)
        criterion: str = "max_hoop",
        return_degrees: bool = True,
        mode_I_only: bool = False,
    ):
        """Kink angle from PK-window force with robust E' handling.

        Uses material.plane_stress (authoritative) for E'.
        If mode_I_only=True, returns KI=sqrt(E'*J1), KII=0.
        """
        F = self.F_PK_window(
            edge_index=edge_index,
            at=at,
            exclude_self=exclude_self,
            window_panels=window_panels,
            window_frac=window_frac,
        )
        if F is None:
            return None
        F = np.asarray(F, float).reshape(2,)

        # local frame
        _, _, _, extra = self.res.reconstruct_cod_csd_panel_midpoints(edge_index=edge_index, enforce_global_tip_zero=False)
        extra = dict(extra)
        ex = np.asarray(extra["ex"], float).reshape(2,)
        ey = np.asarray(extra["ey"], float).reshape(2,)

        J1 = float(np.dot(F, ex))
        J2 = float(np.dot(F, ey))

        E  = float(self.res.calc.material.E)
        nu = float(self.res.calc.material.nu)
        plane_stress = bool(getattr(self.res.calc.material, "plane_stress", False))
        Eprime = E if plane_stress else (E / (1.0 - nu**2))
        # plane strain: E' = E/(1-ν^2) = 2μ/(1-ν) since E=2μ(1+ν)

        if mode_I_only:
            KI = float(np.sqrt(max(Eprime * J1, 0.0)))
            KII = 0.0
        else:
            A = Eprime * J1
            C = -0.5 * Eprime * J2
            disc = max(A*A - 4.0*C*C, 0.0)
            D = float(np.sqrt(disc))
            KI2 = max(0.5*(A + D), 0.0)
            KII2 = max(0.5*(A - D), 0.0)
            KI  = float(np.sqrt(KI2))
            KII = float(np.sign(C) * np.sqrt(KII2))

        theta = self.kink_angle_from_K(KI, KII, criterion=criterion, return_degrees=return_degrees)
        meta = dict(F_global=F, J1=J1, J2=J2, Eprime=Eprime, plane_stress=plane_stress,
                    KI=KI, KII=KII, window_panels=window_panels, window_frac=window_frac,
                    exclude_self=exclude_self, plane_strain_arg=plane_strain, mode_I_only=mode_I_only,
                    edge_index=edge_index, at=at, criterion=criterion)
        return theta, KI, KII, meta
    
    def J_contour_tip(
        self,
        edge_index: int,
        *,
        tip: str = "end",
        R: float = 1e-3,
        n_theta: int = 181,
        fd_h: float | None = None,
        add_remote: bool = True,
    ) -> float | None:
        """Path-independent Rice J-integral on a circular contour around a tip (Mode-I usable).

        Uses stresses from res.stress_field_global and displacement gradients from FD on res.displacement_field_global.
        """
        edge_index = int(edge_index)
        tip = str(tip).lower().strip()
        if tip not in ("start", "end"):
            tip = "end"

        edge = self.res.calc.network.edges[edge_index]
        v0 = self.res.calc.network.V(edge.v0)
        v1 = self.res.calc.network.V(edge.v1)
        p0 = np.array([float(v0.x), float(v0.y)], float)
        p1 = np.array([float(v1.x), float(v1.y)], float)
        tvec = p1 - p0
        L = float(np.hypot(tvec[0], tvec[1]))
        if L <= 0:
            return None
        ex = tvec / L

        pt = p1 if tip == "end" else p0
        R = float(R)
        n_theta = int(max(65, n_theta))
        th = np.linspace(-np.pi, np.pi, n_theta)

        X = pt[0] + R * np.cos(th)
        Y = pt[1] + R * np.sin(th)
        nx = np.cos(th)
        ny = np.sin(th)
        ds = R * (2.0*np.pi / (n_theta - 1))

        sxx, syy, sxy = self.res.stress_field_global(X, Y, add_remote=bool(add_remote))
        sxx = np.asarray(sxx, float); syy = np.asarray(syy, float); sxy = np.asarray(sxy, float)

        E = float(self.res.calc.material.E)
        nu = float(self.res.calc.material.nu)
        plane_stress = bool(getattr(self.res.calc.material, "plane_stress", False))

        if plane_stress:
            exx = (sxx - nu*syy)/E
            eyy = (syy - nu*sxx)/E
            exy = (1.0+nu)/E * sxy
        else:
            Eprime = E/(1.0-nu**2)
            exx = (sxx - nu*syy)/Eprime
            eyy = (syy - nu*sxx)/Eprime
            exy = (1.0+nu)/Eprime * sxy

        W = 0.5*(sxx*exx + syy*eyy + 2.0*sxy*exy)

        if fd_h is None:
            fd_h = max(1e-8, 1e-4*R)

        duxdx, duxdy, duydx, duydy = self.res.displacement_grad_global_fd(X, Y, h=float(fd_h), add_remote=False)

        # derivative along local x (crack tangent)
        uxd = duxdx*ex[0] + duxdy*ex[1]
        uyd = duydx*ex[0] + duydy*ex[1]

        tx = sxx*nx + sxy*ny
        ty = sxy*nx + syy*ny

        integrand = W*nx - (tx*uxd + ty*uyd)
        return float(np.sum(integrand) * ds)

    def K_from_J_contour(
        self,
        edge_index: int,
        *,
        tip: str = "end",
        R: float = 1e-3,
        n_theta: int = 181,
        fd_h: float | None = None,
        add_remote: bool = True,
    ) -> tuple[float, dict] | None:
        """Mode-I K from contour J: KI = sqrt(E' * J)."""
        J = self.J_contour_tip(edge_index, tip=tip, R=R, n_theta=n_theta, fd_h=fd_h, add_remote=add_remote)
        if J is None:
            return None
        E = float(self.res.calc.material.E)
        nu = float(self.res.calc.material.nu)
        plane_stress = bool(getattr(self.res.calc.material, "plane_stress", False))
        Eprime = E if plane_stress else (E/(1.0-nu**2))
        KI = float(np.sqrt(max(Eprime*J, 0.0)))
        meta = dict(J=J, Eprime=Eprime, plane_stress=plane_stress, R=float(R), n_theta=int(n_theta), fd_h=float(fd_h) if fd_h is not None else None)
        return KI, meta
