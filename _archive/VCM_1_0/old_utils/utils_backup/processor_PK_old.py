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

import numpy as np


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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (P, J) arrays along an edge.

        P: (N,2) points on the *midline* of the edge in global coordinates.
        J: (N,2) jump vectors in global coordinates at the same param points.

        This uses COD/CSD reconstructions (kinematics). It never uses deltaJ_branch.

        Parameters
        ----------
        tip_point:
            If provided, this function will apply a *constant* gauge shift so that the
            reconstructed jump satisfies J(tip_point) = 0 (in the discrete sense at the
            nearest end of the returned arrays). This is not a solver modification; it
            only fixes the integration constant used in reconstruction.
        """
        edge_index = int(edge_index)

        # -----------------------------------------------------------------
        # IMPORTANT (symmetry / orientation correctness)
        # -----------------------------------------------------------------
        # For parametrized cracks, COD/CSD returned by reconstruct_cod_csd_parametrized_smoothed
        # are expressed in the *directed polyline-segment* frame (v_start -> v_end) for that edge.
        # Therefore, when mapping (CSD,COD) back to a global jump vector J(s), we must use the
        # SAME directed segment basis, NOT the raw network edge ordering (edge.v0 -> edge.v1),
        # which may be opposite and causes mirrored branches to cross.
        # -----------------------------------------------------------------

        if getattr(self.res, "is_parametrized", lambda: False)():
            # NOTE:
            # For plotting *geometric* jump, we must use COD/CSD returned by the solver's
            # geometric reconstruction (panel-midpoint COD/CSD). The helper
            # reconstruct_cod_csd_parametrized_smoothed integrates densities to build a
            # global J(s) and then projects it; that object is gauge-dependent along a
            # polyline and can drift (exactly the artifact you observed at the right tip).
            if not hasattr(self.res, "reconstruct_cod_csd_panel_midpoints"):
                raise AttributeError(
                    "Parametrized plotting requires res.reconstruct_cod_csd_panel_midpoints() "
                    "to obtain geometric COD/CSD."
                )

            # IMPORTANT: we request the solver's *geometric* COD/CSD and then form
            # J = CSD*ex + COD*ey. Any remaining constant offset is a gauge.
            x_mid, COD, CSD, extra = self.res.reconstruct_cod_csd_panel_midpoints(
                edge_index=edge_index,
                enforce_global_tip_zero=bool(enforce_global_tip_zero),
            )
            x_mid = np.asarray(x_mid, float).reshape(-1,)
            COD = np.asarray(COD, float).reshape(-1,)
            CSD = np.asarray(CSD, float).reshape(-1,)

            # Local basis vectors for this edge (global coordinates) returned by the solver
            ex = np.asarray(extra.get("ex"), float).reshape(2,)
            ey = np.asarray(extra.get("ey"), float).reshape(2,)

            # Midline points in global coordinates (x_mid is along ex from the edge center)
            edge = self.calc.network.edges[edge_index]
            v0 = self.calc.network.V(edge.v0)
            v1 = self.calc.network.V(edge.v1)
            p0 = np.array([float(v0.x), float(v0.y)], float)
            p1 = np.array([float(v1.x), float(v1.y)], float)
            c = 0.5 * (p0 + p1)
            P = c.reshape(1, 2) + x_mid.reshape(-1, 1) * ex.reshape(1, 2)

            # Geometric jump in global coordinates
            J = CSD.reshape(-1, 1) * ex.reshape(1, 2) + COD.reshape(-1, 1) * ey.reshape(1, 2)

            # Optional gauge fix: enforce J=0 at the degree-1 tip end (nearest array end).
            if tip_point is not None and P.shape[0] >= 2:
                tp = np.asarray(tip_point, float).reshape(2,)
                d0 = float(np.hypot(*(P[0, :] - tp)))
                d1 = float(np.hypot(*(P[-1, :] - tp)))
                J_tip = J[0, :].copy() if d0 <= d1 else J[-1, :].copy()
                J = J - J_tip.reshape(1, 2)

            return P, J

        # Non-parametrized: edge_res reconstruction uses the network edge basis
        edge_res = self.res.edge(edge_index)
        x, COD, CSD = edge_res.reconstruct_cod_csd(
            n_theta=int(n_theta),
            enforce_tip_zero=bool(enforce_global_tip_zero),
        )

        x = np.asarray(x, float).reshape(-1,)
        COD = np.asarray(COD, float).reshape(-1,)
        CSD = np.asarray(CSD, float).reshape(-1,)

        edge = self.calc.network.edges[edge_index]
        v0 = self.calc.network.V(edge.v0)
        v1 = self.calc.network.V(edge.v1)
        p0 = np.array([float(v0.x), float(v0.y)], float)
        p1 = np.array([float(v1.x), float(v1.y)], float)
        t = p1 - p0
        L = float(np.hypot(t[0], t[1]))
        if L <= 0:
            ex = np.array([1.0, 0.0], float)
            ey = np.array([0.0, 1.0], float)
        else:
            ex = t / L
            ey = np.array([-ex[1], ex[0]], float)

        c = 0.5 * (p0 + p1)
        P = c.reshape(1, 2) + x.reshape(-1, 1) * ex.reshape(1, 2)
        J = CSD.reshape(-1, 1) * ex.reshape(1, 2) + COD.reshape(-1, 1) * ey.reshape(1, 2)

        if tip_point is not None and P.shape[0] >= 2:
            tp = np.asarray(tip_point, float).reshape(2,)
            d0 = float(np.hypot(*(P[0, :] - tp)))
            d1 = float(np.hypot(*(P[-1, :] - tp)))
            J_tip = J[0, :].copy() if d0 <= d1 else J[-1, :].copy()
            J = J - J_tip.reshape(1, 2)
        return P, J

    # ---------------------------------------------------------------------
    # Optional PK-style force from stress excluding self (best-effort)
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
