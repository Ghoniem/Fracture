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
        cod_window_panels: int = 0,
        csd_window_panels: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (P, J) arrays along an edge.

        P: (N,2) points on the *midline* of the edge in global coordinates.
        J: (N,2) jump vectors in global coordinates at the same param points.

        This uses COD/CSD reconstructions (kinematics). It never uses deltaJ_branch.
        """
        edge_index = int(edge_index)
        # Use the parametrized reconstruction helper if applicable
        if getattr(self.res, "is_parametrized", lambda: False)():
            from .plot_reconstruct import reconstruct_cod_csd_parametrized_smoothed

            x, COD, CSD = reconstruct_cod_csd_parametrized_smoothed(
                self.res,
                edge_index=edge_index,
                n_pts=int(max(400, n_theta)),
                enforce_global_tip_zero=bool(enforce_global_tip_zero),
                cod_window_panels=int(cod_window_panels),
                csd_window_panels=int(csd_window_panels),
            )
        else:
            edge_res = self.res.edge(edge_index)
            x, COD, CSD = edge_res.reconstruct_cod_csd(n_theta=int(n_theta), enforce_tip_zero=bool(enforce_global_tip_zero))

        x = np.asarray(x, float).reshape(-1,)
        COD = np.asarray(COD, float).reshape(-1,)
        CSD = np.asarray(CSD, float).reshape(-1,)

        # Geometry for this edge
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

        # Midline points: center + x*ex
        c = 0.5 * (p0 + p1)
        P = c.reshape(1, 2) + x.reshape(-1, 1) * ex.reshape(1, 2)

        # Jump in global coords from local (CSD,COD)
        J = CSD.reshape(-1, 1) * ex.reshape(1, 2) + COD.reshape(-1, 1) * ey.reshape(1, 2)
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
