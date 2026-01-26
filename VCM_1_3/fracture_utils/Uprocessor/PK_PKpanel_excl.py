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
        stress_excl_panels = getattr(self.res, "stress_field_global_excluding_panels", None)
        stress_tot = getattr(self.res, "stress_field_global", None)

        # Build mapping from *local* edge-segment panel index -> *global* polyline panel index
        # For now, assume direct 1:1 mapping (same indices)
        orig_idx = np.arange(len(ds), dtype=int)

        if exclude_self:
            # Correct PK exclusion: keep the rest of the same polyline; skip only self/near-self panels
            if not callable(stress_excl_panels):
                return None
            if pid < 0:
                return None
        else:
            if not callable(stress_tot):
                return None

        # Integrate window PK contributions
        F = np.zeros(2, float)
        for k in np.asarray(idxs, int):
            dB = (bII[k] * t[k] + bI[k] * n[k]) * float(ds[k])

            # Evaluate stress at panel midpoint; optionally exclude only self/near-self panels
            if exclude_self:
                j0 = int(orig_idx[int(k)])
                # Skip self plus immediate neighbors for robustness
                skip = {j0}
                if j0 - 1 >= 0:
                    skip.add(j0 - 1)
                if j0 + 1 < len(ds):
                    skip.add(j0 + 1)

                sxx, syy, sxy = stress_excl_panels(
                    pid,
                    skip,
                    np.array([xy[k, 0]]),
                    np.array([xy[k, 1]]),
                    add_remote=True,
                )
            else:
                sxx, syy, sxy = stress_tot(
                    np.array([xy[k, 0]]),
                    np.array([xy[k, 1]]),
                    add_remote=True,
                )

            sig = np.array([[float(sxx[0]), float(sxy[0])], [float(sxy[0]), float(syy[0])]], float)
            F += self.pk_force_density_from_stress(sig, dB)

        return F
    

    @staticmethod
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
            # Choose the root that maximizes hoop stress; practically:
            #   use "+" sign in numerator when KII > 0 and "-" when KII < 0
            # which is equivalent to:
            #   theta = 2*atan( (KI + s*sqrt(KI^2+8KII^2)) / (4KII) ), with s = sign(KII)
            #
            s = 1.0 if KII >= 0.0 else -1.0
            disc = KI * KI + 8.0 * KII * KII
            root = math.sqrt(disc)

            denom = 4.0 * KII
            # robust ratio
            ratio = (KI + s * root) / (denom if abs(denom) > eps else (eps if denom >= 0 else -eps))
            theta = 2.0 * math.atan(ratio)

            # Map to principal range [-pi, pi] for stability
            if theta > math.pi:
                theta -= 2.0 * math.pi
            if theta < -math.pi:
                theta += 2.0 * math.pi

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
        window_frac: float = 0.1,     # fallback if window_panels is ignored
        exclude_self: bool = True,
        plane_strain: bool = False,
        criterion: str = "max_hoop",
        return_degrees: bool = True,
    ):
        """
        Compute crack propagation (kink) angle using:

            F_PK_window -> (J1,J2) -> (KI,KII) -> kink criterion

        Returns
        -------
        theta : float
            Kink angle measured from the local crack tangent.
        KI, KII : float
            Stress intensity factors used for the criterion.
        meta : dict
            Diagnostics (Ft, Fn, J1, J2, Eprime, etc.)
        """

        # -------------------------------
        # 1) Compute PK-window force
        # -------------------------------
        try:
            F = self.F_PK_window(
                edge_index=edge_index,
                at=at,
                exclude_self=exclude_self,
                window_panels=window_panels,
                window_frac=window_frac,
            )
        except TypeError:
            # backward compatibility if F_PK_window has no window_panels
            F = self.F_PK_window(
                edge_index=edge_index,
                at=at,
                exclude_self=exclude_self,
                window_frac=window_frac,
            )

        if F is None:
            return None

        F = np.asarray(F, float).reshape(2,)

        # -------------------------------
        # 2) Local crack frame
        # -------------------------------
        _, _, _, extra = self.res.reconstruct_cod_csd_panel_midpoints(
            edge_index=edge_index,
            enforce_global_tip_zero=False,
        )
        extra = dict(extra)

        ex = np.asarray(extra["ex"], float).reshape(2,)   # tangent
        ey = np.asarray(extra["ey"], float).reshape(2,)   # normal

        Ft = float(np.dot(F, ex))   # J1
        Fn = float(np.dot(F, ey))   # J2

        # -------------------------------
        # 3) Convert (J1,J2) -> (KI,KII)
        # -------------------------------
        E  = float(self.res.calc.material.E)
        nu = float(self.res.calc.material.nu)
        Eprime = E / (1.0 - nu**2) if plane_strain else E

        J1 = Ft
        J2 = Fn

        A = Eprime * J1                # KI^2 + KII^2
        C = -0.5 * Eprime * J2         # KI*KII   (sign convention)

        disc = A*A - 4.0*C*C
        disc = max(disc, 0.0)
        D = float(np.sqrt(disc))

        KI2  = 0.5 * (A + D)
        KII2 = 0.5 * (A - D)
        KI2  = max(KI2,  0.0)
        KII2 = max(KII2, 0.0)

        KI  = float(np.sqrt(KI2))
        KII = float(np.sign(C) * np.sqrt(KII2))   # KI ≥ 0 by convention

        # -------------------------------
        # 4) Kink angle from criterion
        # -------------------------------
        theta = self.kink_angle_from_K(
            KI, KII,
            criterion=criterion,
            return_degrees=return_degrees,
        )

        meta = dict(
            F_global=F,
            Ft=Ft, Fn=Fn,
            J1=J1, J2=J2,
            Eprime=Eprime,
            KI=KI, KII=KII,
            window_panels=window_panels,
            window_frac=window_frac,
            exclude_self=exclude_self,
            plane_strain=plane_strain,
            criterion=criterion,
            edge_index=edge_index,
            at=at,
        )

        return theta, KI, KII, meta

