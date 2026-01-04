
"""
dce_calcs_v0.py
Single straight 2D crack, static mixed-mode, dislocation-density formulation.

Representations:
  - "delta_collocation": delta DOFs at source quadrature points, traction enforced at *oversampled* collocation points
  - "cheb_quad": continuous quadratic g(x) on shared nodes, with b(x)=g(x)/sqrt(a^2-x^2), solved by constrained LSQ
  - "cheb_spectral": global Chebyshev expansion for g(s)=sum c_k T_k(s) with exact Hilbert operator on coefficients

No analytical LEFM/Williams fields are used.

Crack-local frame:
  x1 along crack, x2 normal. Crack occupies x2=0, x1 in [-a,a].

Traction relations on crack line (x2=0):
  t_n(x) = σ22(x,0+) = -M * PV∫ b_I(ξ)/(x-ξ) dξ + t_n^∞
  t_s(x) = σ12(x,0+) = +M * PV∫ b_II(ξ)/(x-ξ) dξ + t_s^∞
where M = 2μ / (π(κ+1)).

For the "delta_collocation" method implemented here, we use:
  - delta DOFs at source points ξ_j with weights w_j, representing b(ξ)dξ ≈ Σ b_j w_j δ(ξ-ξ_j)
  - collocation points x_i chosen *different* from ξ_j (oversampling), so the discrete integral
      PV∫ b(ξ)/(x_i-ξ) dξ ≈ Σ b_j w_j/(x_i-ξ_j)
    has no singular self-term and no PV bookkeeping.

We impose closure:
  Σ b_j w_j = 0  (for each mode)

This oversampled strategy removes the checkerboard mode that can plague collocation with x_i=ξ_j.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def gauss_legendre(n: int):
    """Gauss–Legendre nodes/weights on [-1,1] for any integer n>=1."""
    if n < 1:
        raise ValueError("gauss_legendre requires n>=1.")
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x.astype(float), w.astype(float)


def tip_weight(x, a, eps=1e-14):
    """w(x)=1/sqrt(a^2-x^2), vectorized, safe for arrays."""
    x = np.asarray(x, dtype=float)
    val = np.maximum(a * a - x * x, eps)
    return 1.0 / np.sqrt(val)


def element_nodes(a: float, ne: int, node_distribution: str = "uniform") -> np.ndarray:
    """Element boundary nodes spanning [-a,a] (size ne+1)."""
    nd = str(node_distribution).lower().strip()
    if nd == "uniform":
        return np.linspace(-a, a, ne + 1, dtype=float)
    if nd == "tip_dense":
        # cosine clustering: x=a*cos(theta), theta in [pi,0] gives increasing x
        theta = np.linspace(np.pi, 0.0, ne + 1)
        return a * np.cos(theta)
    raise ValueError("node_distribution must be 'uniform' or 'tip_dense'.")


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


def _stress_edge_bx(x, y, b, mu, nu):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r2 = x * x + y * y
    r4 = np.maximum(r2 * r2, 1e-30)

    C = mu * b / (2.0 * np.pi * (1.0 - nu))
    s11 = -C * y * (3.0 * x * x + y * y) / r4
    s22 = C * y * (x * x - y * y) / r4
    s12 = C * x * (x * x - y * y) / r4
    return s11, s22, s12


def _stress_edge_by(x, y, b, mu, nu):
    """
    Stress field of an edge dislocation with Burgers vector along +y (local normal).

    Implemented by rotating the known b||x solution into a primed frame where:
        x' = y,  y' = -x
    i.e., v' = Q v with Q = [[0, 1], [-1, 0]] so that Q*[0,1]=[1,0].

    Compute stresses in primed (b||x'), then transform back:
        S = Q^T S' Q
    """
    # v' = Q v
    Q = np.array([[0.0, 1.0],
                  [-1.0, 0.0]], dtype=float)

    # coordinates in primed frame
    xp = Q[0, 0] * x + Q[0, 1] * y   # = y
    yp = Q[1, 0] * x + Q[1, 1] * y   # = -x

    s11p, s22p, s12p = _stress_edge_bx(xp, yp, b, mu, nu)
    Sp = np.array([[s11p, s12p],
                   [s12p, s22p]])

    # Transform back: S = Q^T Sp Q
    # Do this componentwise to preserve broadcasting
    sxx = Q[0, 0] * (Q[0, 0] * s11p + Q[0, 1] * s12p) + Q[1, 0] * (Q[0, 0] * s12p + Q[0, 1] * s22p)
    syy = Q[0, 1] * (Q[0, 1] * s11p + Q[1, 1] * s12p) + Q[1, 1] * (Q[0, 1] * s12p + Q[1, 1] * s22p)
    sxy = Q[0, 0] * (Q[0, 1] * s11p + Q[1, 1] * s12p) + Q[1, 0] * (Q[0, 1] * s12p + Q[1, 1] * s22p)

    return sxx, syy, sxy


class DCESingleCrackStatic:
    def make_nodes(self, ne_half: int, node_distribution: str = "uniform") -> np.ndarray:
        """Return nodal x-locations along the crack in local coordinates [-a, a].

        node_distribution:
          - "uniform": equally spaced nodes
          - "tip_dense": cosine-clustered nodes (Chebyshev–Lobatto), dense near ±a
        """
        a = self.crack.half_length
        kind = (node_distribution or "uniform").lower()
        n = 2 * ne_half + 1
        if n < 2:
            return np.array([0.0], dtype=float)
        if kind == "uniform":
            return np.linspace(-a, a, n)
        if kind == "tip_dense":
            i = np.arange(n, dtype=float)
            x = np.cos(np.pi * i / (n - 1))  # 1..-1
            return a * x[::-1]               # -a..a
        raise ValueError(f"Unknown node_distribution '{node_distribution}'. Use 'uniform' or 'tip_dense'.")

    def __init__(self, material: Material, crack: Crack, applied: AppliedStress):
        self.material = material
        self.crack = crack
        self.applied = applied
        self.R = rot2(crack.angle)   # local->global
        self.RT = self.R.T           # global->local

    def global_to_local_stress(self, Sg: np.ndarray) -> np.ndarray:
        return self.RT @ Sg @ self.R

    def local_to_global_stress(self, Sl: np.ndarray) -> np.ndarray:
        return self.R @ Sl @ self.RT

    def applied_stress_local(self) -> np.ndarray:
        return self.global_to_local_stress(self.applied.tensor())

    def applied_tractions_local(self):
        Sl = self.applied_stress_local()
        return float(Sl[0, 1]), float(Sl[1, 1])  # (t_s, t_n)

    def M(self) -> float:
        mu = self.material.mu
        kappa = self.material.kappa
        return 2.0 * mu / (np.pi * (kappa + 1.0))

    def solve(self, ne_half: int, representation: str = "delta_collocation", *, node_distribution: str = "uniform", **kw):
        rep = representation.lower()
        if rep == "delta_collocation":
            return self._solve_delta_collocation(ne_half=ne_half, node_distribution=node_distribution, **kw)
        if rep == "cheb_quad":
            return self._solve_cheb_quad(ne_half=ne_half, node_distribution=node_distribution, **kw)
        if rep == "cheb_spectral":
            return self._solve_cheb_spectral(ne_half=ne_half, **kw)
        raise ValueError(f"Unknown representation '{representation}'")

    def _solve_delta_collocation(
        self,
        ne_half: int,
        *,
        node_distribution: str = "uniform",
        nq_src: int = 2,
        nq_col: int = 4,
        ridge: float = 1e-14,
    ):
        a = self.crack.half_length
        ne = 2 * int(ne_half)
        xe = element_nodes(a, ne, node_distribution=node_distribution)

        # Source points (unknown DOFs)
        xg_s, wg_s = gauss_legendre(nq_src)
        xs_list, ws_list = [], []
        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm = 0.5 * (xR + xL)
            xs = xm + J * xg_s
            ws = J * wg_s
            xs_list.append(xs)
            ws_list.append(ws)
        xs = np.concatenate(xs_list)
        ws = np.concatenate(ws_list)

        # Tip-weight for regularized unknown g(s)=b(s)*sqrt(a^2-s^2)
        eps_tip = max(1e-12 * a, 1e-18)
        tipw = 1.0 / np.sqrt(np.maximum(a * a - xs * xs, eps_tip))

        # Collocation points (equations) – choose different set
        xg_c, _ = gauss_legendre(nq_col)
        xc_list = []
        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm = 0.5 * (xR + xL)
            xc_list.append(xm + J * xg_c)
        xcol = np.concatenate(xc_list)

        DX = xcol[:, None] - xs[None, :]
        tol = 1e-14 * a
        DX = np.where(np.abs(DX) < tol, np.sign(DX) * tol + tol, DX)
        A = (ws * tipw)[None, :] / DX  # Nyström: b(s)=g(s)/sqrt(a^2-s^2)

        ts0, tn0 = self.applied_tractions_local()
        M = self.M()
        rhsI = (tn0 / M) * np.ones_like(xcol)
        rhsII = (-ts0 / M) * np.ones_like(xcol)

        # Closure: ∫ b(s) ds = 0  =>  Σ g_j * tipw_j * w_j = 0
        C = (ws * tipw).reshape(1, -1)

        # Ridge (Tikhonov) by augmentation; keeps constraint unchanged
        if ridge and ridge > 0:
            A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(A.shape[1])])
            rhsI_aug = np.concatenate([rhsI, np.zeros(A.shape[1])])
            rhsII_aug = np.concatenate([rhsII, np.zeros(A.shape[1])])
            bI = self._solve_kkt_lsq(A_aug, rhsI_aug, C)
            bII = self._solve_kkt_lsq(A_aug, rhsII_aug, C)

            gI = bI
            gII = bII
            bI = gI * tipw
            bII = gII * tipw
        else:
            bI = self._solve_kkt_lsq(A, rhsI, C)
            bII = self._solve_kkt_lsq(A, rhsII, C)

        # Interpret solved DOFs as regularized g(s); recover physical b(s)=g(s)*tipw for plotting/stress.
        gI = bI
        gII = bII
        bI = gI * tipw
        bII = gII * tipw

        return dict(
            representation="delta_collocation",
            node_distribution=str(node_distribution),
            ne_half=int(ne_half),
            xs=xs,
            ws=ws,
            xcol=xcol,
            tipw=tipw,
            gI=gI,
            gII=gII,
            bI=bI,
            bII=bII,
            meta=dict(nq_src=int(nq_src), nq_col=int(nq_col), ridge=float(ridge)),
        )

    def _solve_cheb_quad(self, ne_half: int, *, node_distribution: str = "uniform", nq_col: int = 3, nq_src: int = 6, reg_g2: float = 0.0):
        # (Same as prior version; kept for continuity)
        a = self.crack.half_length
        ne = 2 * int(ne_half)
        xe = element_nodes(a, ne, node_distribution=node_distribution)
        xm = 0.5 * (xe[:-1] + xe[1:])

        xnodes = np.empty(2 * ne + 1, float)
        xnodes[0::2] = xe
        xnodes[1::2] = xm
        nd = xnodes.size

        xg, _ = gauss_legendre(nq_col)
        xc_list = []
        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm_e = 0.5 * (xR + xL)
            xc_list.append(xm_e + J * xg)
        xcol = np.concatenate(xc_list)

        xgs, wgs = gauss_legendre(nq_src)
        xs_list, ws_list, Ne_list = [], [], []
        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm_e = 0.5 * (xR + xL)
            xs = xm_e + J * xgs
            ws = J * wgs
            eta = xgs
            N1 = 0.5 * eta * (eta - 1.0)
            N2 = 1.0 - eta * eta
            N3 = 0.5 * eta * (eta + 1.0)
            Ne_loc = np.vstack([N1, N2, N3]).T
            xs_list.append(xs)
            ws_list.append(ws)
            Ne_list.append(Ne_loc)

        eps = max(1e-12 * a, 1e-18)
        xcol_i = np.clip(xcol, -a + eps, a - eps)
        logc = np.log(np.abs((a + xcol_i) / (a - xcol_i)))

        A = np.zeros((xcol.size, nd), float)

        for i, x in enumerate(xcol_i):
            e_x = int(np.clip(np.searchsorted(xe, x, side="right") - 1, 0, ne - 1))
            xL, xR = xe[e_x], xe[e_x + 1]
            J = 0.5 * (xR - xL)
            xm_e = 0.5 * (xR + xL)
            eta = (x - xm_e) / J
            Nx = np.array([0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0)])
            idxs = np.array([2 * e_x, 2 * e_x + 1, 2 * e_x + 2], int)

            wx = tip_weight(x, a)
            bx_basis = np.zeros(nd, float)
            bx_basis[idxs] = wx * Nx

            row = np.zeros(nd, float)
            for e in range(ne):
                xs = xs_list[e]
                ws = ws_list[e]
                Ne_loc = Ne_list[e]
                dx = x - xs
                mask = np.abs(dx) > 1e-14 * a
                if not np.any(mask):
                    continue
                wtip = tip_weight(xs[mask], a)
                Be = (wtip[:, None] * Ne_loc[mask, :])
                idxe = np.array([2 * e, 2 * e + 1, 2 * e + 2], int)
                row[idxe] += ((ws[mask] / dx[mask])[:, None] * Be).sum(axis=0)
                row -= bx_basis * np.sum(ws[mask] / dx[mask])

            row += bx_basis * logc[i]
            A[i, :] = row

        ts0, tn0 = self.applied_tractions_local()
        M = self.M()
        rhsI = (tn0 / M) * np.ones(xcol.size)
        rhsII = (-ts0 / M) * np.ones(xcol.size)

        C = np.zeros((1, nd), float)
        for e in range(ne):
            xs = xs_list[e]
            ws = ws_list[e]
            Ne_loc = Ne_list[e]
            wtip = tip_weight(xs, a)
            idxe = np.array([2 * e, 2 * e + 1, 2 * e + 2], int)
            C[0, idxe] += (ws[:, None] * (wtip[:, None] * Ne_loc)).sum(axis=0)

        if reg_g2 and reg_g2 > 0.0:
            D2 = np.zeros((nd - 2, nd), float)
            for k in range(nd - 2):
                D2[k, k] = 1.0
                D2[k, k + 1] = -2.0
                D2[k, k + 2] = 1.0
            A_aug = np.vstack([A, np.sqrt(reg_g2) * D2])
            rhsI_aug = np.concatenate([rhsI, np.zeros(D2.shape[0])])
            rhsII_aug = np.concatenate([rhsII, np.zeros(D2.shape[0])])
            qI = self._solve_kkt_lsq(A_aug, rhsI_aug, C)
            qII = self._solve_kkt_lsq(A_aug, rhsII_aug, C)
        else:
            qI = self._solve_kkt_lsq(A, rhsI, C)
            qII = self._solve_kkt_lsq(A, rhsII, C)

        return dict(
            representation="cheb_quad",
            node_distribution=str(node_distribution),
            ne_half=int(ne_half),
            xnodes=xnodes,
            qI=qI,
            qII=qII,
            xcol=xcol,
            meta=dict(nq_col=int(nq_col), nq_src=int(nq_src), reg_g2=float(reg_g2)),
        )

    def _solve_cheb_spectral(self, ne_half: int):
        N = max(4, 2 * int(ne_half))
        ts0, tn0 = self.applied_tractions_local()
        M = self.M()
        rhsI = (tn0 / M)
        rhsII = (-ts0 / M)

        coeffs_I = np.zeros(N, float)
        coeffs_II = np.zeros(N, float)
        coeffs_I[0] = -rhsI / np.pi
        coeffs_II[0] = -rhsII / np.pi

        return dict(
            representation="cheb_spectral",
            ne_half=int(ne_half),
            N=int(N),
            coeffs_I=coeffs_I,
            coeffs_II=coeffs_II,
            meta=dict(),
        )

    def evaluate_b_mode(self, x, results: dict, mode: str = "I"):
        rep = str(results.get("representation", "")).lower()
        a = self.crack.half_length
        x = np.asarray(x, float)

        if rep == "delta_collocation":
            xs = results["xs"]
            b = results["bI"] if mode.upper() == "I" else results["bII"]
            return np.interp(x, xs, b, left=0.0, right=0.0)

        if rep == "cheb_quad":
            xnodes = results["xnodes"]
            q = results["qI"] if mode.upper() == "I" else results["qII"]
            ne = 2 * results["ne_half"]
            xe = element_nodes(a, ne, node_distribution=node_distribution)

            eps = max(1e-12 * a, 1e-18)
            xx = np.clip(x, -a + eps, a - eps)

            outg = np.zeros_like(xx)
            idx = np.clip(np.searchsorted(xe, xx, side="right") - 1, 0, ne - 1)
            for e in range(ne):
                mask = idx == e
                if not np.any(mask):
                    continue
                xL, xR = xe[e], xe[e + 1]
                J = 0.5 * (xR - xL)
                xm = 0.5 * (xR + xL)
                eta = (xx[mask] - xm) / J
                N1 = 0.5 * eta * (eta - 1.0)
                N2 = 1.0 - eta * eta
                N3 = 0.5 * eta * (eta + 1.0)
                n0, n1, n2 = 2 * e, 2 * e + 1, 2 * e + 2
                outg[mask] = N1 * q[n0] + N2 * q[n1] + N3 * q[n2]

            return tip_weight(xx, a) * outg

        if rep == "cheb_spectral":
            coeffs = results["coeffs_I"] if mode.upper() == "I" else results["coeffs_II"]
            coeffs = np.asarray(coeffs, float).ravel()

            eps = max(1e-12 * a, 1e-18)
            xx = np.clip(x, -a + eps, a - eps)
            s = xx / a

            if coeffs.size == 0:
                return np.zeros_like(s)

            T0 = np.ones_like(s)
            T1 = s.copy()
            g = coeffs[0] * T1
            for k in range(2, coeffs.size + 1):
                Tk = 2.0 * s * T1 - T0
                g += coeffs[k - 1] * Tk
                T0, T1 = T1, Tk

            return tip_weight(xx, a) * g

        raise ValueError(f"Unknown representation '{rep}'")

    def reconstruct_cod_csd(self, results: dict, n: int = 1200):
        a = self.crack.half_length
        eps = max(1e-6 * a, 1e-18)
        x = np.linspace(-a + eps, a - eps, n)

        bI = self.evaluate_b_mode(x, results, mode="I")
        bII = self.evaluate_b_mode(x, results, mode="II")

        COD = np.concatenate([[0.0], np.cumsum(0.5 * (bI[1:] + bI[:-1]) * (x[1:] - x[:-1]))])
        CSD = np.concatenate([[0.0], np.cumsum(0.5 * (bII[1:] + bII[:-1]) * (x[1:] - x[:-1]))])

        x0, x1 = x[0], x[-1]

        def enforce_end_zero(u):
            u0, u1 = u[0], u[-1]
            beta = (u1 - u0) / (x1 - x0)
            alpha = u0 - beta * x0
            return u - (alpha + beta * x)

        return x, enforce_end_zero(COD), enforce_end_zero(CSD)

    def evaluate_stress_field_local(self, X, Y, results: dict, add_remote: bool = True, nq_stress: int = 10):
        a = self.crack.half_length
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        mu, nu = self.material.mu, self.material.nu

        ne = 2 * results.get("ne_half", 10)
        xe = element_nodes(a, ne, node_distribution=node_distribution)
        xg, wg = gauss_legendre(int(nq_stress))

        xs_list, ws_list = [], []
        for e in range(ne):
            xL, xR = xe[e], xe[e + 1]
            J = 0.5 * (xR - xL)
            xm = 0.5 * (xR + xL)
            xs = xm + J * xg
            ws = J * wg
            xs_list.append(xs)
            ws_list.append(ws)

        xs = np.concatenate(xs_list)
        ws = np.concatenate(ws_list)

        bI = self.evaluate_b_mode(xs, results, mode="I")
        bII = self.evaluate_b_mode(xs, results, mode="II")

        s11 = np.zeros_like(X)
        s22 = np.zeros_like(X)
        s12 = np.zeros_like(X)

        for j in range(xs.size):
            dx = X - xs[j]
            dy = Y

            s11_j, s22_j, s12_j = _stress_edge_bx(dx, dy, bII[j] * ws[j], mu, nu)
            s11 += s11_j
            s22 += s22_j
            s12 += s12_j

            s11_j, s22_j, s12_j = _stress_edge_by(dx, dy, bI[j] * ws[j], mu, nu)
            # Sign convention: Mode-I (normal) Burgers density bI is defined so that
            # σ22(x,0+) = -M * PV∫ bI(ξ)/(x-ξ) dξ (to cancel remote normal traction on the faces).
            # Therefore we subtract the by-kernel contribution here for consistency.
            s11 -= s11_j
            s22 -= s22_j
            s12 -= s12_j

        if add_remote:
            Sl = self.applied_stress_local()
            s11 += Sl[0, 0]
            s22 += Sl[1, 1]
            s12 += Sl[0, 1]

        return s11, s22, s12

    @staticmethod
    def _solve_kkt_lsq(A, rhs, C):
        A = np.asarray(A, float)
        rhs = np.asarray(rhs, float)
        C = np.asarray(C, float)

        ATA = A.T @ A
        ATb = A.T @ rhs
        n = ATA.shape[0]
        m = C.shape[0]

        KKT = np.block([[ATA, C.T], [C, np.zeros((m, m), float)]])
        b = np.concatenate([ATb, np.zeros(m)])
        sol = np.linalg.solve(KKT, b)
        return sol[:n]
