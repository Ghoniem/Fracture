
"""
Clean 2D BEM solver (constant elements, midpoint collocation) with constant-strain enrichment
=========================================================================================

Why this version exists
----------------------
Your results show:

- Benchmark A (rigid translation) PASS  -> boundary operator/closure is OK.
- Benchmark B (uniform pressure) FAIL   -> the discrete constant-element displacement BIE
  (as implemented) cannot reproduce a constant hydrostatic stress field accurately unless
  we include a constant-strain "particular solution" (a standard BEM enrichment / patch test fix).

This file keeps the stable v8 block-row-sum diagonal closure and adds a **3-parameter
constant-strain mode** (εxx, εyy, εxy). This mode generates a constant stress field
and corresponding linear displacement field that satisfies Navier's equation in the domain.

We solve for boundary unknowns AND (optionally) the 3 strain parameters by augmenting
the boundary system with their contributions.

Key benefit:
- Uniform pressure in a circular domain becomes representable (constant hydrostatic stress).
- This does not break rigid translation.

Drop-in replacement:
- Save as fracture_utils/Ubem/bem_solver.py
- API matches your harness: solve(gauss_n=8), add_element(...)

Notes on the "uniform pressure" benchmark:
- Pure Neumann problems have rigid displacement nullspace; we add 3 weak constraints on u
  (mean ux, mean uy, mean rotation). These constraints do not affect stresses.

"""

from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

VERSION = "bem_solver_clean_final_enrichment_signfix"

def solver_info():
    import os
    return {"version": VERSION, "file": __file__, "cwd": os.getcwd()}


@dataclass
class Segment:
    x1: float; y1: float
    x2: float; y2: float
    xm: float; ym: float
    nx: float; ny: float
    L: float


def _gauss_legendre(n: int):
    return np.polynomial.legendre.leggauss(int(n))


def _material(E: float, nu: float, plane_strain: bool):
    mu = E / (2.0 * (1.0 + nu))
    if plane_strain:
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        kappa = 3.0 - 4.0 * nu
    else:
        lam = 2.0 * mu * nu / (1.0 - nu)
        kappa = (3.0 - nu) / (1.0 + nu)
    return lam, mu, kappa


def kelvin_U(field: Tuple[float, float],
             source: Tuple[float, float],
             *,
             E: float,
             nu: float,
             plane_strain: bool,
             r0: float) -> np.ndarray:
    x, y = field
    xs, ys = source
    dx = x - xs
    dy = y - ys
    r2 = dx*dx + dy*dy + 1e-30
    r = np.sqrt(r2)

    _, mu, kappa = _material(E, nu, plane_strain)
    coef = 1.0 / (8.0 * np.pi * mu)
    logr = np.log(r / max(r0, 1e-30))

    U = np.zeros((2, 2), dtype=float)
    U[0, 0] = coef * ((kappa - 1.0) * logr + dx*dx / r2)
    U[1, 1] = coef * ((kappa - 1.0) * logr + dy*dy / r2)
    U[0, 1] = coef * (dx*dy / r2)
    U[1, 0] = U[0, 1]
    return U



def kelvin_T(
    field: tuple[float, float],
    source: tuple[float, float],
    n_source: tuple[float, float],
    *,
    E: float,
    nu: float,
    plane_strain: bool = True,
    r0: float = 0.0,
) -> np.ndarray:
    """
    Kelvin traction fundamental solution in 2D elasticity.

    Returns T (2x2) such that for a point force f at 'field' the traction at
    boundary point 'source' with outward normal n_source is:
        t = T @ f

    Conventions:
      - r = source - field
      - U uses the standard 2D Kelvin displacement kernel
      - traction is t_i = sigma_{ij} n_j computed from strains derived from U.
    """
    xs, ys = source
    xf, yf = field
    rx = xs - xf
    ry = ys - yf
    r2 = rx * rx + ry * ry
    if r0 and r2 < r0 * r0:
        r2 = r0 * r0
    r = math.sqrt(r2)
    inv_r2 = 1.0 / r2
    inv_r4 = inv_r2 * inv_r2

    mu = E / (2.0 * (1.0 + nu))
    if plane_strain:
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        lam = E * nu / (1.0 - nu * nu)

    nx, ny = n_source

    # dU_{ik} / dx_j at point x = source, with respect to x (source) coordinates.
    # U_{ik} = A [ (kappa-1) ln r δ_ik + r_i r_k / r^2 ], with A = 1/(8πμ)
    kappa = (3.0 - 4.0 * nu) if plane_strain else (3.0 - nu) / (1.0 + nu)
    A = 1.0 / (8.0 * math.pi * mu)

    # helper: derivative tensor dU[i,k,j]
    dU = np.zeros((2, 2, 2), dtype=float)
    rvec = np.array([rx, ry], dtype=float)
    # term1: (kappa-1) (r_j/r^2) δ_ik
    for i in range(2):
        for k in range(2):
            for j in range(2):
                dU[i, k, j] += A * (kappa - 1.0) * (rvec[j] * inv_r2) * (1.0 if i == k else 0.0)

    # term2: derivative of r_i r_k / r^2
    # ∂/∂x_j (r_i r_k / r^2) = (δ_{ij} r_k + r_i δ_{kj})/r^2 - 2 r_i r_k r_j / r^4
    for i in range(2):
        for k in range(2):
            for j in range(2):
                term = 0.0
                term += ( (1.0 if i == j else 0.0) * rvec[k] + rvec[i] * (1.0 if k == j else 0.0) ) * inv_r2
                term -= 2.0 * rvec[i] * rvec[k] * rvec[j] * inv_r4
                dU[i, k, j] += A * term

    # Build traction matrix T_{p k} = sigma_{p q}^{(k)} n_q,
    # where sigma^{(k)} is stress due to unit force in direction k.
    T = np.zeros((2, 2), dtype=float)

    # For each unit force direction k:
    for k in range(2):
        # strain eps_pq = 0.5(du_p/dx_q + du_q/dx_p) with u_p = U_{p k}
        eps = np.zeros((2, 2), dtype=float)
        for p in range(2):
            for q in range(2):
                eps[p, q] = 0.5 * (dU[p, k, q] + dU[q, k, p])

        tr_eps = float(eps[0, 0] + eps[1, 1])
        # stress
        sigma = 2.0 * mu * eps + lam * tr_eps * np.eye(2)

        # traction
        t = sigma @ np.array([nx, ny], dtype=float)
        T[:, k] = t

    return T

def _u_from_strain(x: float, y: float, eps: np.ndarray) -> np.ndarray:
    """
    Linear displacement field with zero rigid translation:
      u_x = eps_xx x + eps_xy y
      u_y = eps_xy x + eps_yy y
    eps = [eps_xx, eps_yy, eps_xy]
    """
    exx, eyy, exy = float(eps[0]), float(eps[1]), float(eps[2])
    return np.array([exx * x + exy * y,
                     exy * x + eyy * y], dtype=float)


def _t_from_strain(nx: float, ny: float, eps: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """
    Traction t = sigma n from constant strain eps.
    """
    exx, eyy, exy = float(eps[0]), float(eps[1]), float(eps[2])
    tr = exx + eyy
    sxx = lam * tr + 2.0 * mu * exx
    syy = lam * tr + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy

    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return np.array([tx, ty], dtype=float)


class BEMSolver2D:
    def __init__(self, E: float, nu: float, plane_strain: bool = True, enable_strain_mode: bool = True):
        self.E = float(E)
        self.nu = float(nu)
        self.plane_strain = bool(plane_strain)
        self.enable_strain_mode = bool(enable_strain_mode)

        self.segments: List[Segment] = []
        self._is_tr: List[bool] = []
        self._bcx: List[float] = []
        self._bcy: List[float] = []

        self.u_x: Optional[np.ndarray] = None
        self.u_y: Optional[np.ndarray] = None
        self.t_x: Optional[np.ndarray] = None
        self.t_y: Optional[np.ndarray] = None

        # solved strain mode (eps_xx, eps_yy, eps_xy)
        self.eps: Optional[np.ndarray] = None

        # optional scalar multiplier for H operator (keep 1.0 unless you intentionally calibrate)
        self.kH: float = 1.0

    def add_element(self, x1, y1, x2, y2, is_traction: bool, bc_x: float, bc_y: float):
        x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)
        L = float(np.hypot(x2 - x1, y2 - y1))
        tx = (x2 - x1) / L
        ty = (y2 - y1) / L
        nx = ty
        ny = -tx

        self.segments.append(Segment(x1, y1, x2, y2, xm, ym, nx, ny, L))
        self._is_tr.append(bool(is_traction))
        self._bcx.append(float(bc_x))
        self._bcy.append(float(bc_y))

    def _assemble_G_and_Hoff(self, gauss_n: int = 8) -> Tuple[np.ndarray, np.ndarray, float]:
        N = len(self.segments)
        xg, wg = _gauss_legendre(gauss_n)
        Lc = float(np.mean([s.L for s in self.segments]))
        r0 = Lc

        G = np.zeros((2*N, 2*N), dtype=float)
        H_off = np.zeros((2*N, 2*N), dtype=float)

        for i, si in enumerate(self.segments):
            xi, yi = si.xm, si.ym

            for j, sj in enumerate(self.segments):
                Gij = np.zeros((2, 2), dtype=float)
                Hij = np.zeros((2, 2), dtype=float)

                for s, w in zip(xg, wg):
                    xs = 0.5*(1.0 - s)*sj.x1 + 0.5*(1.0 + s)*sj.x2
                    ys = 0.5*(1.0 - s)*sj.y1 + 0.5*(1.0 + s)*sj.y2
                    jac = 0.5 * sj.L

                    U = kelvin_U((xi, yi), (xs, ys), E=self.E, nu=self.nu,
                                 plane_strain=self.plane_strain, r0=r0)
                    Gij += U * (w * jac)

                    if i != j:
                        T = kelvin_T(field=(xi, yi), source=(xs, ys), n_source=(sj.nx, sj.ny), E=self.E, nu=self.nu, plane_strain=self.plane_strain, r0=Lc)
                        Hij += T * (w * jac)

                G[2*i:2*i+2, 2*j:2*j+2] += Gij
                if i != j:
                    H_off[2*i:2*i+2, 2*j:2*j+2] += Hij

        return G, H_off, r0

    @staticmethod
    def _K_from_Hoff(H_off: np.ndarray) -> np.ndarray:
        N2 = H_off.shape[0]
        if N2 % 2 != 0:
            raise ValueError("H_off must be 2N×2N.")
        N = N2 // 2

        K = H_off.copy()
        for i in range(N):
            i0 = 2*i
            rowsum = np.zeros((2, 2), dtype=float)
            for j in range(N):
                j0 = 2*j
                rowsum += H_off[i0:i0+2, j0:j0+2]
            K[i0:i0+2, i0:i0+2] -= rowsum
            K[i0:i0+2, i0:i0+2] += 0.5 * np.eye(2)
        return K

    def solve(self, gauss_n: int = 8):
        N = len(self.segments)
        if N == 0:
            raise ValueError("No boundary elements added.")

        lam, mu, _ = _material(self.E, self.nu, self.plane_strain)

        G, H_off, _ = self._assemble_G_and_Hoff(gauss_n=gauss_n)
        K = self._K_from_Hoff(H_off)
        A = K

        # Known vectors
        u_known = np.zeros(2*N, dtype=float)
        t_known = np.zeros(2*N, dtype=float)

        is_tr_elem = np.array(self._is_tr, dtype=bool)
        for i in range(N):
            if is_tr_elem[i]:  # traction prescribed
                t_known[2*i]   = self._bcx[i]
                t_known[2*i+1] = self._bcy[i]
            else:              # displacement prescribed
                u_known[2*i]   = self._bcx[i]
                u_known[2*i+1] = self._bcy[i]

        is_tr = np.repeat(is_tr_elem, 2)
        idx_u = np.where(is_tr)[0]      # u unknown where traction known
        idx_t = np.where(~is_tr)[0]     # t unknown where u known

        rhs = (G @ t_known) - (A @ u_known)

        # Base unknowns: [u_unknowns, t_unknowns]
        M = np.hstack([A[:, idx_u], -G[:, idx_t]])

        # Optional constant-strain mode columns
        if self.enable_strain_mode:
            # The equation is: A(u - u_eps) = G(t - t_eps)
            # => A u - G t + (-A u_eps + G t_eps) = 0
            # Move known u_known/t_known to rhs, keep eps as unknown -> add columns:
            #   col_eps = (-A * u_eps_basis + G * t_eps_basis)
            cols = []
            for k in range(3):
                e = np.zeros(3, dtype=float)
                e[k] = 1.0

                u_eps = np.zeros(2*N, dtype=float)
                t_eps = np.zeros(2*N, dtype=float)
                for i, seg in enumerate(self.segments):
                    ui = _u_from_strain(seg.xm, seg.ym, e)
                    ti = _t_from_strain(seg.nx, seg.ny, e, lam, mu)
                    u_eps[2*i:2*i+2] = ui
                    t_eps[2*i:2*i+2] = ti

                col = (A @ u_eps) - (G @ t_eps)
                cols.append(col.reshape(-1, 1))

            EpsCols = np.hstack(cols)  # (2N,3)
            M = np.hstack([M, EpsCols])

        # Pure Neumann stabilization: constrain rigid modes on u (only affects displacement gauge)
        pure_neumann = bool(np.all(is_tr_elem))
        if pure_neumann:
            wC = 1e6
            Cfull = np.zeros((3, 2*N), dtype=float)
            Cfull[0, 0::2] = 1.0 / N  # mean ux
            Cfull[1, 1::2] = 1.0 / N  # mean uy
            xm = np.array([s.xm for s in self.segments], dtype=float)
            ym = np.array([s.ym for s in self.segments], dtype=float)
            Cfull[2, 0::2] = (-ym) / N
            Cfull[2, 1::2] = ( xm) / N

            Cu = Cfull[:, idx_u]
            n_u = len(idx_u)
            n_t = len(idx_t)
            n_eps = 3 if self.enable_strain_mode else 0

            aug = np.zeros((3, n_u + n_t + n_eps), dtype=float)
            aug[:, :n_u] = Cu
            M = np.vstack([M, wC * aug])
            rhs = np.concatenate([rhs, np.zeros(3)])

        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)

        # Unpack
        n_u = len(idx_u)
        n_t = len(idx_t)
        u = u_known.copy()
        t = t_known.copy()

        u[idx_u] = sol[:n_u]
        t[idx_t] = sol[n_u:n_u+n_t]

        self.u_x = u[0::2].copy()
        self.u_y = u[1::2].copy()
        self.t_x = t[0::2].copy()
        self.t_y = t[1::2].copy()

        if self.enable_strain_mode:
            self.eps = sol[n_u+n_t:n_u+n_t+3].copy()
        else:
            self.eps = np.zeros(3, dtype=float)

    def compute_displacement_at_point(self, x: float, y: float, gauss_n: int = 8) -> Tuple[float, float]:
        if self.u_x is None:
            raise RuntimeError("Call solve() first.")

        xg, wg = _gauss_legendre(gauss_n)
        Lc = float(np.mean([s.L for s in self.segments]))
        r0 = Lc

        u = np.zeros(2, dtype=float)

        # Add strain-mode particular solution if enabled
        if self.enable_strain_mode and self.eps is not None:
            u += _u_from_strain(x, y, self.eps)

        for j, seg in enumerate(self.segments):
            uj = np.array([self.u_x[j], self.u_y[j]], dtype=float)
            tj = np.array([self.t_x[j], self.t_y[j]], dtype=float)

            for s, w in zip(xg, wg):
                xs = 0.5*(1.0 - s)*seg.x1 + 0.5*(1.0 + s)*seg.x2
                ys = 0.5*(1.0 - s)*seg.y1 + 0.5*(1.0 + s)*seg.y2
                jac = 0.5 * seg.L

                U = kelvin_U((x, y), (xs, ys), E=self.E, nu=self.nu,
                             plane_strain=self.plane_strain, r0=r0)
                T = kelvin_T(field=(x, y), source=(xs, ys), n_source=(seg.nx, seg.ny),
                            E=self.E, nu=self.nu, plane_strain=self.plane_strain, r0=r0)

                # Somigliana identity for interior points:
                # u(x) = ∫_Γ U(x,ξ) t(ξ) dΓ(ξ)  -  ∫_Γ T(x,ξ) u(ξ) dΓ(ξ)
                u += (U @ tj - T @ uj) * (w * jac)

        return float(u[0]), float(u[1])

    def compute_stress_at_point(self, x: float, y: float, fd: Optional[float] = None, gauss_n: int = 8):
        if fd is None:
            Lc = float(np.mean([s.L for s in self.segments]))
            fd = 1e-6 * max(Lc, 1e-12)

        def disp(xx, yy):
            return np.array(self.compute_displacement_at_point(xx, yy, gauss_n=gauss_n), dtype=float)

        uxp = disp(x + fd, y)
        uxm = disp(x - fd, y)
        uyp = disp(x, y + fd)
        uym = disp(x, y - fd)

        du_dx = (uxp - uxm) / (2.0 * fd)
        du_dy = (uyp - uym) / (2.0 * fd)

        lam, mu, _ = _material(self.E, self.nu, self.plane_strain)

        eps_xx = du_dx[0]
        eps_yy = du_dy[1]
        eps_xy = 0.5 * (du_dy[0] + du_dx[1])

        tr = eps_xx + eps_yy
        sxx = lam * tr + 2.0 * mu * eps_xx
        syy = lam * tr + 2.0 * mu * eps_yy
        sxy = 2.0 * mu * eps_xy

        return float(sxx), float(syy), float(sxy)
