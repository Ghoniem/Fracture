"""
bem_solver_v1_from_pdf.py

2D elastostatic BEM (constant elements, collocation) built from the
self-consistent equations in your PDF (Eq. 4.16–4.21, 4.18).

BIE (no body forces):
    c u(P) + ∫Γ p*(P,Q) u(Q) dΓ(Q) = ∫Γ u*(P,Q) t(Q) dΓ(Q)
where c = 1/2 I for smooth boundary collocation.

Kernels implemented exactly in the PDF style:
  u*_{ik}  (Kelvin displacement)  Eq (4.18) first line
  p*_{ik}  (Kelvin traction)      Eq (4.18) second line

Index convention used here (standard):
  U[i,k] = displacement component i due to unit point-force in direction k
  T[i,k] = traction component i (at boundary point, along outward normal) due to unit point-force k

Stress is computed (for now) from numerical displacement gradients:
  ε = sym(∇u), σ = λ tr(ε) I + 2μ ε

Plane strain vs plane stress:
  - κ changes in the displacement kernel u* (via kappa_from_nu)
  - constitutive in-plane σxx,σyy,σxy uses standard (λ,μ) choices for each case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from typing import List, Tuple, Optional


# -----------------------------
# Material helpers
# -----------------------------
def kappa_from_nu(nu: float, plane_strain: bool) -> float:
    # plane strain: κ = 3 - 4ν
    # plane stress: κ = (3 - ν)/(1 + ν)
    return (3.0 - 4.0 * nu) if plane_strain else ((3.0 - nu) / (1.0 + nu))


def shear_modulus(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))


def lame_lambda(E: float, nu: float, plane_strain: bool) -> float:
    if plane_strain:
        # 3D λ used in plane strain constitutive for in-plane stresses
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # plane stress “effective” λ for in-plane σ = λ tr(ε) I + 2μ ε
    return E * nu / (1.0 - nu * nu)


# -----------------------------
# Geometry
# -----------------------------
@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    is_traction: bool   # True => traction prescribed (u unknown); False => displacement prescribed (t unknown)
    bc_x: float
    bc_y: float

    @property
    def dx(self) -> float:
        return self.x2 - self.x1

    @property
    def dy(self) -> float:
        return self.y2 - self.y1

    @property
    def length(self) -> float:
        return float(math.hypot(self.dx, self.dy))

    @property
    def xm(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def ym(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def tx(self) -> float:
        L = self.length
        return self.dx / L

    @property
    def ty(self) -> float:
        L = self.length
        return self.dy / L

    @property
    def nx(self) -> float:
        # For a CCW discretized closed boundary, this is outward.
        return self.ty

    @property
    def ny(self) -> float:
        return -self.tx


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x.astype(float), w.astype(float)


def map_to_segment(seg: Segment, s: float) -> Tuple[float, float, float]:
    """
    s ∈ [-1,1] -> (x(s), y(s), jac), where jac = |d(x,y)/ds| = L/2
    """
    x = 0.5 * (1.0 - s) * seg.x1 + 0.5 * (1.0 + s) * seg.x2
    y = 0.5 * (1.0 - s) * seg.y1 + 0.5 * (1.0 + s) * seg.y2
    jac = 0.5 * seg.length
    return x, y, jac


# -----------------------------
# Kelvin kernels from PDF Eq (4.18)
# -----------------------------
def kelvin_U(field: Tuple[float, float],
             source: Tuple[float, float],
             E: float, nu: float, plane_strain: bool,
             r_floor: float = 1e-16) -> np.ndarray:
    """
    u*_{ik} displacement component i due to unit point-force in direction k at source.

    PDF-style (Eq 4.18 first line):
      u*_{ik} = 1/(8πG(1-ν)) [ κ ln(1/r) δ_{ik} + (∂r/∂x_i)(∂r/∂x_k) ]

    where κ = 3-4ν (plane strain) or (3-ν)/(1+ν) (plane stress),
    and (∂r/∂x_i) = r_i / r, so the last term is r_i r_k / r^2.
    """
    x, y = field
    xs, ys = source
    rx = x - xs
    ry = y - ys
    r2 = rx * rx + ry * ry
    if r2 < r_floor * r_floor:
        r2 = r_floor * r_floor
    r = math.sqrt(r2)

    G = shear_modulus(E, nu)
    coeff = 1.0 / (8.0 * math.pi * G * (1.0 - nu))
    kappa = kappa_from_nu(nu, plane_strain)

    ln1r = -math.log(r)  # ln(1/r)

    inv_r2 = 1.0 / r2
    rr = np.array([[rx * rx, rx * ry],
                   [ry * rx, ry * ry]], dtype=float) * inv_r2

    return coeff * (kappa * ln1r * np.eye(2) + rr)


def kelvin_T(field: Tuple[float, float],
             source: Tuple[float, float],
             n_source: Tuple[float, float],
             E: float, nu: float, plane_strain: bool,
             r_floor: float = 1e-16) -> np.ndarray:
    """
    p*_{ik} traction component i at the boundary point "source" (with outward normal n_source),
    due to unit point-force in direction k applied at "field".

    PDF-style (Eq 4.18 second line):
      p*_{ik} = - 1/(4π(1-ν) r) * [
                ∂r/∂n { (1-2ν) δ_{ik} + 2 (∂r/∂x_k)(∂r/∂x_i) }
                - (1-2ν)( (∂r/∂x_i) n_k - (∂r/∂x_k) n_i )
              ]

    IMPORTANT: derivatives are w.r.t the traction point coordinates (the boundary integration point),
    so we use r = x(source) - x(field) for ∂r/∂x.
    """
    xi, yi = field
    x, y = source
    nx, ny = n_source

    rx = x - xi
    ry = y - yi
    r2 = rx * rx + ry * ry
    if r2 < r_floor * r_floor:
        r2 = r_floor * r_floor
    r = math.sqrt(r2)
    inv_r = 1.0 / r

    drdx = rx * inv_r
    drdy = ry * inv_r
    drdn = drdx * nx + drdy * ny  # ∂r/∂n

    one_m_2nu = 1.0 - 2.0 * nu

    grad = np.array([drdx, drdy], dtype=float)
    outer = np.outer(grad, grad)  # (i,k)

    A = one_m_2nu * np.eye(2) + 2.0 * outer

    nvec = np.array([nx, ny], dtype=float)
    B = np.outer(grad, nvec) - np.outer(nvec, grad)  # (i,k)

    pref = -1.0 / (4.0 * math.pi * (1.0 - nu) * r)
    return pref * (drdn * A - one_m_2nu * B)


# -----------------------------
# Solver
# -----------------------------
class BEMSolver2D:
    def __init__(self, E: float, nu: float, plane_strain: bool = True):
        self.E = float(E)
        self.nu = float(nu)
        self.plane_strain = bool(plane_strain)

        self.segs: List[Segment] = []

        self.u_x: Optional[np.ndarray] = None
        self.u_y: Optional[np.ndarray] = None
        self.t_x: Optional[np.ndarray] = None
        self.t_y: Optional[np.ndarray] = None
        self._colloc_xy: Optional[np.ndarray] = None

    def add_element(self, x1: float, y1: float, x2: float, y2: float,
                    is_traction: bool, bc_x: float, bc_y: float):
        self.segs.append(Segment(float(x1), float(y1), float(x2), float(y2),
                                 bool(is_traction), float(bc_x), float(bc_y)))

    def solve(self, gauss_n: int = 8):
        if not self.segs:
            raise ValueError("No boundary elements added.")

        N = len(self.segs)
        xg, wg = gauss_legendre(int(gauss_n))

        colloc = np.array([[s.xm, s.ym] for s in self.segs], dtype=float)
        self._colloc_xy = colloc

        # Assemble G and H_off (diagonal of H handled by row-sum closure)
        G = np.zeros((2 * N, 2 * N), dtype=float)
        H_off = np.zeros((2 * N, 2 * N), dtype=float)

        for i, si in enumerate(self.segs):
            xi, yi = si.xm, si.ym

            for j, sj in enumerate(self.segs):
                Gij = np.zeros((2, 2), dtype=float)
                Hij = np.zeros((2, 2), dtype=float)

                rf = 1e-16 * max(sj.length, 1e-12)

                for s, w in zip(xg, wg):
                    xq, yq, jac = map_to_segment(sj, float(s))

                    U = kelvin_U((xi, yi), (xq, yq), E=self.E, nu=self.nu,
                                 plane_strain=self.plane_strain, r_floor=rf)
                    Gij += U * (w * jac)

                    if i != j:
                        T = kelvin_T(field=(xi, yi), source=(xq, yq), n_source=(sj.nx, sj.ny),
                                     E=self.E, nu=self.nu, plane_strain=self.plane_strain, r_floor=rf)
                        Hij += T * (w * jac)

                G[2 * i:2 * i + 2, 2 * j:2 * j + 2] = Gij
                if i != j:
                    H_off[2 * i:2 * i + 2, 2 * j:2 * j + 2] = Hij

        # H diagonal by rigid-translation (row-sum) closure:
        # (cI + H) * const_u = 0  for smooth boundary.
        c = 0.5
        H = H_off.copy()
        for i in range(N):
            row_sum = np.zeros((2, 2), dtype=float)
            for j in range(N):
                if i == j:
                    continue
                row_sum += H_off[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            Hii = -c * np.eye(2) - row_sum
            H[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Hii

        CplusH = H.copy()
        for i in range(N):
            CplusH[2 * i:2 * i + 2, 2 * i:2 * i + 2] += c * np.eye(2)

        # Mixed BC system build
        # Unknown vector x has size 2N:
        #   if element has traction prescribed => unknown is u on that element
        #   if element has displacement prescribed => unknown is t on that element
        col_u, col_t = {}, {}
        col = 0
        for j, sj in enumerate(self.segs):
            if sj.is_traction:
                col_u[j] = col
            else:
                col_t[j] = col
            col += 2

        A = np.zeros((2 * N, 2 * N), dtype=float)
        rhs = np.zeros((2 * N,), dtype=float)

        for i in range(N):
            rows = slice(2 * i, 2 * i + 2)

            for j in range(N):
                sj = self.segs[j]
                block_u = CplusH[rows, 2 * j:2 * j + 2]
                block_t = G[rows, 2 * j:2 * j + 2]

                if sj.is_traction:
                    # t known, u unknown
                    A[rows, col_u[j]:col_u[j] + 2] += block_u
                    rhs[rows] += block_t @ np.array([sj.bc_x, sj.bc_y], dtype=float)
                else:
                    # u known, t unknown
                    rhs[rows] -= block_u @ np.array([sj.bc_x, sj.bc_y], dtype=float)
                    A[rows, col_t[j]:col_t[j] + 2] -= block_t

        pure_neumann = all(s.is_traction for s in self.segs)
        if pure_neumann:
            # stabilize rigid-body modes (mean ux, mean uy, mean rotation) = 0
            w = 1e6
            A_aug = np.zeros((2 * N + 3, 2 * N), dtype=float)
            rhs_aug = np.zeros((2 * N + 3,), dtype=float)
            A_aug[:2 * N, :] = A
            rhs_aug[:2 * N] = rhs

            # mean ux
            for j in range(N):
                A_aug[2 * N + 0, col_u[j]] = 1.0
            # mean uy
            for j in range(N):
                A_aug[2 * N + 1, col_u[j] + 1] = 1.0
            # mean rotation about origin: Σ (x uy - y ux) = 0
            for j in range(N):
                xj, yj = colloc[j]
                A_aug[2 * N + 2, col_u[j]] = -yj
                A_aug[2 * N + 2, col_u[j] + 1] = xj

            A_aug[2 * N:, :] *= w
            sol, *_ = np.linalg.lstsq(A_aug, rhs_aug, rcond=None)
        else:
            sol = np.linalg.solve(A, rhs)

        # Unpack boundary fields
        u_x = np.zeros(N, dtype=float)
        u_y = np.zeros(N, dtype=float)
        t_x = np.zeros(N, dtype=float)
        t_y = np.zeros(N, dtype=float)

        for j, sj in enumerate(self.segs):
            if sj.is_traction:
                u_x[j] = sol[col_u[j]]
                u_y[j] = sol[col_u[j] + 1]
                t_x[j] = sj.bc_x
                t_y[j] = sj.bc_y
            else:
                t_x[j] = sol[col_t[j]]
                t_y[j] = sol[col_t[j] + 1]
                u_x[j] = sj.bc_x
                u_y[j] = sj.bc_y

        self.u_x, self.u_y, self.t_x, self.t_y = u_x, u_y, t_x, t_y

    # -----------------------------
    # Interior evaluation (Somigliana)
    # -----------------------------
    def compute_displacement_at_point(self, x: float, y: float, gauss_n: int = 12) -> Tuple[float, float]:
        if self.u_x is None or self.t_x is None:
            raise RuntimeError("Call solve() first.")

        xg, wg = gauss_legendre(int(gauss_n))
        u = np.zeros(2, dtype=float)

        for j, sj in enumerate(self.segs):
            tj = np.array([self.t_x[j], self.t_y[j]], dtype=float)
            uj = np.array([self.u_x[j], self.u_y[j]], dtype=float)
            rf = 1e-16 * max(sj.length, 1e-12)

            for s, w in zip(xg, wg):
                xq, yq, jac = map_to_segment(sj, float(s))

                U = kelvin_U((x, y), (xq, yq), E=self.E, nu=self.nu,
                             plane_strain=self.plane_strain, r_floor=rf)
                T = kelvin_T(field=(x, y), source=(xq, yq), n_source=(sj.nx, sj.ny),
                             E=self.E, nu=self.nu, plane_strain=self.plane_strain, r_floor=rf)

                u += (U @ tj - T @ uj) * (w * jac)

        # interior point coefficient is I (PDF Eq 4.16 for internal point)
        return float(u[0]), float(u[1])

    # -----------------------------
    # Stress from FD displacement gradients
    # -----------------------------
    def compute_stress_at_point(self, x: float, y: float, fd: float = 1e-6) -> Tuple[float, float, float]:
        ux_p, uy_p = self.compute_displacement_at_point(x + fd, y)
        ux_m, uy_m = self.compute_displacement_at_point(x - fd, y)
        vx_p, vy_p = self.compute_displacement_at_point(x, y + fd)
        vx_m, vy_m = self.compute_displacement_at_point(x, y - fd)

        dux_dx = (ux_p - ux_m) / (2.0 * fd)
        duy_dx = (uy_p - uy_m) / (2.0 * fd)
        dux_dy = (vx_p - vx_m) / (2.0 * fd)
        duy_dy = (vy_p - vy_m) / (2.0 * fd)

        exx = dux_dx
        eyy = duy_dy
        exy = 0.5 * (dux_dy + duy_dx)

        mu = shear_modulus(self.E, self.nu)
        lam = lame_lambda(self.E, self.nu, plane_strain=self.plane_strain)

        tr = exx + eyy
        sxx = lam * tr + 2.0 * mu * exx
        syy = lam * tr + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy
        return float(sxx), float(syy), float(sxy)