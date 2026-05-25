#include "vcem/kelvin.h"
#include "vcem/material.h"

#include <cmath>

namespace vcem { namespace bem {

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;
}

// ─── u*_{ik} ─ Kelvin displacement (Eq 4.18 first line) ────────────────────
// U[i,k] = displacement comp i due to unit point-force in direction k at source.
//   u*_{ik} = 1/(8πG(1-ν)) [ κ ln(1/r) δ_{ik} + (r_i r_k / r^2) ]
Eigen::Matrix2d kelvin_U(double xf, double yf,
                         double xs, double ys,
                         double E, double nu, bool plane_strain,
                         double r_floor)
{
    double rx = xf - xs;
    double ry = yf - ys;
    double r2 = rx * rx + ry * ry;
    if (r2 < r_floor * r_floor) r2 = r_floor * r_floor;
    double r = std::sqrt(r2);

    double G     = shear_modulus(E, nu);
    double coeff = 1.0 / (8.0 * PI * G * (1.0 - nu));
    double kappa = kappa_from_nu(nu, plane_strain);

    double ln1r   = -std::log(r);
    double inv_r2 = 1.0 / r2;

    Eigen::Matrix2d rr;
    rr << rx * rx, rx * ry,
          ry * rx, ry * ry;
    rr *= inv_r2;

    Eigen::Matrix2d U = kappa * ln1r * Eigen::Matrix2d::Identity() + rr;
    return coeff * U;
}

// ─── Analytic ∫ U ds over a self-panel evaluated at its midpoint ─────────────
// ∫_{-1}^{1} log(1/r) ds = 2*log(2/L) + 2,  with r(s) = (L/2)|s|.
// ∫_{-1}^{1} (r_i r_k / r^2) ds = 2 t_i t_k     (r is along the panel tangent).
// jac = L/2  ->  integral = L/(8πG(1-ν)) * [ κ (log(2/L)+1) I + t t^T ].
Eigen::Matrix2d kelvin_U_self_panel(const Segment& seg,
                                    double E, double nu, bool plane_strain)
{
    double L     = seg.length();
    double G     = shear_modulus(E, nu);
    double coeff = 1.0 / (8.0 * PI * G * (1.0 - nu));
    double kappa = kappa_from_nu(nu, plane_strain);
    double tx    = seg.tx();
    double ty    = seg.ty();

    Eigen::Matrix2d tt;
    tt << tx * tx, tx * ty,
          tx * ty, ty * ty;

    Eigen::Matrix2d M = kappa * (std::log(2.0 / L) + 1.0) * Eigen::Matrix2d::Identity() + tt;
    return L * coeff * M;
}

// ─── p*_{ik} ─ Kelvin traction (Eq 4.18 second line) ────────────────────────
// p*_{ik} = -1/(4π(1-ν)r) * [ (∂r/∂n){ (1-2ν)δ_{ik} + 2 (∂r/∂x_k)(∂r/∂x_i) }
//                            - (1-2ν)( (∂r/∂x_i) n_k - (∂r/∂x_k) n_i ) ]
// Derivatives are w.r.t. the traction (source) point coords, so rvec = source - field.
Eigen::Matrix2d kelvin_T(double xf, double yf,
                         double xs, double ys,
                         double nx, double ny,
                         double E, double nu, bool plane_strain,
                         double r_floor)
{
    (void)E; (void)plane_strain;  // T does not depend on E or plane-strain flag.

    double rx = xs - xf;
    double ry = ys - yf;
    double r2 = rx * rx + ry * ry;
    if (r2 < r_floor * r_floor) r2 = r_floor * r_floor;
    double r     = std::sqrt(r2);
    double inv_r = 1.0 / r;

    double drdx = rx * inv_r;
    double drdy = ry * inv_r;
    double drdn = drdx * nx + drdy * ny;

    double one_m_2nu = 1.0 - 2.0 * nu;

    // outer(grad, grad)
    Eigen::Matrix2d outer;
    outer << drdx * drdx, drdx * drdy,
             drdy * drdx, drdy * drdy;

    Eigen::Matrix2d A = one_m_2nu * Eigen::Matrix2d::Identity() + 2.0 * outer;

    // B = grad ⊗ n - n ⊗ grad   (antisymmetric)
    Eigen::Matrix2d B;
    B << drdx * nx - nx * drdx, drdx * ny - nx * drdy,
         drdy * nx - ny * drdx, drdy * ny - ny * drdy;

    double pref = -1.0 / (4.0 * PI * (1.0 - nu) * r);
    return pref * (drdn * A - one_m_2nu * B);
}

// ─── Spatial derivatives of U with respect to FIELD coords ──────────────────
// U_{ij} = c [ κ ln(1/r) δ_{ij} + g_i g_j ],   g = (x - ξ)/r.
// (dUdx)_{ij} = ∂U_{ij}/∂x_field   (k = 0)
// (dUdy)_{ij} = ∂U_{ij}/∂y_field   (k = 1)
void kelvin_dU_dfield(double xf, double yf,
                      double xs, double ys,
                      double E, double nu, bool plane_strain,
                      Eigen::Matrix2d& dUdx, Eigen::Matrix2d& dUdy,
                      double r_floor)
{
    double rx = xf - xs;
    double ry = yf - ys;
    double r2 = rx * rx + ry * ry;
    if (r2 < r_floor * r_floor) r2 = r_floor * r_floor;
    double r     = std::sqrt(r2);
    double inv_r = 1.0 / r;

    double gx = rx * inv_r;
    double gy = ry * inv_r;

    double G     = shear_modulus(E, nu);
    double coeff = 1.0 / (8.0 * PI * G * (1.0 - nu));
    double kappa = kappa_from_nu(nu, plane_strain);

    // ∂ ln(1/r) / ∂x_k = -g_k / r
    double dln_dx = -gx * inv_r;
    double dln_dy = -gy * inv_r;

    // ∂g_i/∂x_k = (δ_{ik} - g_i g_k)/r
    double dgx_dx = (1.0 - gx * gx) * inv_r;
    double dgy_dx = (    - gy * gx) * inv_r;
    double dgx_dy = (    - gx * gy) * inv_r;
    double dgy_dy = (1.0 - gy * gy) * inv_r;

    // d(g_i g_j) for k = x and k = y
    Eigen::Matrix2d dgg_dx;
    dgg_dx << dgx_dx * gx + gx * dgx_dx,  dgx_dx * gy + gx * dgy_dx,
              dgy_dx * gx + gy * dgx_dx,  dgy_dx * gy + gy * dgy_dx;

    Eigen::Matrix2d dgg_dy;
    dgg_dy << dgx_dy * gx + gx * dgx_dy,  dgx_dy * gy + gx * dgy_dy,
              dgy_dy * gx + gy * dgx_dy,  dgy_dy * gy + gy * dgy_dy;

    dUdx = coeff * (kappa * dln_dx * Eigen::Matrix2d::Identity() + dgg_dx);
    dUdy = coeff * (kappa * dln_dy * Eigen::Matrix2d::Identity() + dgg_dy);
}

// ─── Spatial derivatives of T with respect to FIELD coords ──────────────────
// kelvin_T uses rvec = source - field, so ∂r_i/∂x_k_field = -δ_{ik}.
// This propagates through the chain rule on (pref, drdn, A, B).
void kelvin_dT_dfield(double xf, double yf,
                      double xs, double ys,
                      double nx, double ny,
                      double E, double nu, bool plane_strain,
                      Eigen::Matrix2d& dTdx, Eigen::Matrix2d& dTdy,
                      double r_floor)
{
    (void)E; (void)plane_strain;

    double rx = xs - xf;
    double ry = ys - yf;
    double r2 = rx * rx + ry * ry;
    if (r2 < r_floor * r_floor) r2 = r_floor * r_floor;
    double r      = std::sqrt(r2);
    double inv_r  = 1.0 / r;
    double inv_r2 = inv_r * inv_r;

    double gx = rx * inv_r;
    double gy = ry * inv_r;

    double one_m_2nu = 1.0 - 2.0 * nu;
    double drdn      = gx * nx + gy * ny;

    double C1   = 1.0 / (4.0 * PI * (1.0 - nu));
    double pref = -C1 * inv_r;

    // ∂(1/r)/∂x_k_field = g_k / r^2
    double dpref_dx = -C1 * (gx * inv_r2);
    double dpref_dy = -C1 * (gy * inv_r2);

    // ∂g_i/∂x_k_field = -(δ_{ik} - g_i g_k)/r
    double dgx_dx = -(1.0 - gx * gx) * inv_r;
    double dgy_dx =  (gy * gx) * inv_r;
    double dgx_dy =  (gx * gy) * inv_r;
    double dgy_dy = -(1.0 - gy * gy) * inv_r;

    // ∂drdn/∂x_k = n_i ∂g_i/∂x_k
    double ddrdn_dx = nx * dgx_dx + ny * dgy_dx;
    double ddrdn_dy = nx * dgx_dy + ny * dgy_dy;

    Eigen::Matrix2d gg;
    gg << gx * gx, gx * gy,
          gy * gx, gy * gy;

    Eigen::Matrix2d A = one_m_2nu * Eigen::Matrix2d::Identity() + 2.0 * gg;

    // dA/dx_k = 2 (∂g ⊗ g + g ⊗ ∂g)
    Eigen::Matrix2d dgg_dx;
    dgg_dx << dgx_dx * gx + gx * dgx_dx,  dgx_dx * gy + gx * dgy_dx,
              dgy_dx * gx + gy * dgx_dx,  dgy_dx * gy + gy * dgy_dx;
    Eigen::Matrix2d dgg_dy;
    dgg_dy << dgx_dy * gx + gx * dgx_dy,  dgx_dy * gy + gx * dgy_dy,
              dgy_dy * gx + gy * dgx_dy,  dgy_dy * gy + gy * dgy_dy;
    Eigen::Matrix2d dA_dx = 2.0 * dgg_dx;
    Eigen::Matrix2d dA_dy = 2.0 * dgg_dy;

    // B = g ⊗ n - n ⊗ g
    Eigen::Matrix2d B;
    B << gx * nx - nx * gx, gx * ny - nx * gy,
         gy * nx - ny * gx, gy * ny - ny * gy;

    // dB/dx_k = ∂g ⊗ n - n ⊗ ∂g
    Eigen::Matrix2d dB_dx;
    dB_dx << dgx_dx * nx - nx * dgx_dx, dgx_dx * ny - nx * dgy_dx,
             dgy_dx * nx - ny * dgx_dx, dgy_dx * ny - ny * dgy_dx;
    Eigen::Matrix2d dB_dy;
    dB_dy << dgx_dy * nx - nx * dgx_dy, dgx_dy * ny - nx * dgy_dy,
             dgy_dy * nx - ny * dgx_dy, dgy_dy * ny - ny * dgy_dy;

    Eigen::Matrix2d core = drdn * A - one_m_2nu * B;

    dTdx = dpref_dx * core + pref * (ddrdn_dx * A + drdn * dA_dx - one_m_2nu * dB_dx);
    dTdy = dpref_dy * core + pref * (ddrdn_dy * A + drdn * dA_dy - one_m_2nu * dB_dy);
}

}}  // namespace vcem::bem
