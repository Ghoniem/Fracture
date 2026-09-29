#include "vcem/edge_dislocation.h"

#include <cmath>

namespace vcem { namespace crack {

namespace {
constexpr double PI  = 3.141592653589793238462643383279502884;
constexpr double EPS = 1.0e-30;
}

// ─── Displacement field from (dBx, dBy) edge dislocation ─────────────────
void edge_dislocation_u(double dx, double dy,
                        double dBx, double dBy,
                        double nu, bool plane_stress,
                        double& ux, double& uy)
{
    nu = nu_eff(nu, plane_stress);

    double r2  = dx * dx + dy * dy + EPS;
    double inv = 1.0 / r2;

    double atan = std::atan2(dy, dx);
    double ln   = std::log(r2);

    double c1 = 1.0 / (2.0 * PI);
    double c2 = 1.0 / (4.0 * PI * (1.0 - nu));

    double diff_xx_yy = dx * dx - dy * dy;
    double xy_inv     = dx * dy * inv;
    double half_diff_inv = 0.5 * diff_xx_yy * inv;
    double half_ln_term  = (1.0 - 2.0 * nu) * 0.5 * ln;

    double ux_bx =  c1 * atan + c2 * xy_inv;
    double uy_bx = -c2 * (half_ln_term + half_diff_inv);

    double ux_by = -c2 * (half_ln_term - half_diff_inv);
    double uy_by =  c1 * atan - c2 * xy_inv;

    ux = dBx * ux_bx + dBy * ux_by;
    uy = dBx * uy_bx + dBy * uy_by;
}

// ─── Stress field from (dBx, dBy) edge dislocation ───────────────────────
// Python uses bx_stress for the x-component and a 90-deg rotation trick for
// the y-component:
//   bx contributes (sxx, syy, sxy) directly.
//   by is rewritten as bx in (x1, y1) = (dy, -dx), giving (sxx1, syy1, sxy1),
//   then the original-frame stresses are (syy1, sxx1, -sxy1).
namespace {

inline void bx_stress(double x, double y, double b, double coef,
                      double& sxx, double& syy, double& sxy)
{
    double r2 = x * x + y * y + EPS;
    double r4 = r2 * r2;
    double y_term_xx = y * (3.0 * x * x + y * y);
    double y_term_yy = y * (x * x - y * y);
    double x_term_xy = x * (x * x - y * y);
    sxx = -coef * b * y_term_xx / r4;
    syy =  coef * b * y_term_yy / r4;
    sxy =  coef * b * x_term_xy / r4;
}

}  // namespace

void edge_dislocation_stress(double dx, double dy,
                             double dBx, double dBy,
                             double mu, double nu, bool plane_stress,
                             double& sxx, double& syy, double& sxy)
{
    nu = nu_eff(nu, plane_stress);
    double coef = mu / (2.0 * PI * (1.0 - nu));

    sxx = 0.0; syy = 0.0; sxy = 0.0;

    if (dBx != 0.0) {
        double a, b, c;
        bx_stress(dx, dy, dBx, coef, a, b, c);
        sxx += a; syy += b; sxy += c;
    }

    if (dBy != 0.0) {
        // 90-degree rotation: (x1, y1) = (dy, -dx)
        double x1 = dy;
        double y1 = -dx;
        double sxx1, syy1, sxy1;
        bx_stress(x1, y1, dBy, coef, sxx1, syy1, sxy1);
        // Stresses in original frame from rotated-frame bx-stress:
        sxx +=  syy1;
        syy +=  sxx1;
        sxy += -sxy1;
    }
}

}}  // namespace vcem::crack
