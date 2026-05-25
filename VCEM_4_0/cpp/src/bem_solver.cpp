#include "vcem/bem_solver.h"
#include "vcem/kelvin.h"
#include "vcem/material.h"
#include "vcem/quadrature.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(VCEM_HAVE_OPENMP)
#include <omp.h>
#endif

namespace vcem { namespace bem {

namespace {

// Internal: compute grad u at (x, y) given a pre-fetched Gauss rule. Mutex-free.
Eigen::Matrix2d grad_u_with_rule(const std::vector<Segment>& segs,
                                 const std::vector<double>& u_x,
                                 const std::vector<double>& u_y,
                                 const std::vector<double>& t_x,
                                 const std::vector<double>& t_y,
                                 double E, double nu, bool plane_strain,
                                 double x, double y,
                                 const GaussRule& gr)
{
    Eigen::Matrix2d dudx_mat = Eigen::Matrix2d::Zero();
    const int N = static_cast<int>(segs.size());
    const int Q = static_cast<int>(gr.nodes.size());

    Eigen::Matrix2d dUdx, dUdy, dTdx, dTdy;

    for (int j = 0; j < N; ++j) {
        const Segment& sj = segs[j];
        Eigen::Vector2d tj(t_x[j], t_y[j]);
        Eigen::Vector2d uj(u_x[j], u_y[j]);

        const double rf  = 1.0e-16 * std::max(sj.length(), 1.0e-12);
        const double njx = sj.nx();
        const double njy = sj.ny();

        for (int q = 0; q < Q; ++q) {
            double s = gr.nodes[q];
            double w = gr.weights[q];
            double xq, yq, jac;
            map_to_segment(sj, s, xq, yq, jac);

            kelvin_dU_dfield(x, y, xq, yq, E, nu, plane_strain, dUdx, dUdy, rf);
            kelvin_dT_dfield(x, y, xq, yq, njx, njy, E, nu, plane_strain, dTdx, dTdy, rf);

            double ww = w * jac;
            dudx_mat.col(0).noalias() += (dUdx * tj - dTdx * uj) * ww;
            dudx_mat.col(1).noalias() += (dUdy * tj - dTdy * uj) * ww;
        }
    }
    return dudx_mat;
}

// Internal: compute stress (sxx, syy, sxy) at (x, y) given a pre-fetched rule.
void stress_with_rule(const std::vector<Segment>& segs,
                      const std::vector<double>& u_x,
                      const std::vector<double>& u_y,
                      const std::vector<double>& t_x,
                      const std::vector<double>& t_y,
                      double E, double nu, bool plane_strain,
                      double x, double y,
                      const GaussRule& gr,
                      double& sxx, double& syy, double& sxy)
{
    Eigen::Matrix2d g = grad_u_with_rule(segs, u_x, u_y, t_x, t_y,
                                         E, nu, plane_strain, x, y, gr);
    double dux_dx = g(0, 0);
    double dux_dy = g(0, 1);
    double duy_dx = g(1, 0);
    double duy_dy = g(1, 1);

    double exx = dux_dx;
    double eyy = duy_dy;
    double exy = 0.5 * (dux_dy + duy_dx);

    if (plane_strain) {
        MaterialParams mp = material_params(E, nu, true);
        sxx = 2.0 * mp.mu * exx + mp.lam * (exx + eyy);
        syy = 2.0 * mp.mu * eyy + mp.lam * (exx + eyy);
        sxy = 2.0 * mp.mu * exy;
    } else {
        double mu  = shear_modulus(E, nu);
        double fac = E / (1.0 - nu * nu);
        sxx = fac * (exx + nu * eyy);
        syy = fac * (eyy + nu * exx);
        sxy = 2.0 * mu * exy;
    }
}

}  // namespace

BEMSolver2D::BEMSolver2D(double E, double nu, double h, bool plane_strain)
    : E_(E), nu_(nu), h_(h), plane_strain_(plane_strain) {}

void BEMSolver2D::add_element(double x1, double y1, double x2, double y2,
                              bool is_traction, double bc_x, double bc_y)
{
    Segment s;
    s.x1 = x1; s.y1 = y1; s.x2 = x2; s.y2 = y2;
    s.is_traction = is_traction;
    s.bc_x = bc_x; s.bc_y = bc_y;
    segs_.push_back(s);
}

void BEMSolver2D::solve(int gauss_n)
{
    if (segs_.empty()) {
        throw std::runtime_error("BEMSolver2D::solve: no boundary elements added");
    }

    const int N  = static_cast<int>(segs_.size());
    const int N2 = 2 * N;

    // Pre-fetch the rule outside any parallel region so the cache mutex
    // never executes inside parallel work.
    const GaussRule& gr = gauss_legendre(gauss_n);
    const int Q = static_cast<int>(gr.nodes.size());

    Eigen::MatrixXd G     = Eigen::MatrixXd::Zero(N2, N2);
    Eigen::MatrixXd H_off = Eigen::MatrixXd::Zero(N2, N2);

    // ── Assemble G and H_off ───────────────────────────────────────────────
    // Each i writes to disjoint rows of G/H_off => race-free across i.
#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int i = 0; i < N; ++i) {
        const Segment& si = segs_[i];
        const double xi = si.xm();
        const double yi = si.ym();

        for (int j = 0; j < N; ++j) {
            const Segment& sj = segs_[j];

            if (i == j) {
                G.block<2,2>(2*i, 2*j) = kelvin_U_self_panel(
                    sj, E_, nu_, plane_strain_);
                continue;
            }

            Eigen::Matrix2d Gij = Eigen::Matrix2d::Zero();
            Eigen::Matrix2d Hij = Eigen::Matrix2d::Zero();

            const double rf  = 1.0e-16 * std::max(sj.length(), 1.0e-12);
            const double njx = sj.nx();
            const double njy = sj.ny();

            for (int q = 0; q < Q; ++q) {
                double s = gr.nodes[q];
                double w = gr.weights[q];
                double xq, yq, jac;
                map_to_segment(sj, s, xq, yq, jac);

                Eigen::Matrix2d U = kelvin_U(xi, yi, xq, yq,
                                             E_, nu_, plane_strain_, rf);
                Eigen::Matrix2d T = kelvin_T(xi, yi, xq, yq, njx, njy,
                                             E_, nu_, plane_strain_, rf);

                Gij.noalias() += U * (w * jac);
                Hij.noalias() += T * (w * jac);
            }

            G.block<2,2>(2*i, 2*j)     = Gij;
            H_off.block<2,2>(2*i, 2*j) = Hij;
        }
    }

    // ── H diagonal by rigid-translation row-sum closure ───────────────────
    constexpr double c = 0.5;
    Eigen::MatrixXd H = H_off;
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix2d row_sum = Eigen::Matrix2d::Zero();
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            row_sum.noalias() += H_off.block<2,2>(2*i, 2*j);
        }
        Eigen::Matrix2d Hii = -c * Eigen::Matrix2d::Identity() - row_sum;
        H.block<2,2>(2*i, 2*i) = Hii;
    }

    Eigen::MatrixXd CplusH = H;
    for (int i = 0; i < N; ++i) {
        CplusH.block<2,2>(2*i, 2*i) += c * Eigen::Matrix2d::Identity();
    }

    // ── Mixed BC system build ──────────────────────────────────────────────
    std::vector<int> col_unknown(N);
    for (int j = 0; j < N; ++j) col_unknown[j] = 2 * j;

    Eigen::MatrixXd A   = Eigen::MatrixXd::Zero(N2, N2);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N2);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const Segment& sj = segs_[j];
            Eigen::Matrix2d block_u = CplusH.block<2,2>(2*i, 2*j);
            Eigen::Matrix2d block_t = G.block<2,2>(2*i, 2*j);
            Eigen::Vector2d bc(sj.bc_x, sj.bc_y);

            if (sj.is_traction) {
                A.block<2,2>(2*i, col_unknown[j])      += block_u;
                rhs.segment<2>(2*i).noalias()          += block_t * bc;
            } else {
                rhs.segment<2>(2*i).noalias()          -= block_u * bc;
                A.block<2,2>(2*i, col_unknown[j])      -= block_t;
            }
        }
    }

    bool pure_neumann = std::all_of(segs_.begin(), segs_.end(),
                                    [](const Segment& s){ return s.is_traction; });

    Eigen::VectorXd sol;
    if (pure_neumann) {
        double diag_scale = 0.0;
        for (int i = 0; i < N2; ++i) diag_scale = std::max(diag_scale, std::abs(A(i, i)));
        if (diag_scale < 1.0) diag_scale = 1.0;
        const double w = 1.0e6 * diag_scale;

        Eigen::MatrixXd A_aug = Eigen::MatrixXd::Zero(N2 + 3, N2);
        Eigen::VectorXd rhs_aug = Eigen::VectorXd::Zero(N2 + 3);

        A_aug.topRows(N2) = A;
        rhs_aug.head(N2)  = rhs;

        for (int j = 0; j < N; ++j) A_aug(N2 + 0, col_unknown[j])     = 1.0;
        for (int j = 0; j < N; ++j) A_aug(N2 + 1, col_unknown[j] + 1) = 1.0;
        for (int j = 0; j < N; ++j) {
            double xj = segs_[j].xm();
            double yj = segs_[j].ym();
            A_aug(N2 + 2, col_unknown[j])     = -yj;
            A_aug(N2 + 2, col_unknown[j] + 1) =  xj;
        }
        A_aug.bottomRows(3) *= w;

        Eigen::BDCSVD<Eigen::MatrixXd> svd(A_aug, Eigen::ComputeThinU | Eigen::ComputeThinV);
        sol = svd.solve(rhs_aug);
    } else {
        sol = A.partialPivLu().solve(rhs);
    }

    u_x_.assign(N, 0.0);
    u_y_.assign(N, 0.0);
    t_x_.assign(N, 0.0);
    t_y_.assign(N, 0.0);

    for (int j = 0; j < N; ++j) {
        const Segment& sj = segs_[j];
        if (sj.is_traction) {
            u_x_[j] = sol(col_unknown[j]);
            u_y_[j] = sol(col_unknown[j] + 1);
            t_x_[j] = sj.bc_x;
            t_y_[j] = sj.bc_y;
        } else {
            t_x_[j] = sol(col_unknown[j]);
            t_y_[j] = sol(col_unknown[j] + 1);
            u_x_[j] = sj.bc_x;
            u_y_[j] = sj.bc_y;
        }
    }

    solved_ = true;
}

void BEMSolver2D::compute_displacement_at_point(double x, double y, int gauss_n,
                                                double& ux, double& uy) const
{
    if (!solved_) throw std::runtime_error("Call solve() first.");

    const GaussRule& gr = gauss_legendre(gauss_n);
    const int Q = static_cast<int>(gr.nodes.size());

    Eigen::Vector2d u = Eigen::Vector2d::Zero();

    const int N = static_cast<int>(segs_.size());
    for (int j = 0; j < N; ++j) {
        const Segment& sj = segs_[j];
        Eigen::Vector2d tj(t_x_[j], t_y_[j]);
        Eigen::Vector2d uj(u_x_[j], u_y_[j]);

        const double rf  = 1.0e-16 * std::max(sj.length(), 1.0e-12);
        const double njx = sj.nx();
        const double njy = sj.ny();

        for (int q = 0; q < Q; ++q) {
            double s = gr.nodes[q];
            double w = gr.weights[q];
            double xq, yq, jac;
            map_to_segment(sj, s, xq, yq, jac);

            Eigen::Matrix2d U = kelvin_U(x, y, xq, yq,
                                         E_, nu_, plane_strain_, rf);
            Eigen::Matrix2d T = kelvin_T(x, y, xq, yq, njx, njy,
                                         E_, nu_, plane_strain_, rf);

            u.noalias() += (U * tj - T * uj) * (w * jac);
        }
    }

    ux = u(0);
    uy = u(1);
}

Eigen::Matrix2d BEMSolver2D::compute_grad_u_at_point(double x, double y, int gauss_n) const
{
    if (!solved_) throw std::runtime_error("Call solve() first.");
    const GaussRule& gr = gauss_legendre(gauss_n);
    return grad_u_with_rule(segs_, u_x_, u_y_, t_x_, t_y_,
                            E_, nu_, plane_strain_, x, y, gr);
}

void BEMSolver2D::compute_stress_at_point(double x, double y, int gauss_n,
                                          double& sxx, double& syy, double& sxy) const
{
    if (!solved_) throw std::runtime_error("Call solve() first.");
    const GaussRule& gr = gauss_legendre(gauss_n);
    stress_with_rule(segs_, u_x_, u_y_, t_x_, t_y_,
                     E_, nu_, plane_strain_, x, y, gr, sxx, syy, sxy);
}

int BEMSolver2D::stress_on_grid(const std::vector<double>& xs,
                                const std::vector<double>& ys,
                                int gauss_n,
                                Eigen::MatrixXd& Sxx,
                                Eigen::MatrixXd& Syy,
                                Eigen::MatrixXd& Sxy) const
{
    if (!solved_) throw std::runtime_error("Call solve() first.");

    const int nx = static_cast<int>(xs.size());
    const int ny = static_cast<int>(ys.size());

    Sxx.resize(ny, nx);
    Syy.resize(ny, nx);
    Sxy.resize(ny, nx);

    // Pre-fetch rule once -- no mutex inside the parallel region.
    const GaussRule& gr = gauss_legendre(gauss_n);

    // Snapshot of mutable boundary state into const refs for the lambda.
    const auto& segs = segs_;
    const auto& ux   = u_x_;
    const auto& uy   = u_y_;
    const auto& tx   = t_x_;
    const auto& ty   = t_y_;
    const double E   = E_;
    const double nu  = nu_;
    const bool   ps  = plane_strain_;

    // MSVC ships OpenMP 2.0 which ignores `collapse`; outer-loop parallelism
    // is sufficient as long as ny >= num_threads (typical: ny=120, 24 cores).
#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            double sxx, syy, sxy;
            stress_with_rule(segs, ux, uy, tx, ty,
                             E, nu, ps, xs[j], ys[i], gr, sxx, syy, sxy);
            Sxx(i, j) = sxx;
            Syy(i, j) = syy;
            Sxy(i, j) = sxy;
        }
    }

    return nx * ny;
}

}}  // namespace vcem::bem
