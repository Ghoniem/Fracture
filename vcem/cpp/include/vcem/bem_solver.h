#pragma once
//
// BEMSolver2D — direct translation of fracture_utils/Ubem/bem_solver.py
// class of the same name.  Constant-element collocation 2D elastostatic BEM
// with mixed displacement/traction BCs.

#include <Eigen/Dense>
#include <vector>

#include "vcem/segment.h"

namespace vcem { namespace bem {

class BEMSolver2D {
public:
    BEMSolver2D(double E, double nu, double h = 1.0, bool plane_strain = true);

    void add_element(double x1, double y1, double x2, double y2,
                     bool is_traction, double bc_x, double bc_y);

    // Assemble + solve the boundary system.  Populates u_x, u_y, t_x, t_y.
    void solve(int gauss_n = 8);

    // Somigliana displacement at an interior point (call after solve()).
    void compute_displacement_at_point(double x, double y, int gauss_n,
                                       double& ux, double& uy) const;

    // ∂u_i/∂x_k at an interior point — analytic kernel derivatives.
    // dudx(i,k) = ∂u_i/∂x_k  (k=0 -> ∂/∂x, k=1 -> ∂/∂y)
    Eigen::Matrix2d compute_grad_u_at_point(double x, double y, int gauss_n) const;

    // (σ_xx, σ_yy, σ_xy) at an interior point.
    void compute_stress_at_point(double x, double y, int gauss_n,
                                 double& sxx, double& syy, double& sxy) const;

    // Vectorized helpers — evaluate stress on a tensor-product grid.
    // Output arrays are shape (ny, nx) row-major and represent S(ys[i], xs[j]).
    // Returns the number of grid points evaluated.
    int stress_on_grid(const std::vector<double>& xs,
                       const std::vector<double>& ys,
                       int gauss_n,
                       Eigen::MatrixXd& Sxx,
                       Eigen::MatrixXd& Syy,
                       Eigen::MatrixXd& Sxy) const;

    // Evaluate stress at an arbitrary cloud of points (xs[i], ys[i]).
    // For disk-conforming polar or any non-tensor grid. xs and ys must
    // have the same length. Returns the number of points evaluated.
    int stress_at_points(const std::vector<double>& xs,
                         const std::vector<double>& ys,
                         int gauss_n,
                         Eigen::VectorXd& Sxx,
                         Eigen::VectorXd& Syy,
                         Eigen::VectorXd& Sxy) const;

    // Read-only accessors for the boundary solution.
    const std::vector<double>& u_x() const { return u_x_; }
    const std::vector<double>& u_y() const { return u_y_; }
    const std::vector<double>& t_x() const { return t_x_; }
    const std::vector<double>& t_y() const { return t_y_; }
    const std::vector<Segment>& segments() const { return segs_; }

    double E() const { return E_; }
    double nu() const { return nu_; }
    double h() const { return h_; }
    bool plane_strain() const { return plane_strain_; }

private:
    double E_;
    double nu_;
    double h_;
    bool   plane_strain_;

    std::vector<Segment> segs_;

    std::vector<double> u_x_;
    std::vector<double> u_y_;
    std::vector<double> t_x_;
    std::vector<double> t_y_;
    bool solved_ = false;
};

}}  // namespace vcem::bem
