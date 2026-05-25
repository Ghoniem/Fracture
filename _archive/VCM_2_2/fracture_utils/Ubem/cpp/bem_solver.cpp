#include "bem_solver.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace BEM {

BEMSolver2D::BEMSolver2D(double young_modulus, double poisson_ratio, bool plane_strain)
    : E(young_modulus), nu(poisson_ratio) {
    
    G = E / (2.0 * (1.0 + nu));
    
    if (plane_strain) {
        kappa = 3.0 - 4.0 * nu;
    } else {
        kappa = (3.0 - nu) / (1.0 + nu);
    }
}

void BEMSolver2D::add_element(double x1, double y1, double x2, double y2,
                              bool is_traction, double bc_x, double bc_y) {
    BoundaryElement elem;
    elem.x1 = x1;
    elem.y1 = y1;
    elem.x2 = x2;
    elem.y2 = y2;
    elem.is_traction_bc = is_traction;
    elem.bc_x = bc_x;
    elem.bc_y = bc_y;
    elements.push_back(elem);
}

void BEMSolver2D::kelvin_u(double xi, double yi, double x, double y,
                           double nx, double ny, double (&U)[2][2]) const {
    double dx = x - xi;
    double dy = y - yi;
    double r = std::sqrt(dx*dx + dy*dy);
    
    if (r < 1e-10) {
        // Singular point - set to zero (will be handled specially)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                U[i][j] = 0.0;
        return;
    }
    
    double c1 = 1.0 / (8.0 * M_PI * G * (1.0 - nu));
    double log_r = std::log(r);
    
    // U_ij = c1 * [-(3-4*nu) * log(r) * delta_ij + r_i * r_j / r^2]
    U[0][0] = c1 * (-(kappa + 1.0) * log_r + dx*dx/(r*r));
    U[0][1] = c1 * (dx*dy/(r*r));
    U[1][0] = c1 * (dx*dy/(r*r));
    U[1][1] = c1 * (-(kappa + 1.0) * log_r + dy*dy/(r*r));
}

void BEMSolver2D::kelvin_t(double xi, double yi, double x, double y,
                           double nx, double ny, double (&T)[2][2]) const {
    double dx = x - xi;
    double dy = y - yi;
    double r = std::sqrt(dx*dx + dy*dy);
    
    if (r < 1e-10) {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                T[i][j] = 0.0;
        return;
    }
    
    double dr_dn = (dx * nx + dy * ny) / r;
    double c2 = -1.0 / (4.0 * M_PI * (1.0 - nu) * r);
    
    // T_ij = c2 * { dr/dn * [(1-2*nu)*delta_ij + 2*r_i*r_j/r^2] 
    //              - (1-2*nu)*(r_i*n_j - r_j*n_i) }
    
    double r_i[2] = {dx/r, dy/r};
    
    T[0][0] = c2 * (dr_dn * ((1.0 - 2.0*nu) + 2.0*r_i[0]*r_i[0])
                    - (1.0 - 2.0*nu) * (r_i[0]*nx - r_i[0]*nx));
    
    T[0][1] = c2 * (dr_dn * (2.0*r_i[0]*r_i[1])
                    - (1.0 - 2.0*nu) * (r_i[0]*ny - r_i[1]*nx));
    
    T[1][0] = c2 * (dr_dn * (2.0*r_i[1]*r_i[0])
                    - (1.0 - 2.0*nu) * (r_i[1]*nx - r_i[0]*ny));
    
    T[1][1] = c2 * (dr_dn * ((1.0 - 2.0*nu) + 2.0*r_i[1]*r_i[1])
                    - (1.0 - 2.0*nu) * (r_i[1]*ny - r_i[1]*ny));
}

void BEMSolver2D::integrate_element(const BoundaryElement& elem, 
                                    double xi, double yi,
                                    double (&H)[2][2], 
                                    double (&G_mat)[2][2]) const {
    // Initialize
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            H[i][j] = 0.0;
            G_mat[i][j] = 0.0;
        }
    }
    
    double nx, ny;
    elem.normal(nx, ny);
    double elem_length = elem.length();
    
    // Check if collocation point is on this element (singular integral)
    double xm, ym;
    elem.midpoint(xm, ym);
    double dist_to_elem = std::sqrt((xi - xm)*(xi - xm) + (yi - ym)*(yi - ym));
    
    bool is_singular = (dist_to_elem < 0.1 * elem_length);
    
    if (is_singular) {
        // For singular integrals, use special treatment
        // The H integral has a free term of -0.5 * I
        H[0][0] = -0.5;
        H[1][1] = -0.5;
        
        // G integral is weakly singular and can be integrated numerically
        // Use higher order quadrature
        int n_gauss = 20;
        std::vector<double> gauss_points, gauss_weights;
        
        // Gauss-Legendre quadrature points and weights on [-1, 1]
        // For simplicity, using a uniform rule here (you can substitute true Gauss points)
        for (int i = 0; i < n_gauss; i++) {
            double s = -1.0 + 2.0 * (i + 0.5) / n_gauss;
            gauss_points.push_back(s);
            gauss_weights.push_back(2.0 / n_gauss);
        }
        
        for (int i = 0; i < n_gauss; i++) {
            double s = gauss_points[i];
            double w = gauss_weights[i];
            
            // Map from [-1, 1] to element
            double x = 0.5 * ((1.0 - s) * elem.x1 + (1.0 + s) * elem.x2);
            double y = 0.5 * ((1.0 - s) * elem.y1 + (1.0 + s) * elem.y2);
            
            double U[2][2];
            kelvin_u(xi, yi, x, y, nx, ny, U);
            
            double jacobian = elem_length / 2.0;
            
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    G_mat[ii][jj] += U[ii][jj] * w * jacobian;
                }
            }
        }
    } else {
        // Regular integration - use standard Gauss quadrature
        int n_gauss = 8;
        std::vector<double> gauss_points, gauss_weights;
        
        for (int i = 0; i < n_gauss; i++) {
            double s = -1.0 + 2.0 * (i + 0.5) / n_gauss;
            gauss_points.push_back(s);
            gauss_weights.push_back(2.0 / n_gauss);
        }
        
        for (int i = 0; i < n_gauss; i++) {
            double s = gauss_points[i];
            double w = gauss_weights[i];
            
            double x = 0.5 * ((1.0 - s) * elem.x1 + (1.0 + s) * elem.x2);
            double y = 0.5 * ((1.0 - s) * elem.y1 + (1.0 + s) * elem.y2);
            
            double U[2][2], T[2][2];
            kelvin_u(xi, yi, x, y, nx, ny, U);
            kelvin_t(xi, yi, x, y, nx, ny, T);
            
            double jacobian = elem_length / 2.0;
            
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    G_mat[ii][jj] += U[ii][jj] * w * jacobian;
                    H[ii][jj] += T[ii][jj] * w * jacobian;
                }
            }
        }
    }
}

void BEMSolver2D::solve(std::vector<double>& u_x, std::vector<double>& u_y,
                        std::vector<double>& t_x, std::vector<double>& t_y) {
    
    int n = elements.size();
    if (n == 0) {
        throw std::runtime_error("No boundary elements defined");
    }
    
    // Initialize solution vectors
    u_x.resize(n);
    u_y.resize(n);
    t_x.resize(n);
    t_y.resize(n);
    
    // Build the system matrix: [A]{x} = {b}
    // The unknowns are organized as either displacements or tractions
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*n, 2*n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2*n);
    
    // For each collocation point (element midpoint)
    for (int i = 0; i < n; i++) {
        double xi, yi;
        elements[i].midpoint(xi, yi);
        
        // For each source element
        for (int j = 0; j < n; j++) {
            double H[2][2], G[2][2];
            integrate_element(elements[j], xi, yi, H, G);
            
            if (elements[j].is_traction_bc) {
                // Known traction, unknown displacement
                // H * u = G * t  =>  H * u - G * t = 0
                // Move G*t to RHS since t is known
                
                // Equation for u_x at node i
                A(2*i, 2*j) = H[0][0];      // coefficient for u_x at node j
                A(2*i, 2*j+1) = H[0][1];    // coefficient for u_y at node j
                
                b(2*i) += G[0][0] * elements[j].bc_x + G[0][1] * elements[j].bc_y;
                
                // Equation for u_y at node i
                A(2*i+1, 2*j) = H[1][0];
                A(2*i+1, 2*j+1) = H[1][1];
                
                b(2*i+1) += G[1][0] * elements[j].bc_x + G[1][1] * elements[j].bc_y;
                
            } else {
                // Known displacement, unknown traction
                // H * u = G * t  =>  G * t = H * u
                // Move H*u to RHS since u is known
                
                // Equation for t_x at node i (solving for traction components)
                A(2*i, 2*j) = -G[0][0];
                A(2*i, 2*j+1) = -G[0][1];
                
                b(2*i) += H[0][0] * elements[j].bc_x + H[0][1] * elements[j].bc_y;
                
                A(2*i+1, 2*j) = -G[1][0];
                A(2*i+1, 2*j+1) = -G[1][1];
                
                b(2*i+1) += H[1][0] * elements[j].bc_x + H[1][1] * elements[j].bc_y;
            }
        }
    }
    
    // Solve the system
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    
    // Extract solutions
    for (int i = 0; i < n; i++) {
        if (elements[i].is_traction_bc) {
            // We solved for displacement
            u_x[i] = x(2*i);
            u_y[i] = x(2*i+1);
            t_x[i] = elements[i].bc_x;
            t_y[i] = elements[i].bc_y;
        } else {
            // We solved for traction
            t_x[i] = x(2*i);
            t_y[i] = x(2*i+1);
            u_x[i] = elements[i].bc_x;
            u_y[i] = elements[i].bc_y;
        }
    }
}

void BEMSolver2D::compute_interior_displacement(double x, double y,
                                               double& u_x, double& u_y) const {
    u_x = 0.0;
    u_y = 0.0;
    
    for (const auto& elem : elements) {
        double nx, ny;
        elem.normal(nx, ny);
        
        // Get the solution for this element (need to pass in from solve)
        // For now, use the BC values - this should be updated to use actual solution
        double elem_u_x = elem.is_traction_bc ? 0.0 : elem.bc_x;
        double elem_u_y = elem.is_traction_bc ? 0.0 : elem.bc_y;
        double elem_t_x = elem.is_traction_bc ? elem.bc_x : 0.0;
        double elem_t_y = elem.is_traction_bc ? elem.bc_y : 0.0;
        
        // Integrate over element
        int n_gauss = 8;
        for (int i = 0; i < n_gauss; i++) {
            double s = -1.0 + 2.0 * (i + 0.5) / n_gauss;
            double w = 2.0 / n_gauss;
            
            double xs = 0.5 * ((1.0 - s) * elem.x1 + (1.0 + s) * elem.x2);
            double ys = 0.5 * ((1.0 - s) * elem.y1 + (1.0 + s) * elem.y2);
            
            double U[2][2], T[2][2];
            kelvin_u(x, y, xs, ys, nx, ny, U);
            kelvin_t(x, y, xs, ys, nx, ny, T);
            
            double jacobian = elem.length() / 2.0;
            
            // u = G*t - H*u (boundary integral equation)
            u_x += (U[0][0] * elem_t_x + U[0][1] * elem_t_y 
                   - T[0][0] * elem_u_x - T[0][1] * elem_u_y) * w * jacobian;
            
            u_y += (U[1][0] * elem_t_x + U[1][1] * elem_t_y
                   - T[1][0] * elem_u_x - T[1][1] * elem_u_y) * w * jacobian;
        }
    }
}

void BEMSolver2D::compute_interior_stress(double x, double y,
                                         double& sigma_xx, double& sigma_yy,
                                         double& sigma_xy) const {
    // This requires derivatives of the fundamental solution - placeholder
    sigma_xx = 0.0;
    sigma_yy = 0.0;
    sigma_xy = 0.0;
}

} // namespace BEM
