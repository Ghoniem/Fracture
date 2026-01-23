#ifndef BEM_SOLVER_H
#define BEM_SOLVER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>

namespace BEM {

// Structure to represent a boundary element
struct BoundaryElement {
    double x1, y1;  // Start point
    double x2, y2;  // End point
    bool is_traction_bc;  // true = traction BC, false = displacement BC
    double bc_x, bc_y;    // Boundary condition values (either u_x, u_y or t_x, t_y)
    
    double length() const {
        return std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    }
    
    void midpoint(double& xm, double& ym) const {
        xm = 0.5 * (x1 + x2);
        ym = 0.5 * (y1 + y2);
    }
    
    void normal(double& nx, double& ny) const {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double len = length();
        // Outward normal (assuming CCW orientation)
        nx = dy / len;
        ny = -dx / len;
    }
};

class BEMSolver2D {
private:
    double E;           // Young's modulus
    double nu;          // Poisson's ratio
    double G;           // Shear modulus
    double kappa;       // Plane strain/stress parameter
    std::vector<BoundaryElement> elements;
    
    // Kelvin fundamental solutions
    void kelvin_u(double xi, double yi, double x, double y, 
                  double nx, double ny, double (&U)[2][2]) const;
    
    void kelvin_t(double xi, double yi, double x, double y,
                  double nx, double ny, double (&T)[2][2]) const;
    
    // Numerical integration over element
    void integrate_element(const BoundaryElement& elem, double xi, double yi,
                          double (&H)[2][2], double (&G_mat)[2][2]) const;
    
public:
    BEMSolver2D(double young_modulus, double poisson_ratio, 
                bool plane_strain = true);
    
    void add_element(double x1, double y1, double x2, double y2,
                    bool is_traction, double bc_x, double bc_y);
    
    void clear_elements() { elements.clear(); }
    
    // Solve the BEM system
    void solve(std::vector<double>& u_x, std::vector<double>& u_y,
               std::vector<double>& t_x, std::vector<double>& t_y);
    
    // Compute displacement at interior point
    void compute_interior_displacement(double x, double y,
                                      double& u_x, double& u_y) const;
    
    // Compute stress at interior point
    void compute_interior_stress(double x, double y,
                                double& sigma_xx, double& sigma_yy, 
                                double& sigma_xy) const;
};

} // namespace BEM

#endif // BEM_SOLVER_H
