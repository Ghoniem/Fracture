#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bem_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(bem_solver, m) {
    m.doc() = "2D Boundary Element Method solver for elasticity";
    
    py::class_<BEM::BEMSolver2D>(m, "BEMSolver2D")
        .def(py::init<double, double, bool>(),
             py::arg("young_modulus"),
             py::arg("poisson_ratio"),
             py::arg("plane_strain") = true,
             "Initialize BEM solver with material properties")
        
        .def("add_element", &BEM::BEMSolver2D::add_element,
             py::arg("x1"), py::arg("y1"),
             py::arg("x2"), py::arg("y2"),
             py::arg("is_traction"),
             py::arg("bc_x"), py::arg("bc_y"),
             "Add a boundary element with either traction or displacement BC")
        
        .def("clear_elements", &BEM::BEMSolver2D::clear_elements,
             "Clear all boundary elements")
        
        .def("solve", [](BEM::BEMSolver2D& self) {
            std::vector<double> u_x, u_y, t_x, t_y;
            self.solve(u_x, u_y, t_x, t_y);
            
            // Convert to numpy arrays
            py::array_t<double> u_x_arr(u_x.size(), u_x.data());
            py::array_t<double> u_y_arr(u_y.size(), u_y.data());
            py::array_t<double> t_x_arr(t_x.size(), t_x.data());
            py::array_t<double> t_y_arr(t_y.size(), t_y.data());
            
            return py::make_tuple(
                py::array(u_x_arr),
                py::array(u_y_arr),
                py::array(t_x_arr),
                py::array(t_y_arr)
            );
        }, "Solve the BEM system and return (u_x, u_y, t_x, t_y)")
        
        .def("compute_interior_displacement", 
             [](const BEM::BEMSolver2D& self, double x, double y) {
                 double u_x, u_y;
                 self.compute_interior_displacement(x, y, u_x, u_y);
                 return py::make_tuple(u_x, u_y);
             },
             py::arg("x"), py::arg("y"),
             "Compute displacement at interior point")
        
        .def("compute_interior_stress",
             [](const BEM::BEMSolver2D& self, double x, double y) {
                 double sigma_xx, sigma_yy, sigma_xy;
                 self.compute_interior_stress(x, y, sigma_xx, sigma_yy, sigma_xy);
                 return py::make_tuple(sigma_xx, sigma_yy, sigma_xy);
             },
             py::arg("x"), py::arg("y"),
             "Compute stress at interior point");
}
