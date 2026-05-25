// pybind11 bindings for the VCEM_4_0 BEM C++ kernel.
//
// Python-side API mirrors fracture_utils.Ubem.bem_solver.BEMSolver2D as
// closely as possible so notebooks can swap the implementation behind a
// flag.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include "vcem/bem_solver.h"
#include "vcem/edge_dislocation.h"
#include "vcem/kelvin.h"
#include "vcem/kkt.h"
#include "vcem/material.h"

#if defined(VCEM_HAVE_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;
using vcem::bem::BEMSolver2D;

PYBIND11_MODULE(bem_cpp, m) {
    m.doc() = "VCEM_4_0 BEM C++ kernel (Eigen + pybind11)";
    m.attr("__version__") = "0.1.0";

#if defined(VCEM_HAVE_OPENMP)
    m.attr("openmp_available") = true;
    m.def("openmp_max_threads", []() { return omp_get_max_threads(); },
          "Maximum thread count OpenMP will use for parallel sections.");
    m.def("openmp_set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("n"),
          "Set the number of threads for subsequent parallel sections.");
#else
    m.attr("openmp_available") = false;
    m.def("openmp_max_threads", []() { return 1; });
    m.def("openmp_set_num_threads", [](int) { /* no-op without OpenMP */ });
#endif

    // ── Kernel introspection helpers (mostly for unit tests) ──────────────
    m.def("kelvin_U",
        [](double xf, double yf, double xs, double ys,
           double E, double nu, bool plane_strain, double r_floor) {
            return vcem::bem::kelvin_U(xf, yf, xs, ys, E, nu, plane_strain, r_floor);
        },
        py::arg("xf"), py::arg("yf"), py::arg("xs"), py::arg("ys"),
        py::arg("E"), py::arg("nu"), py::arg("plane_strain"),
        py::arg("r_floor") = 1e-16);

    m.def("kelvin_T",
        [](double xf, double yf, double xs, double ys,
           double nx, double ny,
           double E, double nu, bool plane_strain, double r_floor) {
            return vcem::bem::kelvin_T(xf, yf, xs, ys, nx, ny,
                                       E, nu, plane_strain, r_floor);
        },
        py::arg("xf"), py::arg("yf"), py::arg("xs"), py::arg("ys"),
        py::arg("nx"), py::arg("ny"),
        py::arg("E"), py::arg("nu"), py::arg("plane_strain"),
        py::arg("r_floor") = 1e-16);

    m.def("kelvin_dU_dfield",
        [](double xf, double yf, double xs, double ys,
           double E, double nu, bool plane_strain, double r_floor) {
            Eigen::Matrix2d dUdx, dUdy;
            vcem::bem::kelvin_dU_dfield(xf, yf, xs, ys,
                                        E, nu, plane_strain,
                                        dUdx, dUdy, r_floor);
            return py::make_tuple(dUdx, dUdy);
        },
        py::arg("xf"), py::arg("yf"), py::arg("xs"), py::arg("ys"),
        py::arg("E"), py::arg("nu"), py::arg("plane_strain"),
        py::arg("r_floor") = 1e-16);

    // ── Edge-dislocation (crack) kernels — Phase 1 of KKT port ────────────
    m.def("edge_dislocation_u",
        [](double dx, double dy, double dBx, double dBy,
           double nu, bool plane_stress) {
            double ux, uy;
            vcem::crack::edge_dislocation_u(dx, dy, dBx, dBy, nu, plane_stress, ux, uy);
            return py::make_tuple(ux, uy);
        },
        py::arg("dx"), py::arg("dy"), py::arg("dBx"), py::arg("dBy"),
        py::arg("nu"), py::arg("plane_stress") = false);

    // ── KKT solver — Phase 2 of KKT pipeline port ────────────────────────
    py::enum_<vcem::crack::KKTBackend>(m, "KKTBackend")
        .value("AutoLU",  vcem::crack::KKTBackend::AutoLU)
        .value("DenseLU", vcem::crack::KKTBackend::DenseLU)
        .value("BDCSVD",  vcem::crack::KKTBackend::BDCSVD)
        .export_values();

    py::class_<vcem::crack::KKTOptions>(m, "KKTOptions")
        .def(py::init<>())
        .def_readwrite("backend",        &vcem::crack::KKTOptions::backend)
        .def_readwrite("ridge",          &vcem::crack::KKTOptions::ridge)
        .def_readwrite("constraint_tol", &vcem::crack::KKTOptions::constraint_tol);

    m.def("compress_constraints",
        [](const Eigen::Ref<const Eigen::MatrixXd>& C,
           const Eigen::Ref<const Eigen::VectorXd>& d,
           double tol) {
            Eigen::MatrixXd Cc; Eigen::VectorXd dc;
            vcem::crack::compress_constraints(C, d, tol, Cc, dc);
            return py::make_tuple(Cc, dc);
        },
        py::arg("C"), py::arg("d"), py::arg("tol") = 1e-12);

    m.def("solve_kkt_lsq_eq",
        [](const Eigen::Ref<const Eigen::MatrixXd>& K,
           const Eigen::Ref<const Eigen::VectorXd>& rhs,
           const Eigen::Ref<const Eigen::MatrixXd>& C,
           const Eigen::Ref<const Eigen::VectorXd>& d,
           double ridge,
           vcem::crack::KKTBackend backend,
           double constraint_tol) {
            vcem::crack::KKTOptions opts;
            opts.backend = backend;
            opts.ridge = ridge;
            opts.constraint_tol = constraint_tol;
            return vcem::crack::solve_kkt_lsq_eq(K, rhs, C, d, opts);
        },
        py::arg("K"), py::arg("rhs"), py::arg("C"), py::arg("d"),
        py::arg("ridge") = 0.0,
        py::arg("backend") = vcem::crack::KKTBackend::AutoLU,
        py::arg("constraint_tol") = 1e-12);

    m.def("edge_dislocation_stress",
        [](double dx, double dy, double dBx, double dBy,
           double mu, double nu, bool plane_stress) {
            double sxx, syy, sxy;
            vcem::crack::edge_dislocation_stress(dx, dy, dBx, dBy, mu, nu, plane_stress,
                                                 sxx, syy, sxy);
            return py::make_tuple(sxx, syy, sxy);
        },
        py::arg("dx"), py::arg("dy"), py::arg("dBx"), py::arg("dBy"),
        py::arg("mu"), py::arg("nu"), py::arg("plane_stress") = false);

    m.def("kelvin_dT_dfield",
        [](double xf, double yf, double xs, double ys,
           double nx, double ny,
           double E, double nu, bool plane_strain, double r_floor) {
            Eigen::Matrix2d dTdx, dTdy;
            vcem::bem::kelvin_dT_dfield(xf, yf, xs, ys, nx, ny,
                                        E, nu, plane_strain,
                                        dTdx, dTdy, r_floor);
            return py::make_tuple(dTdx, dTdy);
        },
        py::arg("xf"), py::arg("yf"), py::arg("xs"), py::arg("ys"),
        py::arg("nx"), py::arg("ny"),
        py::arg("E"), py::arg("nu"), py::arg("plane_strain"),
        py::arg("r_floor") = 1e-16);

    // ── BEMSolver2D ───────────────────────────────────────────────────────
    py::class_<BEMSolver2D>(m, "BEMSolver2D")
        .def(py::init<double, double, double, bool>(),
             py::arg("E"), py::arg("nu"),
             py::arg("h") = 1.0,
             py::arg("plane_strain") = true)

        .def("add_element", &BEMSolver2D::add_element,
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("is_traction"), py::arg("bc_x"), py::arg("bc_y"))

        .def("solve", &BEMSolver2D::solve, py::arg("gauss_n") = 8)

        .def("compute_displacement_at_point",
            [](const BEMSolver2D& self, double x, double y, int gauss_n) {
                double ux, uy;
                self.compute_displacement_at_point(x, y, gauss_n, ux, uy);
                return py::make_tuple(ux, uy);
            },
            py::arg("x"), py::arg("y"), py::arg("gauss_n") = 12)

        .def("compute_grad_u_at_point", &BEMSolver2D::compute_grad_u_at_point,
             py::arg("x"), py::arg("y"), py::arg("gauss_n") = 12)

        .def("compute_stress_at_point",
            [](const BEMSolver2D& self, double x, double y, int gauss_n) {
                double sxx, syy, sxy;
                self.compute_stress_at_point(x, y, gauss_n, sxx, syy, sxy);
                return py::make_tuple(sxx, syy, sxy);
            },
            py::arg("x"), py::arg("y"), py::arg("gauss_n") = 12)

        .def("stress_on_grid",
            [](const BEMSolver2D& self,
               const std::vector<double>& xs,
               const std::vector<double>& ys,
               int gauss_n) {
                Eigen::MatrixXd Sxx, Syy, Sxy;
                self.stress_on_grid(xs, ys, gauss_n, Sxx, Syy, Sxy);
                return py::make_tuple(Sxx, Syy, Sxy);
            },
            py::arg("xs"), py::arg("ys"), py::arg("gauss_n") = 12,
            "Evaluate (Sxx, Syy, Sxy) on the tensor-product grid (ys, xs). "
            "Output arrays have shape (len(ys), len(xs)).")

        .def("stress_at_points",
            [](const BEMSolver2D& self,
               const std::vector<double>& xs,
               const std::vector<double>& ys,
               int gauss_n) {
                Eigen::VectorXd Sxx, Syy, Sxy;
                self.stress_at_points(xs, ys, gauss_n, Sxx, Syy, Sxy);
                return py::make_tuple(Sxx, Syy, Sxy);
            },
            py::arg("xs"), py::arg("ys"), py::arg("gauss_n") = 12,
            "Evaluate (Sxx, Syy, Sxy) at the scattered points (xs[i], ys[i]). "
            "Use for polar / disk-conforming / arbitrary clouds of points. "
            "Returns 1-D arrays of length len(xs).")

        // Read-only properties returning numpy arrays (zero-copy via std::vector ref)
        .def_property_readonly("u_x", [](const BEMSolver2D& s) {
            return py::array_t<double>(s.u_x().size(), s.u_x().data());
        })
        .def_property_readonly("u_y", [](const BEMSolver2D& s) {
            return py::array_t<double>(s.u_y().size(), s.u_y().data());
        })
        .def_property_readonly("t_x", [](const BEMSolver2D& s) {
            return py::array_t<double>(s.t_x().size(), s.t_x().data());
        })
        .def_property_readonly("t_y", [](const BEMSolver2D& s) {
            return py::array_t<double>(s.t_y().size(), s.t_y().data());
        })

        .def_property_readonly("E",            &BEMSolver2D::E)
        .def_property_readonly("nu",           &BEMSolver2D::nu)
        .def_property_readonly("h",            &BEMSolver2D::h)
        .def_property_readonly("plane_strain", &BEMSolver2D::plane_strain)
        .def_property_readonly("N",
            [](const BEMSolver2D& s) { return static_cast<int>(s.segments().size()); })
    ;
}
