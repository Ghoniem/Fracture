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
#include "vcem/crack_assemblers.h"
#include "vcem/edge_dislocation.h"
#include "vcem/edge_intersection.h"
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

    // ── Crack assembler (assemble_operator) — Phase 3 of KKT port ────────
    //
    // Python helper: bem_cpp.assemble_operator(poly_panels, offsets, nunk,
    //                                         E, nu, plane_stress,
    //                                         sigma_at_col)
    // expects `poly_panels` to be the list-of-dicts produced by
    // fracture_utils.Usolver.build_discretize.discretize_polylines.
    m.def("assemble_operator",
        [](py::list poly_panels,
           const std::vector<int>& offsets,
           int nunk,
           double E, double nu, bool plane_stress,
           const Eigen::Ref<const Eigen::MatrixXd>& sigma_at_col) {
            // Unpack list of Python dicts into PolyPanelData structs.
            std::vector<vcem::crack::PolyPanelData> panels;
            panels.reserve(poly_panels.size());
            for (py::handle h : poly_panels) {
                py::dict d = h.cast<py::dict>();
                vcem::crack::PolyPanelData p;
                p.x_col   = d["x_col"].cast<Eigen::MatrixXd>();
                p.t_col   = d["t_col"].cast<Eigen::MatrixXd>();
                p.n_col   = d["n_col"].cast<Eigen::MatrixXd>();
                p.src_pts = d["src_pts"].cast<Eigen::MatrixXd>();
                p.src_t   = d["src_t"].cast<Eigen::MatrixXd>();
                p.src_n   = d["src_n"].cast<Eigen::MatrixXd>();
                p.src_w   = d["src_w"].cast<Eigen::VectorXd>();
                p.Np      = d["Np"].cast<int>();
                py::list psrc = d["panel_src"].cast<py::list>();
                p.panel_src.reserve(psrc.size());
                for (py::handle pp : psrc) {
                    py::tuple t = pp.cast<py::tuple>();
                    p.panel_src.emplace_back(t[0].cast<int>(), t[1].cast<int>());
                }
                panels.push_back(std::move(p));
            }
            Eigen::MatrixXd K;
            Eigen::VectorXd rhs;
            vcem::crack::assemble_operator(
                panels, offsets, nunk, E, nu, plane_stress,
                sigma_at_col, K, rhs);
            return py::make_tuple(K, rhs);
        },
        py::arg("poly_panels"), py::arg("offsets"), py::arg("nunk"),
        py::arg("E"), py::arg("nu"), py::arg("plane_stress"),
        py::arg("sigma_at_col"),
        "Assemble the crack equilibrium operator K (2*ncol_tot, nunk) and "
        "the applied-stress rhs (2*ncol_tot,) for the given polyline-panel "
        "discretization. Parallel via OpenMP over collocation polylines. "
        "sigma_at_col must be (ncol_tot, 3) with columns [Sxx, Syy, Sxy].");

    // ── Topology: segment-segment intersection detection ─────────────────
    // Returns an (M, 4) numpy array with columns [i, j, x, y]. i and j are
    // cast to double for uniform return type; caller can cast back to int.
    m.def("detect_segment_intersections",
        [](const Eigen::Ref<const Eigen::MatrixXd>& edges_p1,
           const Eigen::Ref<const Eigen::MatrixXd>& edges_p2,
           const Eigen::Ref<const Eigen::MatrixXi>& edge_nodes,
           double endpoint_tol,
           double parallel_tol) {
            auto recs = vcem::topology::detect_segment_intersections(
                edges_p1, edges_p2, edge_nodes, endpoint_tol, parallel_tol);
            Eigen::MatrixXd out(static_cast<Eigen::Index>(recs.size()), 4);
            for (std::size_t k = 0; k < recs.size(); ++k) {
                out(static_cast<Eigen::Index>(k), 0) = static_cast<double>(recs[k].i);
                out(static_cast<Eigen::Index>(k), 1) = static_cast<double>(recs[k].j);
                out(static_cast<Eigen::Index>(k), 2) = recs[k].x;
                out(static_cast<Eigen::Index>(k), 3) = recs[k].y;
            }
            return out;
        },
        py::arg("edges_p1"), py::arg("edges_p2"), py::arg("edge_nodes"),
        py::arg("endpoint_tol") = 0.05,
        py::arg("parallel_tol") = 1.0e-10,
        "Brute-force O(E^2) segment-segment intersection detection, "
        "OpenMP-parallel over the outer edge loop. Returns an (M, 4) "
        "matrix [i, j, x, y] sorted by (i, j); skips pairs that share a "
        "vertex (adjacent edges).");

    // Helper used by every crack-assembler binding: unpack a Python
    // poly_panels list-of-dicts into a vector<PolyPanelData>.
    auto unpack_panels = [](py::list poly_panels) {
        std::vector<vcem::crack::PolyPanelData> panels;
        panels.reserve(poly_panels.size());
        for (py::handle h : poly_panels) {
            py::dict d = h.cast<py::dict>();
            vcem::crack::PolyPanelData p;
            p.x_col   = d["x_col"].cast<Eigen::MatrixXd>();
            p.t_col   = d["t_col"].cast<Eigen::MatrixXd>();
            p.n_col   = d["n_col"].cast<Eigen::MatrixXd>();
            p.src_pts = d["src_pts"].cast<Eigen::MatrixXd>();
            p.src_t   = d["src_t"].cast<Eigen::MatrixXd>();
            p.src_n   = d["src_n"].cast<Eigen::MatrixXd>();
            p.src_w   = d["src_w"].cast<Eigen::VectorXd>();
            p.Np      = d["Np"].cast<int>();
            py::list psrc = d["panel_src"].cast<py::list>();
            p.panel_src.reserve(psrc.size());
            for (py::handle pp : psrc) {
                py::tuple t = pp.cast<py::tuple>();
                p.panel_src.emplace_back(t[0].cast<int>(), t[1].cast<int>());
            }
            panels.push_back(std::move(p));
        }
        return panels;
    };

    m.def("assemble_boundary_traction_operator",
        [unpack_panels](py::list poly_panels,
                        const std::vector<int>& offsets,
                        int nunk,
                        double E, double nu, bool plane_stress,
                        const Eigen::Ref<const Eigen::MatrixXd>& boundary_xy,
                        const Eigen::Ref<const Eigen::MatrixXd>& boundary_n) {
            auto panels = unpack_panels(poly_panels);
            Eigen::MatrixXd Mt;
            vcem::crack::assemble_boundary_traction_operator(
                panels, offsets, nunk, E, nu, plane_stress,
                boundary_xy, boundary_n, Mt);
            return Mt;
        },
        py::arg("poly_panels"), py::arg("offsets"), py::arg("nunk"),
        py::arg("E"), py::arg("nu"), py::arg("plane_stress"),
        py::arg("boundary_xy"), py::arg("boundary_n"),
        "Crack-induced boundary traction operator Mt (2*Nb, nunk).");

    m.def("assemble_boundary_displacement_operator",
        [unpack_panels](py::list poly_panels,
                        const std::vector<int>& offsets,
                        int nunk,
                        double nu, bool plane_stress,
                        const Eigen::Ref<const Eigen::MatrixXd>& boundary_xy) {
            auto panels = unpack_panels(poly_panels);
            Eigen::MatrixXd Mu;
            vcem::crack::assemble_boundary_displacement_operator(
                panels, offsets, nunk, nu, plane_stress, boundary_xy, Mu);
            return Mu;
        },
        py::arg("poly_panels"), py::arg("offsets"), py::arg("nunk"),
        py::arg("nu"), py::arg("plane_stress"),
        py::arg("boundary_xy"),
        "Crack-induced boundary displacement operator Mu (2*Nb, nunk).");

    m.def("assemble_bem_boundary_to_crack_traction_operator",
        [unpack_panels](py::list poly_panels,
                        double E, double nu, bool plane_stress,
                        const Eigen::Ref<const Eigen::VectorXd>& boundary_x1,
                        const Eigen::Ref<const Eigen::VectorXd>& boundary_y1,
                        const Eigen::Ref<const Eigen::VectorXd>& boundary_x2,
                        const Eigen::Ref<const Eigen::VectorXd>& boundary_y2,
                        int gauss_n) {
            auto panels = unpack_panels(poly_panels);
            Eigen::MatrixXd Nbc;
            vcem::crack::assemble_bem_boundary_to_crack_traction_operator(
                panels, E, nu, plane_stress,
                boundary_x1, boundary_y1, boundary_x2, boundary_y2,
                gauss_n, Nbc);
            return Nbc;
        },
        py::arg("poly_panels"),
        py::arg("E"), py::arg("nu"), py::arg("plane_stress"),
        py::arg("boundary_x1"), py::arg("boundary_y1"),
        py::arg("boundary_x2"), py::arg("boundary_y2"),
        py::arg("gauss_n") = 4,
        "BEM boundary y=[u_bc,t_bc] -> crack collocation traction operator "
        "Nbc (2*ncol_tot, 4*Nb).");

    m.def("solve_kkt_lsq_eq",
        [](const Eigen::Ref<const Eigen::MatrixXd>& K,
           const Eigen::Ref<const Eigen::VectorXd>& rhs,
           const Eigen::Ref<const Eigen::MatrixXd>& C,
           const Eigen::Ref<const Eigen::VectorXd>& d,
           double ridge,
           vcem::crack::KKTBackend backend,
           double constraint_tol,
           const Eigen::VectorXd& ridge_diag) {
            vcem::crack::KKTOptions opts;
            opts.backend = backend;
            opts.ridge = ridge;
            opts.constraint_tol = constraint_tol;
            return vcem::crack::solve_kkt_lsq_eq(K, rhs, C, d, opts, ridge_diag);
        },
        py::arg("K"), py::arg("rhs"), py::arg("C"), py::arg("d"),
        py::arg("ridge") = 0.0,
        py::arg("backend") = vcem::crack::KKTBackend::AutoLU,
        py::arg("constraint_tol") = 1e-12,
        py::arg("ridge_diag") = Eigen::VectorXd());

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
