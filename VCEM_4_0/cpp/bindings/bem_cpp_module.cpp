// Smoke-test pybind11 module for VCEM_4_0.
//
// Once the BEM core is ported, this file will expose BEMSolver2D and
// stress_on_grid().  For now it carries a trivial echo function so the
// build, install, and import path can be validated end-to-end.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

namespace py = pybind11;

namespace vcem { namespace bem {

// Trivial Eigen round-trip: returns x + 1 element-wise.
Eigen::VectorXd add_one(const Eigen::Ref<const Eigen::VectorXd>& x) {
    return x.array() + 1.0;
}

}}  // namespace vcem::bem

PYBIND11_MODULE(bem_cpp, m) {
    m.doc() = "VCEM_4_0 BEM C++ kernel (skeleton; full BEM port in progress)";

    m.attr("__version__") = "0.0.1";

    m.def("add_one", &vcem::bem::add_one,
          py::arg("x"),
          "Smoke test: return x + 1 element-wise (validates pybind11+Eigen link).");
}
