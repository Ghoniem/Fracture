#pragma once
//
// Crack-network assemblers (Phase 3+ of the KKT pipeline port).
// Each function maps a polyline-panel discretization (the C++ image of
// the Python `poly_panels` list) to one of the dense operators used by
// the augmented KKT solve in
// fracture_utils/Usolver/{build_discretize.py, parametrization.py}.

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace vcem { namespace crack {

// Mirror of the per-polyline fields of the Python poly_panels[i] dict
// that the C++ assemblers actually consume.
struct PolyPanelData {
    // Collocation arrays (one row per collocation point on this polyline).
    Eigen::MatrixXd x_col;   // (Nci, 2)
    Eigen::MatrixXd t_col;   // (Nci, 2)  unit tangent at collocation point
    Eigen::MatrixXd n_col;   // (Nci, 2)  unit normal at collocation point

    // Panel quadrature: Np panels, each with a (s_start, s_end) slice into
    // src_* arrays. src_* arrays carry all Np * nq quadrature points
    // concatenated.
    int Np = 0;
    std::vector<std::pair<int, int>> panel_src;
    Eigen::MatrixXd src_pts;  // (Nsrc, 2)
    Eigen::MatrixXd src_t;    // (Nsrc, 2)  tangent at source quadrature point
    Eigen::MatrixXd src_n;    // (Nsrc, 2)  normal at source quadrature point
    Eigen::VectorXd src_w;    // (Nsrc,)    quadrature weight (incl. singular factor)
};

// Assemble the crack equilibrium operator K (2*ncol_tot, nunk) and the
// applied-stress rhs vector (2*ncol_tot,). Direct port of
// build_discretize.assemble_operator.
//
// sigma_at_col is (ncol_tot, 3) with columns [Sxx, Syy, Sxy] of the
// applied stress evaluated at each collocation point in the same order
// the panels list traverses them (polyline 0 first, then polyline 1, ...).
// The Python side evaluates `applied.tensor_at(X)` once and passes the
// result here so we don't bridge a Python callable across the boundary.
//
// Outputs are resized inside.
void assemble_operator(
    const std::vector<PolyPanelData>& panels,
    const std::vector<int>& offsets,
    int nunk,
    double E, double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::MatrixXd>& sigma_at_col,
    Eigen::MatrixXd& K,
    Eigen::VectorXd& rhs);

// Map crack unknowns q -> boundary tractions t_cr at the boundary points
// (boundary_xy[i], boundary_n[i]). Output shape (2*Nb, nunk) with rows
// packed as [tx0, ty0, tx1, ty1, ...]. Direct port of
// build_discretize.assemble_boundary_traction_operator.
void assemble_boundary_traction_operator(
    const std::vector<PolyPanelData>& panels,
    const std::vector<int>& offsets,
    int nunk,
    double E, double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::MatrixXd>& boundary_xy,  // (Nb, 2)
    const Eigen::Ref<const Eigen::MatrixXd>& boundary_n,   // (Nb, 2)
    Eigen::MatrixXd& Mt);

// Map crack unknowns q -> boundary displacements u_cr at boundary_xy
// points. Output shape (2*Nb, nunk) packed as [ux0, uy0, ux1, uy1, ...].
// Direct port of build_discretize.assemble_boundary_displacement_operator.
void assemble_boundary_displacement_operator(
    const std::vector<PolyPanelData>& panels,
    const std::vector<int>& offsets,
    int nunk,
    double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::MatrixXd>& boundary_xy,  // (Nb, 2)
    Eigen::MatrixXd& Mu);

// Map BEM boundary unknowns y = [u_bc(2Nb), t_bc(2Nb)] -> traction rows
// on crack collocation points. Output shape (2*ncol_tot, 4*Nb), arranged
// to be hstacked with K from assemble_operator (Kz = [K | Nbc]).
// Direct port of build_discretize.assemble_bem_boundary_to_crack_traction_operator.
void assemble_bem_boundary_to_crack_traction_operator(
    const std::vector<PolyPanelData>& panels,
    double E, double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::VectorXd>& boundary_x1,  // (Nb,)
    const Eigen::Ref<const Eigen::VectorXd>& boundary_y1,  // (Nb,)
    const Eigen::Ref<const Eigen::VectorXd>& boundary_x2,  // (Nb,)
    const Eigen::Ref<const Eigen::VectorXd>& boundary_y2,  // (Nb,)
    int gauss_n,
    Eigen::MatrixXd& Nbc);

}}  // namespace vcem::crack
