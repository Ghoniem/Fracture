#include "vcem/crack_assemblers.h"
#include "vcem/edge_dislocation.h"
#include "vcem/kelvin.h"
#include "vcem/material.h"
#include "vcem/quadrature.h"
#include "vcem/segment.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(VCEM_HAVE_OPENMP)
#include <omp.h>
#endif

namespace vcem { namespace crack {

void assemble_operator(
    const std::vector<PolyPanelData>& panels,
    const std::vector<int>& offsets,
    int nunk,
    double E, double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::MatrixXd>& sigma_at_col,
    Eigen::MatrixXd& K,
    Eigen::VectorXd& rhs)
{
    const int Npoly = static_cast<int>(panels.size());
    if (static_cast<int>(offsets.size()) != Npoly) {
        throw std::invalid_argument(
            "assemble_operator: offsets.size() != panels.size()");
    }

    // Total collocation point count and per-polyline start offsets.
    std::vector<int> col_starts(Npoly + 1, 0);
    for (int p = 0; p < Npoly; ++p) {
        col_starts[p + 1] = col_starts[p] + static_cast<int>(panels[p].x_col.rows());
    }
    const int ncol_tot = col_starts[Npoly];

    if (sigma_at_col.rows() != ncol_tot || sigma_at_col.cols() != 3) {
        throw std::invalid_argument(
            "assemble_operator: sigma_at_col must be (ncol_tot, 3) with [Sxx, Syy, Sxy]");
    }

    K   = Eigen::MatrixXd::Zero(2 * ncol_tot, nunk);
    rhs = Eigen::VectorXd::Zero(2 * ncol_tot);

    const double mu = E / (2.0 * (1.0 + nu));

    // Parallelize over collocation polylines: each writes to a disjoint
    // row block of K so there's no race condition. (Splitting at the
    // outermost loop keeps inner loops cache-friendly and avoids OpenMP
    // sync overhead per collocation point.)
#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int pid_i = 0; pid_i < Npoly; ++pid_i) {
        const PolyPanelData& pp_i = panels[pid_i];
        const int Nci  = static_cast<int>(pp_i.x_col.rows());
        const int row0 = col_starts[pid_i];

        for (int ic = 0; ic < Nci; ++ic) {
            const double xi  = pp_i.x_col(ic, 0);
            const double yi  = pp_i.x_col(ic, 1);
            const double tix = pp_i.t_col(ic, 0);
            const double tiy = pp_i.t_col(ic, 1);
            const double nix = pp_i.n_col(ic, 0);
            const double niy = pp_i.n_col(ic, 1);

            // RHS: traction from applied stress, projected onto (n_i, t_i),
            // signed -1 to move to the LHS as the negative-of-applied
            // self-equilibrium constraint.
            const double sxx_app = sigma_at_col(row0 + ic, 0);
            const double syy_app = sigma_at_col(row0 + ic, 1);
            const double sxy_app = sigma_at_col(row0 + ic, 2);
            const double tr_x = sxx_app * nix + sxy_app * niy;
            const double tr_y = sxy_app * nix + syy_app * niy;
            const double tn0  = nix * tr_x + niy * tr_y;
            const double ts0  = tix * tr_x + tiy * tr_y;
            const int row     = row0 + ic;
            rhs(row)              = -tn0;
            rhs(ncol_tot + row)   = -ts0;

            // K rows: accumulate kernel contributions from each source panel
            // and each Burgers mode (I = along source normal, II = along
            // source tangent).
            for (int pid_j = 0; pid_j < Npoly; ++pid_j) {
                const PolyPanelData& pp_j = panels[pid_j];
                const int offj = offsets[pid_j];

                for (int k = 0; k < pp_j.Np; ++k) {
                    const int s0 = pp_j.panel_src[k].first;
                    const int s1 = pp_j.panel_src[k].second;

                    double tn_I = 0.0, ts_I = 0.0;
                    double tn_II = 0.0, ts_II = 0.0;

                    for (int q = s0; q < s1; ++q) {
                        const double xs = pp_j.src_pts(q, 0);
                        const double ys = pp_j.src_pts(q, 1);
                        const double tx_s = pp_j.src_t(q, 0);
                        const double ty_s = pp_j.src_t(q, 1);
                        const double nx_s = pp_j.src_n(q, 0);
                        const double ny_s = pp_j.src_n(q, 1);
                        const double ww   = pp_j.src_w(q);

                        const double dx = xi - xs;
                        const double dy = yi - ys;

                        // ─── Mode I: Burgers = (nx_s, ny_s) * ww ────────
                        {
                            double sxx, syy, sxy;
                            edge_dislocation_stress(dx, dy,
                                                    nx_s * ww, ny_s * ww,
                                                    mu, nu, plane_stress,
                                                    sxx, syy, sxy);
                            const double tx = sxx * nix + sxy * niy;
                            const double ty = sxy * nix + syy * niy;
                            tn_I += nix * tx + niy * ty;
                            ts_I += tix * tx + tiy * ty;
                        }
                        // ─── Mode II: Burgers = (tx_s, ty_s) * ww ───────
                        {
                            double sxx, syy, sxy;
                            edge_dislocation_stress(dx, dy,
                                                    tx_s * ww, ty_s * ww,
                                                    mu, nu, plane_stress,
                                                    sxx, syy, sxy);
                            const double tx = sxx * nix + sxy * niy;
                            const double ty = sxy * nix + syy * niy;
                            tn_II += nix * tx + niy * ty;
                            ts_II += tix * tx + tiy * ty;
                        }
                    }

                    const int col_I  = offj + 2 * k + 0;
                    const int col_II = offj + 2 * k + 1;
                    K(row,            col_I)  = tn_I;
                    K(ncol_tot + row, col_I)  = ts_I;
                    K(row,            col_II) = tn_II;
                    K(ncol_tot + row, col_II) = ts_II;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Mt: maps crack unknowns q -> boundary tractions at boundary points
// ─────────────────────────────────────────────────────────────────────────
void assemble_boundary_traction_operator(
    const std::vector<PolyPanelData>& panels,
    const std::vector<int>& offsets,
    int nunk,
    double E, double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::MatrixXd>& boundary_xy,
    const Eigen::Ref<const Eigen::MatrixXd>& boundary_n,
    Eigen::MatrixXd& Mt)
{
    if (boundary_xy.cols() != 2 || boundary_n.cols() != 2 ||
        boundary_xy.rows() != boundary_n.rows()) {
        throw std::invalid_argument(
            "assemble_boundary_traction_operator: boundary_xy and boundary_n must be (Nb, 2)");
    }
    const int Nb    = static_cast<int>(boundary_xy.rows());
    const int Npoly = static_cast<int>(panels.size());

    Mt = Eigen::MatrixXd::Zero(2 * Nb, nunk);
    const double mu = E / (2.0 * (1.0 + nu));

#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int ib = 0; ib < Nb; ++ib) {
        const double x  = boundary_xy(ib, 0);
        const double y  = boundary_xy(ib, 1);
        const double nx = boundary_n(ib, 0);
        const double ny = boundary_n(ib, 1);
        const int row_tx = 2 * ib + 0;
        const int row_ty = 2 * ib + 1;

        for (int pid_j = 0; pid_j < Npoly; ++pid_j) {
            const PolyPanelData& pp_j = panels[pid_j];
            const int offj = offsets[pid_j];

            for (int k = 0; k < pp_j.Np; ++k) {
                const int s0 = pp_j.panel_src[k].first;
                const int s1 = pp_j.panel_src[k].second;

                double tx_I = 0.0, ty_I = 0.0;
                double tx_II = 0.0, ty_II = 0.0;

                for (int q = s0; q < s1; ++q) {
                    const double dx = x - pp_j.src_pts(q, 0);
                    const double dy = y - pp_j.src_pts(q, 1);
                    const double ww = pp_j.src_w(q);
                    const double tx_s = pp_j.src_t(q, 0);
                    const double ty_s = pp_j.src_t(q, 1);
                    const double nx_s = pp_j.src_n(q, 0);
                    const double ny_s = pp_j.src_n(q, 1);

                    {  // mode I (n-Burgers)
                        double sxx, syy, sxy;
                        edge_dislocation_stress(dx, dy, nx_s * ww, ny_s * ww,
                                                mu, nu, plane_stress, sxx, syy, sxy);
                        tx_I += sxx * nx + sxy * ny;
                        ty_I += sxy * nx + syy * ny;
                    }
                    {  // mode II (t-Burgers)
                        double sxx, syy, sxy;
                        edge_dislocation_stress(dx, dy, tx_s * ww, ty_s * ww,
                                                mu, nu, plane_stress, sxx, syy, sxy);
                        tx_II += sxx * nx + sxy * ny;
                        ty_II += sxy * nx + syy * ny;
                    }
                }

                const int col_I  = offj + 2 * k + 0;
                const int col_II = offj + 2 * k + 1;
                Mt(row_tx, col_I)  = tx_I;
                Mt(row_ty, col_I)  = ty_I;
                Mt(row_tx, col_II) = tx_II;
                Mt(row_ty, col_II) = ty_II;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Mu: maps crack unknowns q -> boundary displacements
// ─────────────────────────────────────────────────────────────────────────
void assemble_boundary_displacement_operator(
    const std::vector<PolyPanelData>& panels,
    const std::vector<int>& offsets,
    int nunk,
    double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::MatrixXd>& boundary_xy,
    Eigen::MatrixXd& Mu)
{
    if (boundary_xy.cols() != 2) {
        throw std::invalid_argument(
            "assemble_boundary_displacement_operator: boundary_xy must be (Nb, 2)");
    }
    const int Nb    = static_cast<int>(boundary_xy.rows());
    const int Npoly = static_cast<int>(panels.size());

    Mu = Eigen::MatrixXd::Zero(2 * Nb, nunk);

#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int ib = 0; ib < Nb; ++ib) {
        const double x = boundary_xy(ib, 0);
        const double y = boundary_xy(ib, 1);
        const int row_ux = 2 * ib + 0;
        const int row_uy = 2 * ib + 1;

        for (int pid_j = 0; pid_j < Npoly; ++pid_j) {
            const PolyPanelData& pp_j = panels[pid_j];
            const int offj = offsets[pid_j];

            for (int k = 0; k < pp_j.Np; ++k) {
                const int s0 = pp_j.panel_src[k].first;
                const int s1 = pp_j.panel_src[k].second;

                double ux_I = 0.0, uy_I = 0.0;
                double ux_II = 0.0, uy_II = 0.0;

                for (int q = s0; q < s1; ++q) {
                    const double dx = x - pp_j.src_pts(q, 0);
                    const double dy = y - pp_j.src_pts(q, 1);
                    const double ww = pp_j.src_w(q);
                    const double tx_s = pp_j.src_t(q, 0);
                    const double ty_s = pp_j.src_t(q, 1);
                    const double nx_s = pp_j.src_n(q, 0);
                    const double ny_s = pp_j.src_n(q, 1);

                    {  // mode I
                        double ux, uy;
                        edge_dislocation_u(dx, dy, nx_s * ww, ny_s * ww,
                                            nu, plane_stress, ux, uy);
                        ux_I += ux; uy_I += uy;
                    }
                    {  // mode II
                        double ux, uy;
                        edge_dislocation_u(dx, dy, tx_s * ww, ty_s * ww,
                                            nu, plane_stress, ux, uy);
                        ux_II += ux; uy_II += uy;
                    }
                }

                const int col_I  = offj + 2 * k + 0;
                const int col_II = offj + 2 * k + 1;
                Mu(row_ux, col_I)  = ux_I;
                Mu(row_uy, col_I)  = uy_I;
                Mu(row_ux, col_II) = ux_II;
                Mu(row_uy, col_II) = uy_II;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Nbc: BEM boundary y=[u_bc, t_bc] -> crack collocation tractions
// uses the Kelvin kernels we already have in vcem::bem
// ─────────────────────────────────────────────────────────────────────────
void assemble_bem_boundary_to_crack_traction_operator(
    const std::vector<PolyPanelData>& panels,
    double E, double nu, bool plane_stress,
    const Eigen::Ref<const Eigen::VectorXd>& boundary_x1,
    const Eigen::Ref<const Eigen::VectorXd>& boundary_y1,
    const Eigen::Ref<const Eigen::VectorXd>& boundary_x2,
    const Eigen::Ref<const Eigen::VectorXd>& boundary_y2,
    int gauss_n,
    Eigen::MatrixXd& Nbc)
{
    if (boundary_x1.size() != boundary_y1.size() ||
        boundary_x1.size() != boundary_x2.size() ||
        boundary_x1.size() != boundary_y2.size()) {
        throw std::invalid_argument(
            "assemble_bem_boundary_to_crack_traction_operator: "
            "boundary endpoint arrays must have equal lengths");
    }
    const int Nb    = static_cast<int>(boundary_x1.size());
    const int Npoly = static_cast<int>(panels.size());

    int ncol_tot = 0;
    std::vector<int> col_starts(Npoly + 1, 0);
    for (int p = 0; p < Npoly; ++p) {
        col_starts[p + 1] = col_starts[p] + static_cast<int>(panels[p].x_col.rows());
    }
    ncol_tot = col_starts[Npoly];

    if (Nb == 0 || ncol_tot == 0) {
        Nbc = Eigen::MatrixXd::Zero(2 * ncol_tot, 4 * Nb);
        return;
    }

    Nbc = Eigen::MatrixXd::Zero(2 * ncol_tot, 4 * Nb);

    const bool   plane_strain = !plane_stress;
    const double shear_mod    = vcem::bem::shear_modulus(E, nu);
    const double lam          = vcem::bem::lame_lambda(E, nu, plane_strain);

    // Pre-fetch the Gauss rule once outside the parallel region (the
    // legendre cache uses a mutex; we never call it inside the loop).
    const int qn  = std::max(2, gauss_n);
    const vcem::bem::GaussRule& gr = vcem::bem::gauss_legendre(qn);

    // Pre-build C++ Segment objects for each boundary panel so we can
    // reuse map_to_segment / segment .nx() / .ny() inside the loop.
    std::vector<vcem::bem::Segment> segs(Nb);
    for (int i = 0; i < Nb; ++i) {
        segs[i].x1 = boundary_x1(i);
        segs[i].y1 = boundary_y1(i);
        segs[i].x2 = boundary_x2(i);
        segs[i].y2 = boundary_y2(i);
        segs[i].is_traction = true;
        segs[i].bc_x = 0.0;
        segs[i].bc_y = 0.0;
    }

#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int pid_i = 0; pid_i < Npoly; ++pid_i) {
        const PolyPanelData& pp = panels[pid_i];
        const int Nci  = static_cast<int>(pp.x_col.rows());
        const int row0_poly = col_starts[pid_i];

        for (int ic = 0; ic < Nci; ++ic) {
            const double xi  = pp.x_col(ic, 0);
            const double yi  = pp.x_col(ic, 1);
            const double tix = pp.t_col(ic, 0);
            const double tiy = pp.t_col(ic, 1);
            const double nix = pp.n_col(ic, 0);
            const double niy = pp.n_col(ic, 1);

            const int row_tn = row0_poly + ic;
            const int row_ts = ncol_tot + row_tn;

            for (int j = 0; j < Nb; ++j) {
                const vcem::bem::Segment& sj = segs[j];
                const double rf  = 1.0e-16 * std::max(sj.length(), 1.0e-12);
                const double njx = sj.nx();
                const double njy = sj.ny();

                double c_ux = 0.0, c_uy = 0.0, c_tx = 0.0, c_ty = 0.0;
                double c2_ux = 0.0, c2_uy = 0.0, c2_tx = 0.0, c2_ty = 0.0;

                Eigen::Matrix2d dUdx, dUdy, dTdx, dTdy;

                for (int qi = 0; qi < qn; ++qi) {
                    const double s = gr.nodes[qi];
                    const double w = gr.weights[qi];
                    double xq, yq, jac;
                    vcem::bem::map_to_segment(sj, s, xq, yq, jac);
                    const double ww = w * jac;

                    vcem::bem::kelvin_dU_dfield(xi, yi, xq, yq,
                                                E, nu, plane_strain,
                                                dUdx, dUdy, rf);
                    vcem::bem::kelvin_dT_dfield(xi, yi, xq, yq, njx, njy,
                                                E, nu, plane_strain,
                                                dTdx, dTdy, rf);

                    // u contributions: dudx = -dTdx * u, dudy = -dTdy * u
                    for (int kdof = 0; kdof < 2; ++kdof) {
                        const double dux_dx = -dTdx(0, kdof) * ww;
                        const double duy_dy = -dTdy(1, kdof) * ww;
                        const double dux_dy = -dTdy(0, kdof) * ww;
                        const double duy_dx = -dTdx(1, kdof) * ww;
                        const double exx = dux_dx;
                        const double eyy = duy_dy;
                        const double exy = 0.5 * (dux_dy + duy_dx);
                        const double tr  = exx + eyy;
                        const double sxx = lam * tr + 2.0 * shear_mod * exx;
                        const double syy = lam * tr + 2.0 * shear_mod * eyy;
                        const double sxy = 2.0 * shear_mod * exy;
                        const double txv = sxx * nix + sxy * niy;
                        const double tyv = sxy * nix + syy * niy;
                        const double tn  = nix * txv + niy * tyv;
                        const double tsv = tix * txv + tiy * tyv;
                        if (kdof == 0) { c_ux  += tn; c2_ux += tsv; }
                        else            { c_uy  += tn; c2_uy += tsv; }
                    }
                    // t contributions: dudx = dUdx * t, dudy = dUdy * t
                    for (int kdof = 0; kdof < 2; ++kdof) {
                        const double dux_dx = dUdx(0, kdof) * ww;
                        const double duy_dy = dUdy(1, kdof) * ww;
                        const double dux_dy = dUdy(0, kdof) * ww;
                        const double duy_dx = dUdx(1, kdof) * ww;
                        const double exx = dux_dx;
                        const double eyy = duy_dy;
                        const double exy = 0.5 * (dux_dy + duy_dx);
                        const double tr  = exx + eyy;
                        const double sxx = lam * tr + 2.0 * shear_mod * exx;
                        const double syy = lam * tr + 2.0 * shear_mod * eyy;
                        const double sxy = 2.0 * shear_mod * exy;
                        const double txv = sxx * nix + sxy * niy;
                        const double tyv = sxy * nix + syy * niy;
                        const double tn  = nix * txv + niy * tyv;
                        const double tsv = tix * txv + tiy * tyv;
                        if (kdof == 0) { c_tx += tn; c2_tx += tsv; }
                        else            { c_ty += tn; c2_ty += tsv; }
                    }
                }

                const int cux = 2 * j + 0;
                const int cuy = 2 * j + 1;
                const int ctx = 2 * Nb + 2 * j + 0;
                const int cty = 2 * Nb + 2 * j + 1;

                Nbc(row_tn, cux) = c_ux;
                Nbc(row_tn, cuy) = c_uy;
                Nbc(row_tn, ctx) = c_tx;
                Nbc(row_tn, cty) = c_ty;

                Nbc(row_ts, cux) = c2_ux;
                Nbc(row_ts, cuy) = c2_uy;
                Nbc(row_ts, ctx) = c2_tx;
                Nbc(row_ts, cty) = c2_ty;
            }
        }
    }
}

}}  // namespace vcem::crack
