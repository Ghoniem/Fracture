#include "vcem/crack_assemblers.h"
#include "vcem/edge_dislocation.h"

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

}}  // namespace vcem::crack
