#include "vcem/kkt.h"

#include <Eigen/SVD>
#include <stdexcept>

namespace vcem { namespace crack {

void compress_constraints(const Eigen::Ref<const Eigen::MatrixXd>& C,
                          const Eigen::Ref<const Eigen::VectorXd>& d,
                          double tol,
                          Eigen::MatrixXd& Cc,
                          Eigen::VectorXd& dc)
{
    if (C.size() == 0 || C.rows() == 0) {
        Cc = Eigen::MatrixXd(0, C.cols());
        dc = Eigen::VectorXd(0);
        return;
    }

    // SVD on C: U * diag(s) * V^T. Row-rank determined by singular-value
    // ratio against the largest one.
    Eigen::BDCSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& s = svd.singularValues();
    if (s.size() == 0) {
        Cc = Eigen::MatrixXd(0, C.cols());
        dc = Eigen::VectorXd(0);
        return;
    }

    const double s_max = s(0);
    Eigen::Index r = 0;
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        if (s(i) > tol * s_max) ++r;
        else break;   // singular values are sorted desc
    }

    if (r >= C.rows()) {
        // Already full row rank — keep original.
        Cc = C;
        dc = d.size() ? d : Eigen::VectorXd::Zero(C.rows());
        return;
    }

    // Compress: keep an orthonormal basis for the row space of C.
    // Python uses Ur^T as the projector; equivalent here:
    const Eigen::MatrixXd UrT = svd.matrixU().leftCols(r).transpose();
    Cc = UrT * C;                                   // (r, n)
    Eigen::VectorXd d_use = d.size() ? d : Eigen::VectorXd::Zero(C.rows());
    dc = UrT * d_use;                               // (r,)
}


Eigen::VectorXd solve_kkt_lsq_eq(
    const Eigen::Ref<const Eigen::MatrixXd>& K,
    const Eigen::Ref<const Eigen::VectorXd>& rhs,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const KKTOptions& opts,
    const Eigen::VectorXd& ridge_diag)
{
    const Eigen::Index n = K.cols();
    if (K.rows() != rhs.size()) {
        throw std::invalid_argument("solve_kkt_lsq_eq: K.rows() != rhs.size()");
    }
    if (C.size() > 0 && C.cols() != n) {
        throw std::invalid_argument("solve_kkt_lsq_eq: C.cols() != K.cols()");
    }
    if (d.size() > 0 && d.size() != C.rows()) {
        throw std::invalid_argument("solve_kkt_lsq_eq: len(d) != C.rows()");
    }

    // Normal equations: A = K^T K, b = K^T rhs
    Eigen::MatrixXd A = K.transpose() * K;
    Eigen::VectorXd b = K.transpose() * rhs;

    if (opts.ridge > 0.0) {
        A.diagonal().array() += opts.ridge;
    }
    if (ridge_diag.size() > 0) {
        if (ridge_diag.size() != n) {
            throw std::invalid_argument(
                "solve_kkt_lsq_eq: ridge_diag.size() must equal K.cols()");
        }
        // Match numpy: A += diag(maximum(ridge_diag, 0))
        A.diagonal().array() += ridge_diag.array().max(0.0);
    }

    // Compress constraints to full row rank (avoids singular KKT).
    Eigen::MatrixXd Cc;
    Eigen::VectorXd dc;
    if (C.size() > 0) {
        compress_constraints(C, d.size() ? d : Eigen::VectorXd::Zero(C.rows()),
                              opts.constraint_tol, Cc, dc);
    } else {
        Cc = Eigen::MatrixXd(0, n);
        dc = Eigen::VectorXd(0);
    }
    const Eigen::Index m = Cc.rows();

    // Diagonal stabilization eps * I on A — matches the Python
    //   eps = 1e-12 * max|diag(A)|
    double diag_max = 0.0;
    if (A.size() > 0) {
        diag_max = A.diagonal().cwiseAbs().maxCoeff();
    }
    double eps = 1.0e-12 * (diag_max > 0.0 ? diag_max : 1.0);
    if (!std::isfinite(eps) || eps <= 0.0) eps = 1.0e-12;
    A.diagonal().array() += eps;

    // Assemble KKT block: [[A, Cc^T], [Cc, 0]]
    const Eigen::Index N = n + m;
    Eigen::MatrixXd KKT(N, N);
    KKT.topLeftCorner(n, n) = A;
    if (m > 0) {
        KKT.topRightCorner(n, m) = Cc.transpose();
        KKT.bottomLeftCorner(m, n) = Cc;
        KKT.bottomRightCorner(m, m).setZero();
    }
    Eigen::VectorXd bb(N);
    bb.head(n) = b;
    if (m > 0) bb.tail(m) = dc;

    Eigen::VectorXd sol;

    auto solve_lu = [&]() -> bool {
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(KKT);
        // PartialPivLU has no info() flag, but very small pivots indicate
        // numerical trouble. Compare against the Python try/except: numpy
        // raises LinAlgError only on exact singularity, so we mirror that
        // by checking the LU's |det| (cheap given factorization).
        const double det = lu.determinant();
        if (!std::isfinite(det) || det == 0.0) return false;
        sol = lu.solve(bb);
        return sol.allFinite();
    };

    auto solve_svd = [&]() {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(KKT, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.setThreshold(1.0e-12);
        sol = svd.solve(bb);
    };

    switch (opts.backend) {
    case KKTBackend::DenseLU:
        if (!solve_lu()) {
            throw std::runtime_error(
                "KKT: PartialPivLU produced non-finite solution and backend=DenseLU forbids fallback");
        }
        break;
    case KKTBackend::BDCSVD:
        solve_svd();
        break;
    case KKTBackend::AutoLU:
    default:
        if (!solve_lu()) solve_svd();   // mirror Python's try/except path
        break;
    }

    return sol.head(n);
}

}}  // namespace vcem::crack
