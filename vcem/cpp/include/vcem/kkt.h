#pragma once
//
// Constrained least-squares KKT solve, direct C++/Eigen port of
// fracture_utils/Usolver/KKT.py {solve_kkt_lsq_eq, _compress_constraints}.
//
// Solves
//     min_x ||K x - rhs||^2 + ridge ||x||^2
//     s.t.  C x = d
// by forming the (n+m) x (n+m) KKT block matrix
//     [[A + ridge I + eps I,  Cc^T],
//      [             Cc    ,   0  ]]
// where Cc is C compressed to full row rank via SVD, and solving for x
// (the first n entries of the joint solution).
//
// Backend dispatch follows the RadCluster_2_0 pattern: a runtime enum
// selects the linear solver so future sparse / iterative / CUDA backends
// can drop in without changing callers.

#include <Eigen/Dense>

namespace vcem { namespace crack {

enum class KKTBackend {
    // PartialPivLU first; if Eigen reports a numerical problem, fall back
    // to BDCSVD-based least-squares. Matches Python's
    // `try np.linalg.solve except np.linalg.lstsq` exactly.
    AutoLU,
    // PartialPivLU always (fastest; throws if KKT is genuinely singular).
    DenseLU,
    // BDCSVD lstsq always (slowest; robust to rank deficiency).
    BDCSVD,
    // Reserved for future backends:
    //   SparseLU   -- Eigen::SparseLU on a sparse KKT block
    //   GMRES      -- iterative on the symmetric indefinite system
    //   CudaDense  -- cuSOLVER getrf/getrs on device
};

struct KKTOptions {
    KKTBackend backend = KKTBackend::AutoLU;
    double ridge = 0.0;            // adds ridge * I to A = K^T K
    double constraint_tol = 1e-12; // singular-value cutoff for row-rank
};

// Return a full-row-rank equivalent (Cc, dc) of the constraint pair (C, d).
// Cc has at most r = rank(C) rows; the solution set { x : C x = d } is the
// same as { x : Cc x = dc }. Equivalent to KKT.py _compress_constraints.
void compress_constraints(const Eigen::Ref<const Eigen::MatrixXd>& C,
                          const Eigen::Ref<const Eigen::VectorXd>& d,
                          double tol,
                          Eigen::MatrixXd& Cc,
                          Eigen::VectorXd& dc);

// Constrained least-squares solve. Returns x (length n = K.cols()).
//
// d may be empty (length 0) when C has 0 rows; otherwise len(d) must
// equal C.rows(). Pass an empty 0xn C if the problem is unconstrained.
// ridge_diag may be an empty vector (length 0) when unused. When non-empty,
// it must have length K.cols() and is added to the diagonal of A = K^T K
// (i.e. equivalent to numpy's `A += np.diag(np.maximum(ridge_diag, 0))`).
Eigen::VectorXd solve_kkt_lsq_eq(
    const Eigen::Ref<const Eigen::MatrixXd>& K,
    const Eigen::Ref<const Eigen::VectorXd>& rhs,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const KKTOptions& options = KKTOptions{},
    const Eigen::VectorXd& ridge_diag = Eigen::VectorXd());

}}  // namespace vcem::crack
