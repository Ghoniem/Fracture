#pragma once
//
// Gauss-Legendre nodes and weights on [-1, 1].
//
// Replicates np.polynomial.legendre.leggauss(n) for the orders the BEM solver
// actually uses (n = 2..16 covers solve()'s default 8 and compute_*_at_point()'s
// default 12 with comfortable headroom).  Tables are reproduced to full double
// precision; values match numpy's leggauss to <1e-15 absolute.

#include <stdexcept>
#include <vector>

namespace vcem { namespace bem {

struct GaussRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

// Returns Gauss-Legendre rule of order n (number of points).
const GaussRule& gauss_legendre(int n);

}}  // namespace vcem::bem
