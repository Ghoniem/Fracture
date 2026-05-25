#include "vcem/quadrature.h"

#include <cmath>
#include <map>
#include <mutex>

namespace vcem { namespace bem {

namespace {

// Evaluate P_n(x) and dP_n/dx using the 3-term recurrence
//   P_{k+1}(x) = ((2k+1) x P_k(x) - k P_{k-1}(x)) / (k+1)
// and the well-known derivative identity
//   (1 - x^2) P_n'(x) = n (P_{n-1}(x) - x P_n(x)).
void legendre_p_and_dp(int n, double x, double& Pn, double& dPn) {
    double Pkm1 = 1.0;       // P_0
    double Pk   = x;         // P_1
    for (int k = 1; k < n; ++k) {
        double Pkp1 = ((2.0 * k + 1.0) * x * Pk - k * Pkm1) / (k + 1);
        Pkm1 = Pk;
        Pk   = Pkp1;
    }
    Pn  = Pk;
    // (1 - x^2) P_n' = n (P_{n-1} - x P_n)
    dPn = n * (Pkm1 - x * Pn) / (1.0 - x * x);
}

GaussRule compute_rule(int n) {
    if (n < 1 || n > 64) {
        throw std::invalid_argument("gauss_legendre: n must be in [1, 64]");
    }

    GaussRule rule;
    rule.nodes.resize(n);
    rule.weights.resize(n);

    // Roots are symmetric about 0; compute only the upper half.
    const int half = (n + 1) / 2;
    const double pi = 3.141592653589793238462643383279502884;

    for (int i = 0; i < half; ++i) {
        // Chebyshev-style initial guess for the (i+1)-th root from the top.
        double x = std::cos(pi * (i + 0.75) / (n + 0.5));

        // Newton iteration on P_n(x) = 0.
        double Pn = 0.0, dPn = 0.0;
        for (int it = 0; it < 100; ++it) {
            legendre_p_and_dp(n, x, Pn, dPn);
            double dx = Pn / dPn;
            x -= dx;
            if (std::abs(dx) < 1.0e-15) break;
        }

        // Weight: w = 2 / ((1 - x^2) (P_n'(x))^2)
        double w = 2.0 / ((1.0 - x * x) * dPn * dPn);

        // Place symmetric pair: smallest root first (negative x), then largest.
        // numpy returns nodes sorted ascending — mirror that order.
        int idx_lo = n - 1 - i;
        int idx_hi = i;
        rule.nodes[idx_hi]   =  x;
        rule.nodes[idx_lo]   = -x;
        rule.weights[idx_hi] =  w;
        rule.weights[idx_lo] =  w;
    }

    // numpy's leggauss returns nodes ascending in x. Our loop placed
    // negative roots at the high indices; flip so that nodes[0] is most
    // negative — matches Python ordering exactly.
    for (int i = 0; i < n / 2; ++i) {
        std::swap(rule.nodes[i],   rule.nodes[n - 1 - i]);
        std::swap(rule.weights[i], rule.weights[n - 1 - i]);
    }

    return rule;
}

std::map<int, GaussRule>& cache() {
    static std::map<int, GaussRule> c;
    return c;
}

std::mutex& cache_mutex() {
    static std::mutex m;
    return m;
}

}  // namespace

const GaussRule& gauss_legendre(int n) {
    std::lock_guard<std::mutex> lk(cache_mutex());
    auto& c = cache();
    auto it = c.find(n);
    if (it != c.end()) return it->second;
    auto [ins, _] = c.emplace(n, compute_rule(n));
    return ins->second;
}

}}  // namespace vcem::bem
