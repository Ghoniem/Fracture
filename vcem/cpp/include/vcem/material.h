#pragma once
//
// 2D elastostatics material helpers — direct translations of
// fracture_utils/Ubem/bem_solver.py {kappa_from_nu, shear_modulus,
// lame_lambda, _material}.

namespace vcem { namespace bem {

inline double kappa_from_nu(double nu, bool plane_strain) {
    return plane_strain ? (3.0 - 4.0 * nu) : ((3.0 - nu) / (1.0 + nu));
}

inline double shear_modulus(double E, double nu) {
    return E / (2.0 * (1.0 + nu));
}

inline double lame_lambda(double E, double nu, bool plane_strain) {
    if (plane_strain) {
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    }
    return E * nu / (1.0 - nu * nu);
}

// Mirrors Python _material(): returns (lambda, mu, kappa).
struct MaterialParams {
    double lam;
    double mu;
    double kappa;
};

inline MaterialParams material_params(double E, double nu, bool plane_strain) {
    MaterialParams m;
    m.mu = shear_modulus(E, nu);
    if (plane_strain) {
        m.lam   = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        m.kappa = 3.0 - 4.0 * nu;
    } else {
        m.lam   = 2.0 * m.mu * nu / (1.0 - nu);
        m.kappa = (3.0 - nu) / (1.0 + nu);
    }
    return m;
}

}}  // namespace vcem::bem
