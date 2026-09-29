#pragma once
//
// Edge-dislocation fundamental solutions in 2D elastostatics. Direct
// translation of fracture_utils/Usolver/solve_kernels.py
// {edge_dislocation_u, stress_edge_dislocation}.
//
// Convention: differential Burgers vector dB = (dBx, dBy) at the source;
// (dx, dy) = field - source. Plane-strain by default; plane-stress is the
// usual nu -> nu/(1+nu) substitution.

namespace vcem { namespace crack {

inline double nu_eff(double nu, bool plane_stress) {
    return plane_stress ? (nu / (1.0 + nu)) : nu;
}

// Displacement (ux, uy) at the field point from a (dBx, dBy) edge dislocation.
void edge_dislocation_u(double dx, double dy,
                        double dBx, double dBy,
                        double nu, bool plane_stress,
                        double& ux, double& uy);

// Stress (sxx, syy, sxy) at the field point from a (dBx, dBy) edge
// dislocation. mu is the shear modulus.
void edge_dislocation_stress(double dx, double dy,
                             double dBx, double dBy,
                             double mu, double nu, bool plane_stress,
                             double& sxx, double& syy, double& sxy);

}}  // namespace vcem::crack
