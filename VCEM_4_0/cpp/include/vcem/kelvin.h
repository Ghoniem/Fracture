#pragma once
//
// Kelvin fundamental solutions and their analytic derivatives for 2D
// elastostatics. Direct translation of fracture_utils/Ubem/bem_solver.py
// {kelvin_U, kelvin_U_self_panel, kelvin_T, kelvin_dU_dfield, kelvin_dT_dfield}.

#include <Eigen/Dense>

#include "vcem/segment.h"

namespace vcem { namespace bem {

// All five kernels return Eigen::Matrix2d (the (i,k) index convention from the
// Python source: row i, column k). Pairs return them as two separate matrices
// for clarity at call sites.

Eigen::Matrix2d kelvin_U(double xf, double yf,
                         double xs, double ys,
                         double E, double nu, bool plane_strain,
                         double r_floor = 1e-16);

Eigen::Matrix2d kelvin_U_self_panel(const Segment& seg,
                                    double E, double nu, bool plane_strain);

Eigen::Matrix2d kelvin_T(double xf, double yf,
                         double xs, double ys,
                         double nx, double ny,
                         double E, double nu, bool plane_strain,
                         double r_floor = 1e-16);

void kelvin_dU_dfield(double xf, double yf,
                      double xs, double ys,
                      double E, double nu, bool plane_strain,
                      Eigen::Matrix2d& dUdx, Eigen::Matrix2d& dUdy,
                      double r_floor = 1e-16);

void kelvin_dT_dfield(double xf, double yf,
                      double xs, double ys,
                      double nx, double ny,
                      double E, double nu, bool plane_strain,
                      Eigen::Matrix2d& dTdx, Eigen::Matrix2d& dTdy,
                      double r_floor = 1e-16);

}}  // namespace vcem::bem
