#pragma once
//
// Boundary element data — direct translation of bem_solver.py Segment dataclass.
// Outward normal convention assumes the boundary is discretized CCW.

#include <cmath>

namespace vcem { namespace bem {

struct Segment {
    double x1, y1, x2, y2;
    bool   is_traction;   // true => traction prescribed (u unknown)
                          // false => displacement prescribed (t unknown)
    double bc_x, bc_y;

    double dx() const { return x2 - x1; }
    double dy() const { return y2 - y1; }
    double length() const { return std::hypot(dx(), dy()); }
    double xm() const { return 0.5 * (x1 + x2); }
    double ym() const { return 0.5 * (y1 + y2); }
    double tx() const { return dx() / length(); }
    double ty() const { return dy() / length(); }
    // For a CCW closed boundary, (ty, -tx) is the outward normal.
    double nx() const { return  ty(); }
    double ny() const { return -tx(); }
};

// s ∈ [-1, 1] -> (x(s), y(s), jac)  with jac = L/2.
inline void map_to_segment(const Segment& seg, double s,
                           double& x, double& y, double& jac) {
    x   = 0.5 * (1.0 - s) * seg.x1 + 0.5 * (1.0 + s) * seg.x2;
    y   = 0.5 * (1.0 - s) * seg.y1 + 0.5 * (1.0 + s) * seg.y2;
    jac = 0.5 * seg.length();
}

}}  // namespace vcem::bem
