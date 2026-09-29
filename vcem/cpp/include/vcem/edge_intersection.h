#pragma once
//
// Segment-segment intersection detection for crack-network topology
// post-processing. Direct port of the inner math used by
// fracture_utils/Ugenerator/generator.py::_find_edge_intersection
// + _detect_single_pass.
//
// Returns the same set of (edge_i, edge_j, intersection_point) tuples
// as the Python reference. OpenMP-parallel over the outer edge loop
// with thread-local accumulators; the merged result is sorted by
// (i, j) so the downstream split logic operates in the same order as
// the brute-force Python version.

#include <Eigen/Dense>
#include <vector>

namespace vcem { namespace topology {

struct EdgeIntersection {
    int    i;   // index of edge 1 in the input arrays
    int    j;   // index of edge 2  (always j > i)
    double x;   // intersection x-coordinate
    double y;   // intersection y-coordinate
};

// Detect all segment-segment intersections among the given edges.
//
// Inputs:
//   edges_p1   shape (E, 2)  start-point coordinates of each edge
//   edges_p2   shape (E, 2)  end-point   coordinates of each edge
//   edge_nodes shape (E, 2)  (node_id_1, node_id_2) per edge -- used to
//                            skip pairs that share a vertex (these are
//                            adjacent edges, not true intersections).
//
// Geometric parameters (match the Python defaults exactly):
//   endpoint_tol  : reject hits whose t,s are within this fraction of
//                   either endpoint (default 0.05 = 5%)
//   parallel_tol  : reject pairs whose 2D cross-product magnitude is
//                   below this threshold (default 1e-10)
//
// Returns: vector of EdgeIntersection records sorted by (i, j).
std::vector<EdgeIntersection> detect_segment_intersections(
    const Eigen::Ref<const Eigen::MatrixXd>& edges_p1,
    const Eigen::Ref<const Eigen::MatrixXd>& edges_p2,
    const Eigen::Ref<const Eigen::MatrixXi>& edge_nodes,
    double endpoint_tol = 0.05,
    double parallel_tol = 1.0e-10);

}}  // namespace vcem::topology
