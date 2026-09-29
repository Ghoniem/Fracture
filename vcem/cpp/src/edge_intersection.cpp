#include "vcem/edge_intersection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(VCEM_HAVE_OPENMP)
#include <omp.h>
#endif

namespace vcem { namespace topology {

std::vector<EdgeIntersection> detect_segment_intersections(
    const Eigen::Ref<const Eigen::MatrixXd>& edges_p1,
    const Eigen::Ref<const Eigen::MatrixXd>& edges_p2,
    const Eigen::Ref<const Eigen::MatrixXi>& edge_nodes,
    double endpoint_tol,
    double parallel_tol)
{
    const int E = static_cast<int>(edges_p1.rows());
    if (edges_p2.rows() != E || edge_nodes.rows() != E) {
        throw std::invalid_argument(
            "detect_segment_intersections: edges_p1, edges_p2 and edge_nodes "
            "must have the same number of rows");
    }
    if (edges_p1.cols() != 2 || edges_p2.cols() != 2 || edge_nodes.cols() != 2) {
        throw std::invalid_argument(
            "detect_segment_intersections: arrays must have 2 columns");
    }

    const double tol  = endpoint_tol;
    const double ptol = std::abs(parallel_tol);
    const double t_lo = tol;
    const double t_hi = 1.0 - tol;

    // One thread-local result vector per OpenMP thread to avoid mutex
    // contention on the result buffer.
#if defined(VCEM_HAVE_OPENMP)
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<std::vector<EdgeIntersection>> per_thread(max_threads);

#if defined(VCEM_HAVE_OPENMP)
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::vector<EdgeIntersection>& local = per_thread[tid];

        #pragma omp for schedule(dynamic, 64)
#else
    {
        std::vector<EdgeIntersection>& local = per_thread[0];
#endif
        for (int i = 0; i < E; ++i) {
            const double p1x = edges_p1(i, 0);
            const double p1y = edges_p1(i, 1);
            const double p2x = edges_p2(i, 0);
            const double p2y = edges_p2(i, 1);
            const int    a1  = edge_nodes(i, 0);
            const int    a2  = edge_nodes(i, 1);
            const double d1x = p2x - p1x;
            const double d1y = p2y - p1y;

            for (int j = i + 1; j < E; ++j) {
                const int b1 = edge_nodes(j, 0);
                const int b2 = edge_nodes(j, 1);
                // Skip pairs that share a vertex (adjacent edges -> not a
                // true intersection). Matches the Python `if edge1[0] in
                // edge2 or edge1[1] in edge2` short-circuit.
                if (a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2) continue;

                const double p3x = edges_p1(j, 0);
                const double p3y = edges_p1(j, 1);
                const double p4x = edges_p2(j, 0);
                const double p4y = edges_p2(j, 1);
                const double d2x = p4x - p3x;
                const double d2y = p4y - p3y;

                const double cross_d1_d2 = d1x * d2y - d1y * d2x;
                if (std::abs(cross_d1_d2) < ptol) continue;   // parallel

                const double d3x = p3x - p1x;
                const double d3y = p3y - p1y;
                const double inv = 1.0 / cross_d1_d2;
                const double t   = (d3x * d2y - d3y * d2x) * inv;
                const double s   = (d3x * d1y - d3y * d1x) * inv;

                if (t > t_lo && t < t_hi && s > t_lo && s < t_hi) {
                    EdgeIntersection rec;
                    rec.i = i;
                    rec.j = j;
                    rec.x = p1x + t * d1x;
                    rec.y = p1y + t * d1y;
                    local.push_back(rec);
                }
            }
        }
    }

    // Merge per-thread buffers, then sort by (i, j) so the downstream
    // Python split logic matches the Python brute-force ordering.
    std::size_t total = 0;
    for (const auto& v : per_thread) total += v.size();
    std::vector<EdgeIntersection> out;
    out.reserve(total);
    for (const auto& v : per_thread) {
        out.insert(out.end(), v.begin(), v.end());
    }
    std::sort(out.begin(), out.end(),
              [](const EdgeIntersection& a, const EdgeIntersection& b) {
                  if (a.i != b.i) return a.i < b.i;
                  return a.j < b.j;
              });
    return out;
}

}}  // namespace vcem::topology
