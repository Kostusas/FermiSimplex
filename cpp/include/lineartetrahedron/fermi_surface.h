#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

class SpectralMesh;

struct FermiSurfaceStats {
    std::uint64_t evaluations = 0;
    std::int64_t terminal_visible_simplices = 0;
    std::int64_t terminal_inconclusive_simplices = 0;
};

struct FermiSurfaceResult {
    // points is (npoints, ndim), cells is (ncells, ndim), and cell_bands is
    // (ncells); all are flattened. cell_bands contains the zero-based
    // eigenvalue-band index from which each surface cell was extracted.
    std::vector<double> points;
    std::vector<std::int64_t> cells;
    std::vector<std::int64_t> cell_bands;
    std::size_t ndim = 0;
    bool completed = true;  // False when the evaluation budget was exhausted.
    // Certifies classification and surface coverage down to the requested
    // feature size. It does not bound geometric error or certify topology.
    bool coverage_certified = true;
    FermiSurfaceStats stats;
};

FermiSurfaceResult fermi_surface(
    SpectralMesh &mesh,
    double mu,
    double min_feature_size,
    std::int64_t max_evaluations = -1,
    double curvature_bound = 0.0
);

}  // namespace lineartetrahedron
