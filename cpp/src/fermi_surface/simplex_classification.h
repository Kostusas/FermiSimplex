#pragma once

#include <lineartetrahedron/spectral_mesh.h>

#include <cstdint>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {

namespace core = adaptivesimplex::core;

struct SimplexClassification {
    std::vector<core::SimplexId> refine;
    std::vector<core::SimplexId> terminal_surface;
    std::int64_t terminal_visible = 0;
    std::int64_t terminal_inconclusive = 0;
};

SimplexClassification classify_frontier(
    const SpectralMesh &mesh,
    const std::vector<core::SimplexId> &frontier,
    double mu,
    double min_feature_size,
    double curvature_bound
);

}  // namespace lineartetrahedron::fermi_surface_detail
