#pragma once

#include "fermi_surface/vertex_evaluation.h"

#include <cstdint>
#include <set>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {

struct SimplexClassification {
    std::vector<core::SimplexId> refine;
    std::vector<core::SimplexId> terminal_surface;
    std::int64_t visible_gapless_terminal = 0;
    std::int64_t inconclusive_terminal = 0;
    std::int64_t unresolved = 0;
};

SimplexClassification classify_frontier(
    const core::Geometry &geometry,
    const SpectraCache &cache,
    const std::vector<core::SimplexId> &frontier,
    double mu,
    double min_feature_size,
    double margin,
    double tol
);

std::set<core::SimplexId> simplex_set(const std::vector<core::SimplexId> &simplex_ids);

std::vector<core::SimplexId> next_frontier(
    const core::Geometry &geometry,
    const std::set<core::SimplexId> &previous_active,
    const std::vector<core::SimplexId> &requested_refinement,
    const std::vector<core::SimplexId> &refined
);

}  // namespace lineartetrahedron::fermi_surface_detail
