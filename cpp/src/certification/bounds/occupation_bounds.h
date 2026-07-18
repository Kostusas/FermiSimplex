#pragma once

#include "certification/linalg/matrix.h"

#include <cstddef>
#include <vector>

namespace lineartetrahedron::certification::detail {

struct OccupationRankEstimate {
    size_t rank = 0;
    double mu_radius = 0.0;
};

size_t estimate_ordered_subset_rank(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double linearization_error_bound,
    double tolerance
);

OccupationRankEstimate estimate_ordered_subset_rank_with_mu_radius(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double linearization_error_bound,
    double tolerance
);

}  // namespace lineartetrahedron::certification::detail
