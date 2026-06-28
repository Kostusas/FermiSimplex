#pragma once

#include "certificate/linalg/matrix.h"

#include <cstddef>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

struct OccupationRankEstimate {
    size_t rank = 0;
    double mu_radius = 0.0;
};

size_t estimate_ordered_subset_rank(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double tol,
    double margin = 0.0
);

OccupationRankEstimate estimate_ordered_subset_rank_with_mu_radius(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double tol,
    double margin = 0.0
);

}  // namespace lineartetrahedron::simplex_certificate::detail
