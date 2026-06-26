#pragma once

#include "certificate/linalg/matrix.h"

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

Complex hermitian_entry(
    const std::vector<Complex> &matrix,
    size_t size,
    size_t row,
    size_t column
);

double gershgorin_row_lower_bound(
    const std::vector<Complex> &matrix,
    size_t size,
    size_t row,
    size_t active_size
);

std::vector<double> frame_gram_row_bounds(
    std::span<const Complex> rotation,
    size_t size,
    size_t opposite_size
);

double occupation_block_mu_radius(
    const std::vector<Complex> &occupation_block,
    std::span<const double> gram_row_bounds,
    size_t size,
    double tol
);

}  // namespace lineartetrahedron::simplex_certificate::detail
