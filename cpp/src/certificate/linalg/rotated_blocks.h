#pragma once

#include "certificate/linalg/matrix.h"

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

std::vector<Complex> rotated_block(
    std::span<const Complex> base_block,
    double base_scale,
    std::span<const Complex> opposite_block,
    std::span<const Complex> coupling,
    std::span<const Complex> rotation,
    size_t size,
    size_t opposite_size,
    double cross_scale,
    double quadratic_scale
);

void subtract_frame_margin(
    std::vector<Complex> &block,
    std::span<const Complex> rotation,
    size_t size,
    size_t opposite_size,
    double margin
);

}  // namespace lineartetrahedron::simplex_certificate::detail
