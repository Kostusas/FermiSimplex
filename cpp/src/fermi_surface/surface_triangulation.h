#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

std::vector<std::int64_t> product_simplex_triangulation_cells(
    size_t negative_count,
    size_t positive_count
);

}  // namespace lineartetrahedron
