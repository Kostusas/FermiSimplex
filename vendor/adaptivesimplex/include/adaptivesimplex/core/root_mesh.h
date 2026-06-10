#pragma once

#include "adaptivesimplex/core/dyadic_vertex.h"
#include "adaptivesimplex/core/geometry.h"
#include "adaptivesimplex/core/simplex_table.h"
#include "adaptivesimplex/core/vertex_table.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adaptivesimplex::core {

namespace detail {

inline double root_simplex_volume(size_t ndim, size_t root_subcells) {
    if (ndim == 0) {
        return 1.0;
    }

    auto denominator = 1.0;
    for (auto axis = size_t{0}; axis < ndim; ++axis) {
        denominator *= static_cast<double>(root_subcells);
    }
    for (auto value = size_t{2}; value <= ndim; ++value) {
        denominator *= static_cast<double>(value);
    }
    return 1.0 / denominator;
}

inline void append_root_simplices(
    size_t ndim,
    std::uint32_t root_level,
    VertexTable &vertices,
    SimplexTable &simplices,
    std::vector<SimplexId> &active_simplices
) {
    const auto root_subcells = size_t{1} << root_level;
    const auto volume = root_simplex_volume(ndim, root_subcells);

    if (ndim == 0) {
        const auto vertex_id = vertices.get_or_add(DyadicVertex({}, 0));
        active_simplices.push_back(simplices.add({vertex_id}, volume));
        return;
    }

    auto n_offsets = size_t{1};
    for (auto axis = size_t{0}; axis < ndim; ++axis) {
        n_offsets *= root_subcells;
    }

    for (auto offset_index = size_t{0}; offset_index < n_offsets; ++offset_index) {
        std::vector<std::int64_t> base(ndim, 0);
        auto remainder = offset_index;
        for (auto axis = size_t{0}; axis < ndim; ++axis) {
            base[axis] = static_cast<std::int64_t>(remainder % root_subcells);
            remainder /= root_subcells;
        }

        std::vector<size_t> permutation(ndim);
        std::iota(permutation.begin(), permutation.end(), size_t{0});
        do {
            auto coords = base;
            std::vector<VertexId> simplex;
            simplex.reserve(ndim + 1);

            simplex.push_back(vertices.get_or_add(DyadicVertex(coords, root_level)));
            for (const auto axis : permutation) {
                ++coords[axis];
                simplex.push_back(vertices.get_or_add(DyadicVertex(coords, root_level)));
            }

            active_simplices.push_back(simplices.add(std::move(simplex), volume));
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
}

}  // namespace detail

inline Geometry root_geometry(size_t ndim, std::uint32_t root_level = 1) {
    if (root_level >= 31) {
        throw std::runtime_error("root_geometry: root_level must be in [0, 31)");
    }

    VertexTable vertices(ndim);
    SimplexTable simplices;
    std::vector<SimplexId> active_simplices;
    detail::append_root_simplices(ndim, root_level, vertices, simplices, active_simplices);
    simplices.replace_active_simplices(std::move(active_simplices));
    return Geometry(ndim, std::move(vertices), std::move(simplices));
}

}  // namespace adaptivesimplex::core
