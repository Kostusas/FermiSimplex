#pragma once

#include <adaptivesimplex/core/geometry.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace lineartetrahedron {

namespace core = adaptivesimplex::core;

inline double simplex_diameter(
    const core::Geometry &geometry,
    const core::Simplex &simplex
) {
    auto max_squared = 0.0L;
    for (std::size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        const auto &left_vertex =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[left]);
        for (std::size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            const auto &right_vertex =
                geometry.vertices().dyadic_vertex(simplex.vertex_ids[right]);
            max_squared = std::max(
                max_squared,
                left_vertex.squared_distance_to(right_vertex)
            );
        }
    }
    return static_cast<double>(std::sqrt(max_squared));
}

inline double simplex_diameter(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    return simplex_diameter(geometry, geometry.simplices().simplex(simplex_id));
}

inline double symmetric_linearization_error_bound(
    double curvature_bound,
    double diameter
) {
    return 0.5 * curvature_bound * diameter * diameter;
}

}  // namespace lineartetrahedron
