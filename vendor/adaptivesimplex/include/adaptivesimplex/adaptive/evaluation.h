#pragma once

#include "adaptivesimplex/core/geometry.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adaptivesimplex::adaptive {

template <class Integrand>
void evaluate_vertices(
    core::Geometry &geometry,
    Integrand &integrand,
    std::span<const core::SimplexId> simplex_ids,
    std::uint32_t preview_depth,
    std::int64_t &work
) {
    const auto missing_vertices =
        geometry.missing_vertices(simplex_ids, integrand.cache(), preview_depth);
    if (missing_vertices.empty()) {
        return;
    }

    using VertexValue = typename Integrand::vertex_value_type;
    std::vector<VertexValue> values(missing_vertices.size());

#pragma omp parallel for
    for (
        auto index = std::int64_t{0};
        index < static_cast<std::int64_t>(missing_vertices.size());
        ++index
    ) {
        const auto item = static_cast<size_t>(index);
        values[item] = integrand.evaluate_vertex(geometry, missing_vertices[item]);
    }

    integrand.cache().insert_many(missing_vertices, std::move(values));
    work += static_cast<std::int64_t>(missing_vertices.size());
}

template <class Integrand>
auto estimate_simplices(
    core::Geometry &geometry,
    Integrand &integrand,
    const std::vector<core::SimplexId> &simplex_ids,
    std::uint32_t preview_depth,
    std::int64_t &work
) {
    evaluate_vertices(
        geometry,
        integrand,
        std::span<const core::SimplexId>(simplex_ids.data(), simplex_ids.size()),
        preview_depth,
        work
    );

    auto estimates = integrand.estimate_simplices(
        geometry,
        std::span<const core::SimplexId>(simplex_ids.data(), simplex_ids.size()),
        preview_depth
    );
    if (estimates.size() != simplex_ids.size()) {
        throw std::runtime_error("Adaptive loop: estimate count mismatch");
    }
    return estimates;
}

}  // namespace adaptivesimplex::adaptive
