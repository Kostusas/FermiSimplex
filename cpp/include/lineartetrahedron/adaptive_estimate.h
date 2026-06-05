#pragma once

#include <adaptivesimplex/adaptive/types.h>
#include <adaptivesimplex/core/geometry.h>

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron {

template <class Value, class SimplexValue, class ZeroValue, class Indicator>
std::vector<adaptivesimplex::adaptive::Estimate<Value>> estimate_with_preview(
    adaptivesimplex::core::Geometry &geometry,
    std::span<const adaptivesimplex::core::SimplexId> simplex_ids,
    std::uint32_t preview_depth,
    SimplexValue &&simplex_value,
    ZeroValue &&zero_value,
    Indicator &&indicator
) {
    std::vector<adaptivesimplex::adaptive::Estimate<Value>> estimates;
    estimates.reserve(simplex_ids.size());
    for (const auto simplex_id : simplex_ids) {
        const auto coarse = simplex_value(simplex_id);
        auto preview = zero_value();
        for (const auto preview_id : geometry.preview_active(simplex_id, preview_depth)) {
            preview += simplex_value(preview_id);
        }
        auto correction = preview;
        correction -= coarse;
        const double error_indicator = indicator(correction);
        estimates.push_back(adaptivesimplex::adaptive::Estimate<Value>{
            .value = std::move(preview),
            .correction = std::move(correction),
            .indicator = error_indicator,
        });
    }
    return estimates;
}

}  // namespace lineartetrahedron
