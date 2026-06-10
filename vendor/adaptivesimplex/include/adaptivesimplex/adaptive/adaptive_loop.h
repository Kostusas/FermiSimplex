#pragma once

#include "adaptivesimplex/adaptive/evaluation.h"
#include "adaptivesimplex/adaptive/refinement_queue.h"
#include "adaptivesimplex/adaptive/types.h"
#include "adaptivesimplex/core/geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace adaptivesimplex::adaptive {

template <class Integrand>
auto run(core::Geometry &geometry, Integrand &integrand, Options options) {
    using Value = typename Integrand::value_type;

    const auto preview_depth = std::max<std::uint32_t>(options.preview_depth, 1);
    auto remaining = options.max_refinements;
    auto queue = RefinementQueue{};
    auto result = Result<Value>{};
    result.value = integrand.zero_value();
    result.correction = integrand.zero_value();

    const auto active = geometry.simplices().active_simplices();
    const auto active_simplices = std::vector<core::SimplexId>(active.begin(), active.end());
    for (size_t index = 0; const auto &estimate :
         estimate_simplices(geometry, integrand, active_simplices, preview_depth, result.work)) {
        result.value += estimate.value;
        result.correction += estimate.correction;
        queue.push(active_simplices[index], estimate.indicator);
        ++index;
    }

    while (true) {
        result.error_scalar = integrand.error_norm(result.correction);
        if (result.error_scalar <= options.target_error) {
            result.converged = true;
            return result;
        }

        if (remaining == 0) {
            throw std::runtime_error("Adaptive simplex loop did not converge");
        }

        const auto parents = queue.select_for_reduction(
            result.error_scalar - options.target_error,
            remaining,
            options.min_refinement_batch_size,
            options.max_refinement_batch_size
        );
        if (parents.empty()) {
            throw std::runtime_error("Adaptive simplex loop did not converge");
        }

        for (const auto &estimate :
             estimate_simplices(geometry, integrand, parents, preview_depth, result.work)) {
            result.correction -= estimate.correction;
        }

        const auto children = geometry.refine_active(parents, preview_depth);
        for (size_t index = 0; const auto &estimate :
             estimate_simplices(geometry, integrand, children, preview_depth, result.work)) {
            result.value += estimate.correction;
            result.correction += estimate.correction;
            queue.push(children[index], estimate.indicator);
            ++index;
        }

        const auto refined_count = static_cast<std::int64_t>(parents.size());
        result.refinements += refined_count;
        if (remaining > 0) {
            remaining -= refined_count;
            if (remaining < 0) {
                throw std::runtime_error("Adaptive simplex loop did not converge");
            }
        }
    }
}

}  // namespace adaptivesimplex::adaptive
