#pragma once

#include "lineartetrahedron/integration_state.h"

#include <adaptivesimplex/adaptive/types.h>

#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron {

class ChargeIntegrand {
public:
    // Names required by adaptivesimplex::adaptive::run.
    using vertex_value_type = VertexSpectra;
    using value_type = ChargeValue;

    ChargeIntegrand(IntegrationState &state, double mu);

    adaptivesimplex::core::VertexCache<VertexSpectra> &cache() { return state_.cache(); }
    ChargeValue zero_value() const { return {}; }
    double error_norm(const ChargeValue &correction) const;
    VertexSpectra evaluate_vertex(
        adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::VertexId vertex_id
    );
    std::vector<adaptivesimplex::adaptive::Estimate<ChargeValue>> estimate_simplices(
        adaptivesimplex::core::Geometry &geometry,
        std::span<const adaptivesimplex::core::SimplexId> simplex_ids,
        std::uint32_t preview_depth
    ) const;

private:
    ChargeValue simplex_value(
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::SimplexId simplex_id
    ) const;

    IntegrationState &state_;
    double mu_ = 0.0;
};

}  // namespace lineartetrahedron
