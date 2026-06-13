#pragma once

#include "lineartetrahedron/integration_state.h"

#include <adaptivesimplex/adaptive/dense_value.h>
#include <adaptivesimplex/adaptive/types.h>

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron {

class DensityIntegrand {
public:
    // Names required by adaptivesimplex::adaptive::run.
    using vertex_value_type = VertexSpectra;
    using value_type = adaptivesimplex::adaptive::DenseValue<std::complex<double>>;

    DensityIntegrand(IntegrationState &state, double mu);

    adaptivesimplex::core::VertexCache<VertexSpectra> &cache() { return state_.cache(); }
    adaptivesimplex::adaptive::DenseValue<std::complex<double>> zero_value() const;
    double error_norm(
        const adaptivesimplex::adaptive::DenseValue<std::complex<double>> &correction
    ) const;
    VertexSpectra evaluate_vertex(
        adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::VertexId vertex_id
    );
    std::vector<adaptivesimplex::adaptive::Estimate<value_type>> estimate_simplices(
        adaptivesimplex::core::Geometry &geometry,
        std::span<const adaptivesimplex::core::SimplexId> simplex_ids,
        std::uint32_t preview_depth
    ) const;

private:
    value_type simplex_value(
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::SimplexId simplex_id
    ) const;

    IntegrationState &state_;
    double mu_ = 0.0;
};

}  // namespace lineartetrahedron
