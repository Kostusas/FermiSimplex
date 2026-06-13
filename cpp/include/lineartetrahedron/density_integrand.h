#pragma once

#include "lineartetrahedron/integration_state.h"

#include <adaptivesimplex/adaptive/dense_value.h>
#include <adaptivesimplex/adaptive/types.h>

#include <complex>
#include <cstdint>

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
    adaptivesimplex::adaptive::Result<
        adaptivesimplex::adaptive::DenseValue<std::complex<double>>
    > estimate_density(const adaptivesimplex::adaptive::Options &options);

private:
    IntegrationState &state_;
    double mu_ = 0.0;
};

}  // namespace lineartetrahedron
