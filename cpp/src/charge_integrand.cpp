#include "lineartetrahedron/charge_integrand.h"

#include "lineartetrahedron/adaptive_estimate.h"
#include "lineartetrahedron/band_weights.h"

#include <cmath>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

ChargeIntegrand::ChargeIntegrand(IntegrationState &state, double mu)
    : state_(state), mu_(mu) {}

double ChargeIntegrand::error_norm(const ChargeValue &correction) const {
    return std::abs(correction.charge);
}

VertexSpectra ChargeIntegrand::evaluate_vertex(
    core::Geometry &geometry,
    core::VertexId vertex_id
) {
    return state_.evaluate_vertex(geometry, vertex_id);
}

ChargeValue ChargeIntegrand::simplex_value(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) const {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto entries = gather_vertex_spectra(geometry, state_.cache(), simplex_id);
    ChargeValue result;

    for (size_t band = 0; band < state_.ndof(); ++band) {
        const auto weights = band_weights_on_simplex(
            entries,
            band,
            mu_,
            simplex.volume,
            state_.ndim(),
            state_.tol()
        );
        result.charge += weights.charge;
        result.derivative += weights.derivative;
    }
    return result;
}

std::vector<adaptive::Estimate<ChargeValue>> ChargeIntegrand::estimate_simplices(
    core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    std::uint32_t preview_depth
) const {
    return estimate_with_preview<ChargeValue>(
        geometry,
        simplex_ids,
        preview_depth,
        [&](core::SimplexId simplex_id) { return simplex_value(geometry, simplex_id); },
        [&]() { return zero_value(); },
        [&](const ChargeValue &correction) { return error_norm(correction); }
    );
}

}  // namespace lineartetrahedron
