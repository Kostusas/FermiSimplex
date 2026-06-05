#include "lineartetrahedron/density_integrand.h"

#include "lineartetrahedron/adaptive_estimate.h"
#include "lineartetrahedron/band_weights.h"

#include <cmath>
#include <complex>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

namespace {

std::complex<double> phase_for_key(
    const std::vector<std::int64_t> &keys,
    size_t key_index,
    size_t ndim,
    const std::vector<double> &reduced_point
) {
    double phase_arg = 0.0;
    for (size_t axis = 0; axis < ndim; ++axis) {
        phase_arg +=
            (2.0 * kPi * reduced_point[axis] - kPi) *
            static_cast<double>(keys[key_index * ndim + axis]);
    }
    return std::exp(std::complex<double>(0.0, phase_arg));
}

}  // namespace

DensityIntegrand::DensityIntegrand(IntegrationState &state, double mu)
    : state_(state), mu_(mu) {}

adaptive::DenseValue<std::complex<double>> DensityIntegrand::zero_value() const {
    return adaptive::DenseValue<std::complex<double>>(state_.density_component_count());
}

double DensityIntegrand::error_norm(
    const adaptive::DenseValue<std::complex<double>> &correction
) const {
    return correction.max_abs();
}

VertexSpectra DensityIntegrand::evaluate_vertex(
    core::Geometry &geometry,
    core::VertexId vertex_id
) {
    return state_.evaluate_vertex(geometry, vertex_id);
}

adaptive::DenseValue<std::complex<double>> DensityIntegrand::simplex_value(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) const {
    auto result = zero_value();
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto entries = gather_vertex_spectra(geometry, state_.cache(), simplex_id);
    const auto &vertex_ids = simplex.vertex_ids;

    for (size_t band = 0; band < state_.ndof(); ++band) {
        const auto weights = band_weights_on_simplex(
            entries,
            band,
            mu_,
            simplex.volume,
            state_.ndim(),
            state_.tol()
        );
        for (size_t vertex = 0; vertex < vertex_ids.size(); ++vertex) {
            const double weight = weights.vertex_weights[vertex];
            if (weight == 0.0) {
                continue;
            }
            const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_ids[vertex]).to_point();
            const auto &eigenvectors = entries[vertex]->eigenvectors;
            for (size_t comp = 0; comp < state_.components().size(); ++comp) {
                const auto &component = state_.components()[comp];
                const std::complex<double> projector =
                    eigenvectors[band * state_.ndof() + component.row] *
                    std::conj(eigenvectors[band * state_.ndof() + component.col]);
                result[comp] +=
                    weight *
                    phase_for_key(state_.keys(), component.key_index, state_.ndim(), reduced_point) *
                    projector;
            }
        }
    }
    return result;
}

std::vector<adaptive::Estimate<adaptive::DenseValue<std::complex<double>>>>
DensityIntegrand::estimate_simplices(
    core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    std::uint32_t preview_depth
) const {
    return estimate_with_preview<adaptive::DenseValue<std::complex<double>>>(
        geometry,
        simplex_ids,
        preview_depth,
        [&](core::SimplexId simplex_id) { return simplex_value(geometry, simplex_id); },
        [&]() { return zero_value(); },
        [&](const adaptive::DenseValue<std::complex<double>> &correction) {
            return error_norm(correction);
        }
    );
}

}  // namespace lineartetrahedron
