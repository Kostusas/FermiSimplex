#include "lineartetrahedron/density_integrand.h"

#include "lineartetrahedron/adaptive_estimate.h"
#include "lineartetrahedron/band_weights.h"

#include <array>
#include <cmath>
#include <complex>
#include <span>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

namespace {

void scaled_phases_for_keys(
    const std::vector<std::int64_t> &keys,
    size_t key_count,
    size_t ndim,
    const std::vector<double> &reduced_point,
    double scale,
    std::span<std::complex<double>> out
) {
    std::array<double, 3> k_point{};
    for (size_t axis = 0; axis < ndim; ++axis) {
        k_point[axis] = 2.0 * kPi * reduced_point[axis] - kPi;
    }
    for (size_t key_index = 0; key_index < key_count; ++key_index) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim; ++axis) {
            phase_arg +=
                k_point[axis] *
                static_cast<double>(keys[key_index * ndim + axis]);
        }
        out[key_index] = scale * std::exp(std::complex<double>(0.0, phase_arg));
    }
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
    std::vector<std::complex<double>> beta(state_.key_count());

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
            scaled_phases_for_keys(
                state_.keys(),
                state_.key_count(),
                state_.ndim(),
                reduced_point,
                weight,
                beta
            );
            for (const auto &group : state_.component_pair_groups()) {
                const std::complex<double> projector =
                    eigenvectors[band * state_.ndof() + group.row] *
                    std::conj(eigenvectors[band * state_.ndof() + group.col]);
                for (const auto &contribution : group.contributions) {
                    result[contribution.component_index] +=
                        beta[contribution.key_index] * projector;
                }
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
