#include "lineartetrahedron/density_integrand.h"

#include "lineartetrahedron/band_weights.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

namespace {

void scaled_phases_for_keys(
    const std::vector<std::int64_t> &keys,
    size_t key_count,
    size_t ndim,
    const core::DyadicVertex &reduced_vertex,
    double scale,
    std::span<std::complex<double>> out
) {
    std::array<double, 3> k_point{};
    const auto coords = reduced_vertex.coords();
    for (size_t axis = 0; axis < ndim; ++axis) {
        const auto reduced_coord = std::ldexp(
            static_cast<double>(coords[axis]),
            -static_cast<int>(reduced_vertex.level())
        );
        k_point[axis] = 2.0 * kPi * reduced_coord - kPi;
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

struct DensityScratch {
    adaptive::DenseValue<std::complex<double>> coarse;
    std::vector<std::complex<double>> beta;
    std::vector<const VertexSpectra *> entries;
    BandWeightScratch band_weights;

    DensityScratch(size_t component_count, size_t key_count, size_t vertex_count)
        : coarse(component_count), beta(key_count) {
        entries.reserve(vertex_count);
    }
};

void reset_value(adaptive::DenseValue<std::complex<double>> &value) {
    for (size_t index = 0; index < value.size(); ++index) {
        value[index] = {};
    }
}

void gather_vertex_spectra_into(
    const core::Geometry &geometry,
    const core::VertexCache<VertexSpectra> &cache,
    core::SimplexId simplex_id,
    std::vector<const VertexSpectra *> &entries
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    entries.clear();
    entries.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        entries.push_back(&cache.get(vertex_id));
    }
}

void accumulate_simplex_value(
    const IntegrationState &state,
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    DensityScratch &scratch,
    adaptive::DenseValue<std::complex<double>> &result
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    gather_vertex_spectra_into(geometry, state.cache(), simplex_id, scratch.entries);
    const auto &vertex_ids = simplex.vertex_ids;

    for (size_t band = 0; band < state.ndof(); ++band) {
        const auto weights = band_weights_on_simplex(
            scratch.entries,
            band,
            mu,
            simplex.volume,
            state.ndim(),
            state.tol(),
            scratch.band_weights
        );
        for (size_t vertex = 0; vertex < vertex_ids.size(); ++vertex) {
            const double weight = weights.vertex_weights[vertex];
            if (weight == 0.0) {
                continue;
            }
            const auto &reduced_vertex = geometry.vertices().dyadic_vertex(vertex_ids[vertex]);
            const auto &eigenvectors = scratch.entries[vertex]->eigenvectors;
            scaled_phases_for_keys(
                state.keys(),
                state.key_count(),
                state.ndim(),
                reduced_vertex,
                weight,
                scratch.beta
            );
            for (const auto &group : state.component_pair_groups()) {
                const std::complex<double> projector =
                    eigenvectors[band * state.ndof() + group.row] *
                    std::conj(eigenvectors[band * state.ndof() + group.col]);
                for (const auto &contribution : group.contributions) {
                    result[contribution.component_index] +=
                        scratch.beta[contribution.key_index] * projector;
                }
            }
        }
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
    DensityScratch scratch(
        state_.density_component_count(),
        state_.key_count(),
        state_.ndim() + 1
    );
    accumulate_simplex_value(state_, mu_, geometry, simplex_id, scratch, result);
    return result;
}

std::vector<adaptive::Estimate<adaptive::DenseValue<std::complex<double>>>>
DensityIntegrand::estimate_simplices(
    core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    std::uint32_t preview_depth
) const {
    std::vector<std::vector<core::SimplexId>> preview_ids(simplex_ids.size());
    for (size_t index = 0; index < simplex_ids.size(); ++index) {
        preview_ids[index] = geometry.preview_active(simplex_ids[index], preview_depth);
    }

    std::vector<adaptive::Estimate<adaptive::DenseValue<std::complex<double>>>> estimates(
        simplex_ids.size()
    );
    const auto simplex_count = static_cast<std::int64_t>(simplex_ids.size());
    const size_t component_count = state_.density_component_count();
    const size_t key_count = state_.key_count();
    const size_t vertex_count = state_.ndim() + 1;

#pragma omp parallel if(simplex_count > 1)
    {
        DensityScratch scratch(component_count, key_count, vertex_count);

#pragma omp for schedule(dynamic)
        for (std::int64_t index = 0; index < simplex_count; ++index) {
            const auto item = static_cast<size_t>(index);
            auto preview = zero_value();
            reset_value(scratch.coarse);
            accumulate_simplex_value(
                state_,
                mu_,
                geometry,
                simplex_ids[item],
                scratch,
                scratch.coarse
            );
            for (const auto preview_id : preview_ids[item]) {
                accumulate_simplex_value(state_, mu_, geometry, preview_id, scratch, preview);
            }

            auto correction = preview;
            correction -= scratch.coarse;
            const double error_indicator = error_norm(correction);
            estimates[item] =
                adaptive::Estimate<adaptive::DenseValue<std::complex<double>>>{
                    .value = std::move(preview),
                    .correction = std::move(correction),
                    .indicator = error_indicator,
                };
        }
    }

    return estimates;
}

}  // namespace lineartetrahedron
