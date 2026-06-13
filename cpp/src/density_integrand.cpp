#include "lineartetrahedron/density_integrand.h"

#include "lineartetrahedron/adaptive_estimate.h"
#include "lineartetrahedron/band_weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

extern "C" {
void LINEARTETRAHEDRON_BLAS_ZGEMM(
    const char *transa,
    const char *transb,
    const int *m,
    const int *n,
    const int *k,
    const std::complex<double> *alpha,
    const std::complex<double> *a,
    const int *lda,
    const std::complex<double> *b,
    const int *ldb,
    const std::complex<double> *beta,
    std::complex<double> *c,
    const int *ldc
);
}

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

void gather_touched_vertices(
    const core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    std::vector<core::VertexId> &vertices,
    std::unordered_map<core::VertexId, size_t> &vertex_index
) {
    vertices.clear();
    vertex_index.clear();
    vertices.reserve(simplex_ids.size() * (geometry.ndim() + 1));
    vertex_index.reserve(vertices.capacity());

    for (const auto simplex_id : simplex_ids) {
        const auto &simplex = geometry.simplices().simplex(simplex_id);
        for (const auto vertex_id : simplex.vertex_ids) {
            if (vertex_index.contains(vertex_id)) {
                continue;
            }
            const auto index = vertices.size();
            vertices.push_back(vertex_id);
            vertex_index.emplace(vertex_id, index);
        }
    }
}

void accumulate_vertex_band_weights(
    const IntegrationState &state,
    double mu,
    const core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    const std::unordered_map<core::VertexId, size_t> &vertex_index,
    std::span<double> vertex_band_weights,
    std::vector<const VertexSpectra *> &entries,
    BandWeightScratch &band_weights
) {
    const size_t ndof = state.ndof();
    for (const auto simplex_id : simplex_ids) {
        const auto &simplex = geometry.simplices().simplex(simplex_id);
        gather_vertex_spectra_into(geometry, state.cache(), simplex_id, entries);
        for (size_t band = 0; band < ndof; ++band) {
            const auto weights = band_weights_on_simplex(
                entries,
                band,
                mu,
                simplex.volume,
                state.ndim(),
                state.tol(),
                band_weights
            );
            for (
                size_t local_vertex = 0;
                local_vertex < simplex.vertex_ids.size();
                ++local_vertex
            ) {
                const double weight = weights.vertex_weights[local_vertex];
                if (weight == 0.0) {
                    continue;
                }
                const auto global_vertex = simplex.vertex_ids[local_vertex];
                const auto compact_vertex = vertex_index.at(global_vertex);
                vertex_band_weights[compact_vertex * ndof + band] += weight;
            }
        }
    }
}

bool dense_density_matrix_for_vertex(
    std::span<const std::complex<double>> eigenvectors,
    std::span<const double> weights,
    size_t ndof,
    std::vector<std::complex<double>> &weighted_eigenvectors,
    std::vector<std::complex<double>> &density_matrix
) {
    if (ndof > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("DensityIntegrand: BLAS matrix dimension exceeds LP64 range");
    }

    weighted_eigenvectors.resize(ndof * ndof);
    density_matrix.resize(ndof * ndof);
    bool has_nonzero_weight = false;
    for (size_t band = 0; band < ndof; ++band) {
        const auto weight = std::complex<double>(weights[band], 0.0);
        if (weight != std::complex<double>(0.0, 0.0)) {
            has_nonzero_weight = true;
        }
        for (size_t row = 0; row < ndof; ++row) {
            weighted_eigenvectors[band * ndof + row] =
                weight * eigenvectors[band * ndof + row];
        }
    }
    if (!has_nonzero_weight) {
        std::fill(
            density_matrix.begin(),
            density_matrix.end(),
            std::complex<double>(0.0, 0.0)
        );
        return false;
    }

    const char transa = 'N';
    const char transb = 'C';
    const int n = static_cast<int>(ndof);
    const std::complex<double> alpha = 1.0;
    const std::complex<double> beta = 0.0;
    LINEARTETRAHEDRON_BLAS_ZGEMM(
        &transa,
        &transb,
        &n,
        &n,
        &n,
        &alpha,
        weighted_eigenvectors.data(),
        &n,
        eigenvectors.data(),
        &n,
        &beta,
        density_matrix.data(),
        &n
    );
    return true;
}

void accumulate_simplices_by_vertex_band(
    const IntegrationState &state,
    double mu,
    const core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    adaptive::DenseValue<std::complex<double>> &result
) {
    std::vector<core::VertexId> touched_vertices;
    std::unordered_map<core::VertexId, size_t> vertex_index;
    gather_touched_vertices(geometry, simplex_ids, touched_vertices, vertex_index);
    if (touched_vertices.empty()) {
        return;
    }

    const size_t ndof = state.ndof();
    std::vector<double> vertex_band_weights(touched_vertices.size() * ndof, 0.0);
    std::vector<const VertexSpectra *> entries;
    entries.reserve(state.ndim() + 1);
    BandWeightScratch band_weights;
    accumulate_vertex_band_weights(
        state,
        mu,
        geometry,
        simplex_ids,
        vertex_index,
        vertex_band_weights,
        entries,
        band_weights
    );

    std::vector<std::complex<double>> phases(state.key_count());
    std::vector<std::complex<double>> weighted_eigenvectors;
    std::vector<std::complex<double>> density_matrix;
    for (
        size_t compact_vertex = 0;
        compact_vertex < touched_vertices.size();
        ++compact_vertex
    ) {
        const auto vertex_id = touched_vertices[compact_vertex];
        const auto &spectra = state.cache().get(vertex_id);
        const auto &eigenvectors = spectra.eigenvectors;
        const auto &reduced_vertex = geometry.vertices().dyadic_vertex(vertex_id);
        const auto weights = std::span<const double>(
            vertex_band_weights.data() + compact_vertex * ndof,
            ndof
        );

        scaled_phases_for_keys(
            state.keys(),
            state.key_count(),
            state.ndim(),
            reduced_vertex,
            1.0,
            phases
        );

        const auto eigenvector_view = std::span<const std::complex<double>>(
            eigenvectors.data(),
            eigenvectors.size()
        );
        if (!dense_density_matrix_for_vertex(
                eigenvector_view,
                weights,
                ndof,
                weighted_eigenvectors,
                density_matrix
            )) {
            continue;
        }
        for (const auto &group : state.component_pair_groups()) {
            const auto density = density_matrix[group.col * ndof + group.row];
            if (density == std::complex<double>(0.0, 0.0)) {
                continue;
            }
            for (const auto &contribution : group.contributions) {
                result[contribution.component_index] +=
                    phases[contribution.key_index] * density;
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

DensityIntegrand::value_type DensityIntegrand::simplex_value(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) const {
    auto value = zero_value();
    accumulate_simplices_by_vertex_band(
        state_,
        mu_,
        geometry,
        std::span<const core::SimplexId>(&simplex_id, 1),
        value
    );
    return value;
}

std::vector<adaptive::Estimate<DensityIntegrand::value_type>>
DensityIntegrand::estimate_simplices(
    core::Geometry &geometry,
    std::span<const core::SimplexId> simplex_ids,
    std::uint32_t preview_depth
) const {
    return estimate_with_preview<value_type>(
        geometry,
        simplex_ids,
        preview_depth,
        [&](core::SimplexId simplex_id) { return simplex_value(geometry, simplex_id); },
        [&]() { return zero_value(); },
        [&](const value_type &correction) { return error_norm(correction); }
    );
}

}  // namespace lineartetrahedron
