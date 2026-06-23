#include "lineartetrahedron/density.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

std::uint64_t component_pair_key(size_t row, size_t col, size_t ndof) {
    return static_cast<std::uint64_t>(row * ndof + col);
}

std::vector<std::complex<double>> phases_for_vertex(
    const std::vector<std::int64_t> &keys,
    size_t key_count,
    size_t ndim,
    const core::DyadicVertex &reduced_vertex
) {
    std::vector<double> k_point(ndim, 0.0);
    const auto coords = reduced_vertex.coords();
    for (size_t axis = 0; axis < ndim; ++axis) {
        const auto reduced_coord =
            std::ldexp(static_cast<double>(coords[axis]), -static_cast<int>(reduced_vertex.level()));
        k_point[axis] = 2.0 * kPi * reduced_coord;
    }

    std::vector<std::complex<double>> phases(key_count);
    for (size_t key_index = 0; key_index < key_count; ++key_index) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim; ++axis) {
            phase_arg += k_point[axis] * static_cast<double>(keys[key_index * ndim + axis]);
        }
        phases[key_index] = std::exp(std::complex<double>(0.0, phase_arg));
    }
    return phases;
}

std::complex<double> density_matrix_component(
    const VertexSpectra &spectra,
    std::span<const double> band_weights,
    size_t row,
    size_t col,
    size_t ndof
) {
    std::complex<double> result = 0.0;
    for (size_t band = 0; band < ndof; ++band) {
        result += band_weights[band] *
                  spectra.eigenvectors[band * ndof + row] *
                  std::conj(spectra.eigenvectors[band * ndof + col]);
    }
    return result;
}

}  // namespace

DensityComponents::DensityComponents(
    size_t ndim,
    size_t ndof,
    KeyArray keys,
    ComponentIndexArray component_rows,
    ComponentIndexArray component_cols,
    ComponentIndexArray component_key_indices
) : ndim_(ndim),
    ndof_(ndof) {
    if (keys.shape(1) != ndim_) {
        throw std::runtime_error("DensityComponents: density key dimension mismatch");
    }
    key_count_ = keys.shape(0);
    if (key_count_ == 0) {
        throw std::runtime_error("DensityComponents: keys must be non-empty");
    }
    keys_.assign(keys.data(), keys.data() + key_count_ * ndim_);

    const size_t count = component_rows.shape(0);
    if (component_cols.shape(0) != count || component_key_indices.shape(0) != count) {
        throw std::runtime_error("DensityComponents: selected component arrays must match");
    }
    component_count_ = count;

    std::unordered_map<std::uint64_t, size_t> pair_index;
    pair_index.reserve(std::min(count, ndof_ * ndof_));
    pairs_.reserve(std::min(count, ndof_ * ndof_));
    for (size_t index = 0; index < count; ++index) {
        const auto row = component_rows.data()[index];
        const auto col = component_cols.data()[index];
        const auto key_index = component_key_indices.data()[index];
        if (row < 0 || col < 0 || key_index < 0) {
            throw std::runtime_error("DensityComponents: selected component indices must be non-negative");
        }
        if (static_cast<size_t>(row) >= ndof_ || static_cast<size_t>(col) >= ndof_) {
            throw std::runtime_error("DensityComponents: selected row/column out of bounds");
        }
        if (static_cast<size_t>(key_index) >= key_count_) {
            throw std::runtime_error("DensityComponents: selected key index out of bounds");
        }

        const auto key = component_pair_key(static_cast<size_t>(row), static_cast<size_t>(col), ndof_);
        auto item = pair_index.find(key);
        if (item == pair_index.end()) {
            const auto new_index = pairs_.size();
            pairs_.push_back(ComponentPair{
                static_cast<size_t>(row),
                static_cast<size_t>(col),
                {},
            });
            item = pair_index.emplace(key, new_index).first;
        }
        pairs_[item->second].contributions.push_back(ComponentContribution{
            index,
            static_cast<size_t>(key_index),
        });
    }
}

DensityComponents::DensityValue DensityComponents::on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache
) const {
    auto result = DensityValue(component_count_);
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const size_t nvertices = simplex.vertex_ids.size();

    std::vector<double> vertex_band_weights(nvertices * ndof_, 0.0);
    for (size_t band = 0; band < ndof_; ++band) {
        auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{.level = mu, .level_tolerance = workspace.tol()}
        );

        if (moments.kind == cut::SimplexCutKind::on_level) {
            std::fill(
                moments.barycentric_moments.begin(),
                moments.barycentric_moments.end(),
                0.5 * simplex.volume / static_cast<double>(nvertices)
            );
        }

        for (size_t local_vertex = 0; local_vertex < nvertices; ++local_vertex) {
            vertex_band_weights[local_vertex * ndof_ + band] =
                moments.barycentric_moments[local_vertex];
        }
    }

    for (size_t local_vertex = 0; local_vertex < nvertices; ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto &spectra = cache.get(vertex_id);
        const auto phases = phases_for_vertex(
            keys_,
            key_count_,
            ndim_,
            geometry.vertices().dyadic_vertex(vertex_id)
        );
        const auto band_weights = std::span<const double>(
            vertex_band_weights.data() + local_vertex * ndof_,
            ndof_
        );

        for (const auto &pair : pairs_) {
            const auto density =
                density_matrix_component(spectra, band_weights, pair.row, pair.col, ndof_);
            if (density == std::complex<double>(0.0, 0.0)) {
                continue;
            }
            for (const auto &contribution : pair.contributions) {
                result[contribution.component_index] +=
                    phases[contribution.key_index] * density;
            }
        }
    }
    return result;
}

}  // namespace lineartetrahedron
