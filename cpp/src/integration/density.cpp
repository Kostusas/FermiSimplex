#include "integration/density.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fermisimplex::integration_detail {
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

std::vector<std::complex<double>> lattice_phases(
    std::span<const std::int64_t> lattice_vectors,
    std::size_t lattice_vector_count,
    std::span<const double> point
) {
    std::vector<std::complex<double>> phases(lattice_vector_count);
    for (std::size_t vector_index = 0;
         vector_index < lattice_vector_count;
         ++vector_index) {
        auto phase = 0.0;
        for (std::size_t axis = 0; axis < point.size(); ++axis) {
            phase += point[axis] *
                     static_cast<double>(
                         lattice_vectors[vector_index * point.size() + axis]
                     );
        }
        phases[vector_index] = std::exp(
            std::complex<double>(0.0, 2.0 * std::numbers::pi_v<double> * phase)
        );
    }
    return phases;
}

std::complex<double> density_element(
    const Eigensystem &spectra,
    std::span<const double> band_weights,
    std::size_t row,
    std::size_t column,
    std::size_t ndof
) {
    auto value = std::complex<double>{};
    for (std::size_t band = 0; band < ndof; ++band) {
        value += band_weights[band] *
                 spectra.eigenvectors[band * ndof + row] *
                 std::conj(spectra.eigenvectors[band * ndof + column]);
    }
    return value;
}

}  // namespace

DensityMatrixRule::DensityMatrixRule(
    std::size_t ndim,
    std::size_t ndof,
    std::vector<LatticeVector> lattice_vectors
) : ndim_(ndim),
    ndof_(ndof) {
    if (ndim_ == 0 || ndof_ == 0) {
        throw std::runtime_error("DensityMatrixRule: dimensions must be positive");
    }
    if (lattice_vectors.empty()) {
        throw std::runtime_error("DensityMatrixRule: invalid lattice-vector shape");
    }
    lattice_vector_count_ = lattice_vectors.size();
    lattice_vectors_.reserve(lattice_vector_count_ * ndim_);
    for (const auto &lattice_vector : lattice_vectors) {
        if (lattice_vector.size() != ndim_) {
            throw std::runtime_error("DensityMatrixRule: invalid lattice-vector shape");
        }
        lattice_vectors_.insert(
            lattice_vectors_.end(),
            lattice_vector.begin(),
            lattice_vector.end()
        );
    }
}

DensityMatrixRule::Value DensityMatrixRule::on_simplex(
    double mu,
    const SpectralMesh &mesh,
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) const {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto vertex_count = simplex.vertex_ids.size();
    const auto &cache = mesh.eigensystems();
    auto band_weights = std::vector<double>(vertex_count * ndof_, 0.0);

    for (std::size_t band = 0; band < ndof_; ++band) {
        auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{.level = mu, .level_tolerance = mesh.tolerance()}
        );
        if (moments.kind == cut::SimplexCutKind::on_level) {
            std::fill(
                moments.barycentric_moments.begin(),
                moments.barycentric_moments.end(),
                0.5 * simplex.volume / static_cast<double>(vertex_count)
            );
        }
        for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
            band_weights[vertex * ndof_ + band] = moments.barycentric_moments[vertex];
        }
    }

    auto result = Value(lattice_vector_count_ * ndof_ * ndof_);
    for (std::size_t local_vertex = 0; local_vertex < vertex_count; ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        const auto phases = lattice_phases(
            lattice_vectors_,
            lattice_vector_count_,
            std::span<const double>(point.data(), point.size())
        );
        const auto weights = std::span<const double>(
            band_weights.data() + local_vertex * ndof_,
            ndof_
        );
        const auto &spectra = cache.get(vertex_id);

        for (std::size_t row = 0; row < ndof_; ++row) {
            for (std::size_t column = 0; column < ndof_; ++column) {
                const auto element =
                    density_element(spectra, weights, row, column, ndof_);
                for (std::size_t vector_index = 0;
                     vector_index < lattice_vector_count_;
                     ++vector_index) {
                    result[(vector_index * ndof_ + row) * ndof_ + column] +=
                        phases[vector_index] * element;
                }
            }
        }
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
