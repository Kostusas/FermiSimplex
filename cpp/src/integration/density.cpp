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

std::vector<std::int64_t> flatten_lattice_vectors(
    std::size_t ndim,
    const std::vector<LatticeVector> &lattice_vectors
) {
    if (ndim == 0 || lattice_vectors.empty()) {
        throw std::runtime_error("DensityRule: invalid lattice-vector shape");
    }
    auto result = std::vector<std::int64_t>{};
    result.reserve(lattice_vectors.size() * ndim);
    for (const auto &lattice_vector : lattice_vectors) {
        if (lattice_vector.size() != ndim) {
            throw std::runtime_error("DensityRule: invalid lattice-vector shape");
        }
        result.insert(
            result.end(),
            lattice_vector.begin(),
            lattice_vector.end()
        );
    }
    return result;
}

struct PendingContribution {
    std::size_t pair_index = 0;
    std::size_t output_index = 0;
    std::size_t lattice_vector_index = 0;
};

}  // namespace

DensityRule::DensityRule(
    std::size_t ndim,
    std::size_t ndof,
    std::vector<LatticeVector> lattice_vectors
) : ndim_(ndim),
    ndof_(ndof),
    lattice_vector_count_(lattice_vectors.size()),
    output_size_(lattice_vector_count_ * ndof_ * ndof_),
    lattice_vectors_(flatten_lattice_vectors(ndim_, lattice_vectors)) {
    if (ndof_ == 0) {
        throw std::runtime_error("DensityRule: dimensions must be positive");
    }
    pairs_.reserve(ndof_ * ndof_);
    contributions_.reserve(output_size_);
    for (std::size_t row = 0; row < ndof_; ++row) {
        for (std::size_t column = 0; column < ndof_; ++column) {
            const auto begin = contributions_.size();
            for (std::size_t vector = 0;
                 vector < lattice_vector_count_;
                 ++vector) {
                contributions_.push_back(Contribution{
                    .output_index = (vector * ndof_ + row) * ndof_ + column,
                    .lattice_vector_index = vector,
                });
            }
            pairs_.push_back(ComponentPair{
                .row = row,
                .column = column,
                .contribution_begin = begin,
                .contribution_end = contributions_.size(),
            });
        }
    }
}

DensityRule::DensityRule(
    std::size_t ndim,
    std::size_t ndof,
    std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components
) : ndim_(ndim),
    ndof_(ndof),
    lattice_vector_count_(lattice_vectors.size()),
    output_size_(components.size()),
    lattice_vectors_(flatten_lattice_vectors(ndim_, lattice_vectors)) {
    if (ndof_ == 0) {
        throw std::runtime_error("DensityRule: dimensions must be positive");
    }
    if (components.empty()) {
        throw std::runtime_error("DensityRule: components must be non-empty");
    }

    auto pending = std::vector<PendingContribution>{};
    pending.reserve(components.size());
    for (std::size_t output = 0; output < components.size(); ++output) {
        const auto &component = components[output];
        if (
            component.lattice_vector_index >= lattice_vector_count_ ||
            component.row >= ndof_ ||
            component.column >= ndof_
        ) {
            throw std::runtime_error("DensityRule: component index out of range");
        }
        pending.push_back(PendingContribution{
            .pair_index = component.row * ndof_ + component.column,
            .output_index = output,
            .lattice_vector_index = component.lattice_vector_index,
        });
    }
    std::sort(
        pending.begin(),
        pending.end(),
        [](const auto &left, const auto &right) {
            return left.pair_index < right.pair_index;
        }
    );

    contributions_.reserve(pending.size());
    for (std::size_t begin = 0; begin < pending.size();) {
        auto end = begin + 1;
        while (
            end < pending.size() &&
            pending[end].pair_index == pending[begin].pair_index
        ) {
            ++end;
        }
        const auto contribution_begin = contributions_.size();
        for (auto index = begin; index < end; ++index) {
            contributions_.push_back(Contribution{
                .output_index = pending[index].output_index,
                .lattice_vector_index = pending[index].lattice_vector_index,
            });
        }
        pairs_.push_back(ComponentPair{
            .row = pending[begin].pair_index / ndof_,
            .column = pending[begin].pair_index % ndof_,
            .contribution_begin = contribution_begin,
            .contribution_end = contributions_.size(),
        });
        begin = end;
    }
}

DensityRule::Value DensityRule::on_simplex(
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

    auto result = Value(output_size_);
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

        for (const auto &pair : pairs_) {
            const auto element = density_element(
                spectra,
                weights,
                pair.row,
                pair.column,
                ndof_
            );
            if (element == std::complex<double>{}) {
                continue;
            }
            const auto *contribution =
                contributions_.data() + pair.contribution_begin;
            const auto *end =
                contributions_.data() + pair.contribution_end;
            for (; contribution != end; ++contribution) {
                result[contribution->output_index] +=
                    phases[contribution->lattice_vector_index] * element;
            }
        }
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
