#pragma once

#include "lineartetrahedron/linalg.h"
#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron::signed_inertia {

namespace core = adaptivesimplex::core;

using Complex = std::complex<double>;
using SignedInertiaOverlapCache =
    std::map<std::pair<core::VertexId, core::VertexId>, std::vector<Complex>>;

constexpr double kSignBlockMargin = 1e-4;
constexpr double kSignBlockLanczosUncertainty = 5e-5;

struct SignedInertiaDecision {
    bool certified_safe = false;
    bool visible_cut = false;
};

inline size_t column_major_index(size_t row, size_t column, size_t rows) {
    return row + column * rows;
}

inline double signed_eigenvalue(const VertexSpectra &spectra, size_t band, double mu) {
    return spectra.eigenvalues[band] - mu;
}

inline const std::vector<Complex> &overlap_matrix(
    const VertexSpectra &anchor,
    const VertexSpectra &target,
    core::VertexId anchor_id,
    core::VertexId target_id,
    size_t ndof,
    SignedInertiaOverlapCache &cache
) {
    const auto key = std::make_pair(anchor_id, target_id);
    auto [it, inserted] = cache.emplace(key, std::vector<Complex>{});
    if (!inserted) {
        return it->second;
    }

    auto &overlap = it->second;
    overlap.assign(ndof * ndof, Complex{0.0, 0.0});
    for (size_t target_column = 0; target_column < ndof; ++target_column) {
        for (size_t anchor_column = 0; anchor_column < ndof; ++anchor_column) {
            auto value = Complex{0.0, 0.0};
            for (size_t row = 0; row < ndof; ++row) {
                value +=
                    std::conj(anchor.eigenvectors[column_major_index(row, anchor_column, ndof)]) *
                    target.eigenvectors[column_major_index(row, target_column, ndof)];
            }
            overlap[column_major_index(anchor_column, target_column, ndof)] = value;
        }
    }
    return overlap;
}

inline std::vector<Complex> build_sign_block(
    const std::vector<Complex> &overlap,
    const VertexSpectra &target,
    std::span<const size_t> anchor_indices,
    std::span<const double> inverse_sqrt_anchor_gaps,
    double mu,
    double sign
) {
    const auto ndof = target.eigenvalues.size();
    const auto block_size = anchor_indices.size();
    std::vector<Complex> block(block_size * block_size, Complex{0.0, 0.0});

    for (size_t column = 0; column < block_size; ++column) {
        const auto anchor_column = anchor_indices[column];
        for (size_t row = 0; row < block_size; ++row) {
            const auto anchor_row = anchor_indices[row];
            auto value = Complex{0.0, 0.0};
            for (size_t target_band = 0; target_band < ndof; ++target_band) {
                const auto d = signed_eigenvalue(target, target_band, mu);
                value +=
                    overlap[column_major_index(anchor_row, target_band, ndof)] *
                    d *
                    std::conj(overlap[column_major_index(anchor_column, target_band, ndof)]);
            }
            value *= sign * inverse_sqrt_anchor_gaps[row] * inverse_sqrt_anchor_gaps[column];
            block[column_major_index(row, column, block_size)] = value;
        }
    }
    return block;
}

inline bool sign_block_passes(
    const std::vector<Complex> &overlap,
    const VertexSpectra &target,
    std::span<const size_t> anchor_indices,
    std::span<const double> inverse_sqrt_anchor_gaps,
    double mu,
    double sign
) {
    const auto block_size = anchor_indices.size();
    if (block_size == 0) {
        return true;
    }

    const auto block = build_sign_block(
        overlap,
        target,
        anchor_indices,
        inverse_sqrt_anchor_gaps,
        mu,
        sign
    );
    return hermitian_min_eigenvalue_lanczos(
        std::span<const Complex>(block.data(), block.size()),
        block_size,
        kSignBlockLanczosUncertainty
    ) > kSignBlockMargin;
}

inline SignedInertiaDecision classify_linear_simplex(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    SignedInertiaOverlapCache &overlap_cache,
    double tol
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    if (simplex.vertex_ids.empty()) {
        return {};
    }

    const auto ndof = cache.get(simplex.vertex_ids.front()).eigenvalues.size();
    auto reference_occupation = size_t{0};
    auto has_reference_occupation = false;
    auto best_anchor_local = size_t{0};
    auto best_anchor_gap = -std::numeric_limits<double>::infinity();

    for (size_t local_vertex = 0; local_vertex < simplex.vertex_ids.size(); ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto &spectra = cache.get(vertex_id);
        auto occupation = size_t{0};
        auto min_gap = std::numeric_limits<double>::infinity();

        for (size_t band = 0; band < ndof; ++band) {
            const auto d = signed_eigenvalue(spectra, band, mu);
            const auto gap = std::abs(d);
            if (gap <= tol) {
                return SignedInertiaDecision{.visible_cut = true};
            }
            if (d < -tol) {
                ++occupation;
            }
            min_gap = std::min(min_gap, gap);
        }

        if (!has_reference_occupation) {
            reference_occupation = occupation;
            has_reference_occupation = true;
        } else if (occupation != reference_occupation) {
            return SignedInertiaDecision{.visible_cut = true};
        }

        if (min_gap > best_anchor_gap) {
            best_anchor_gap = min_gap;
            best_anchor_local = local_vertex;
        }
    }

    if (!(best_anchor_gap > tol) || !std::isfinite(best_anchor_gap)) {
        return {};
    }

    const auto anchor_id = simplex.vertex_ids[best_anchor_local];
    const auto &anchor = cache.get(anchor_id);
    std::vector<size_t> positive_indices;
    std::vector<size_t> negative_indices;
    std::vector<double> positive_inverse_sqrt;
    std::vector<double> negative_inverse_sqrt;
    positive_indices.reserve(ndof);
    negative_indices.reserve(ndof);
    positive_inverse_sqrt.reserve(ndof);
    negative_inverse_sqrt.reserve(ndof);

    for (size_t band = 0; band < ndof; ++band) {
        const auto d = signed_eigenvalue(anchor, band, mu);
        if (d > tol) {
            positive_indices.push_back(band);
            positive_inverse_sqrt.push_back(1.0 / std::sqrt(d));
        } else if (d < -tol) {
            negative_indices.push_back(band);
            negative_inverse_sqrt.push_back(1.0 / std::sqrt(-d));
        } else {
            return {};
        }
    }

    for (const auto vertex_id : simplex.vertex_ids) {
        const auto &target = cache.get(vertex_id);
        const auto &overlap = overlap_matrix(
            anchor,
            target,
            anchor_id,
            vertex_id,
            ndof,
            overlap_cache
        );

        if (!sign_block_passes(
                overlap,
                target,
                std::span<const size_t>(positive_indices.data(), positive_indices.size()),
                std::span<const double>(positive_inverse_sqrt.data(), positive_inverse_sqrt.size()),
                mu,
                1.0
            )) {
            return {};
        }
        if (!sign_block_passes(
                overlap,
                target,
                std::span<const size_t>(negative_indices.data(), negative_indices.size()),
                std::span<const double>(negative_inverse_sqrt.data(), negative_inverse_sqrt.size()),
                mu,
                -1.0
            )) {
            return {};
        }
    }

    return SignedInertiaDecision{.certified_safe = true};
}

}  // namespace lineartetrahedron::signed_inertia
