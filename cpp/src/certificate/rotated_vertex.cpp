#include "certificate/simplex_certificate.h"

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron::simplex_certificate {

namespace {

SimplexCertificate certificate(
    SimplexCertificateStatus status,
    size_t vertex_occupation
) {
    return SimplexCertificate{
        .status = status,
        .vertex_occupation = vertex_occupation,
    };
}

detail::Complex lower_triangle_entry(
    const std::vector<detail::Complex> &matrix,
    size_t size,
    size_t row,
    size_t column
) {
    if (row >= column) {
        return matrix[detail::column_major_index(row, column, size)];
    }
    return std::conj(matrix[detail::column_major_index(column, row, size)]);
}

double gershgorin_row_lower_bound(
    const std::vector<detail::Complex> &matrix,
    size_t size,
    size_t row
) {
    auto bound = std::real(matrix[detail::column_major_index(row, row, size)]);
    for (size_t column = 0; column < size; ++column) {
        if (column != row) {
            bound -= std::abs(lower_triangle_entry(matrix, size, row, column));
        }
    }
    return bound;
}

double side_mu_radius(
    const std::vector<detail::Complex> &signed_block,
    std::span<const double> gram_row_bounds,
    size_t size,
    double tol
) {
    if (size == 0) {
        return std::numeric_limits<double>::infinity();
    }

    auto radius = std::numeric_limits<double>::infinity();
    const auto margin = detail::positive_definite_margin(tol);
    for (size_t row = 0; row < size; ++row) {
        const auto denominator = gram_row_bounds[row];
        if (denominator <= 0.0) {
            return 0.0;
        }
        const auto row_radius =
            (gershgorin_row_lower_bound(signed_block, size, row) - margin) / denominator;
        radius = std::min(radius, row_radius);
    }
    return std::max(0.0, radius);
}

std::vector<double> positive_frame_gram_row_bounds(
    std::span<const detail::Complex> rotation,
    size_t npos,
    size_t nneg
) {
    std::vector<double> result(npos, 1.0);
    if (npos == 0 || nneg == 0) {
        return result;
    }

    std::vector<detail::Complex> gram(npos * npos, detail::Complex{0.0, 0.0});
    detail::gemm(
        'N',
        'C',
        npos,
        npos,
        nneg,
        detail::Complex{1.0, 0.0},
        rotation.data(),
        npos,
        rotation.data(),
        npos,
        detail::Complex{0.0, 0.0},
        gram.data(),
        npos
    );
    for (size_t row = 0; row < npos; ++row) {
        result[row] += std::real(gram[detail::column_major_index(row, row, npos)]);
        for (size_t column = 0; column < npos; ++column) {
            if (column != row) {
                result[row] += std::abs(lower_triangle_entry(gram, npos, row, column));
            }
        }
    }
    return result;
}

std::vector<double> negative_frame_gram_row_bounds(
    std::span<const detail::Complex> rotation,
    size_t npos,
    size_t nneg
) {
    std::vector<double> result(nneg, 1.0);
    if (npos == 0 || nneg == 0) {
        return result;
    }

    std::vector<detail::Complex> gram(nneg * nneg, detail::Complex{0.0, 0.0});
    detail::gemm(
        'C',
        'N',
        nneg,
        nneg,
        npos,
        detail::Complex{1.0, 0.0},
        rotation.data(),
        npos,
        rotation.data(),
        npos,
        detail::Complex{0.0, 0.0},
        gram.data(),
        nneg
    );
    for (size_t row = 0; row < nneg; ++row) {
        result[row] += std::real(gram[detail::column_major_index(row, row, nneg)]);
        for (size_t column = 0; column < nneg; ++column) {
            if (column != row) {
                result[row] += std::abs(lower_triangle_entry(gram, nneg, row, column));
            }
        }
    }
    return result;
}

SimplexCertificate inconclusive_certificate(
    double mu,
    size_t vertex_occupation,
    size_t ndof,
    size_t npos,
    size_t nneg,
    const std::vector<detail::VertexBlocks> &blocks,
    double tol,
    bool estimate_occupation_bounds
) {
    auto result = certificate(SimplexCertificateStatus::Inconclusive, vertex_occupation);
    if (!estimate_occupation_bounds) {
        return result;
    }

    std::vector<std::vector<detail::Complex>> positive_blocks;
    std::vector<std::vector<detail::Complex>> negative_blocks;
    positive_blocks.reserve(blocks.size());
    negative_blocks.reserve(blocks.size());
    for (const auto &block : blocks) {
        positive_blocks.push_back(block.positive_same_sign);

        auto negative_block = block.negative_same_sign;
        detail::negate_in_place(negative_block);
        negative_blocks.push_back(std::move(negative_block));
    }

    const auto minus_estimate =
        detail::estimate_common_rank_with_mu_radius(negative_blocks, nneg, tol);
    const auto plus_estimate =
        detail::estimate_common_rank_with_mu_radius(positive_blocks, npos, tol);
    const auto lower = minus_estimate.rank;
    const auto upper = ndof - plus_estimate.rank;
    if (lower <= upper) {
        result.has_occupation_bounds = true;
        result.lower_occupation_bound = lower;
        result.upper_occupation_bound = upper;
        result.lower_mu_bound = mu - minus_estimate.mu_radius;
        result.upper_mu_bound = mu + plus_estimate.mu_radius;
    }
    return result;
}

}  // namespace

SimplexCertificate certify_simplex_gap(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double margin,
    double tol,
    bool estimate_occupation_bounds
) {
    using namespace detail;

    const auto &simplex = geometry.simplices().simplex(simplex_id);
    if (simplex.vertex_ids.empty()) {
        throw std::runtime_error("certify_simplex_gap: simplex must not be empty");
    }

    const auto &first = vertex_cache.get(simplex.vertex_ids.front());
    const auto ndof = first.eigenvalues.size();
    auto has_reference_occupation = false;
    auto reference_occupation = size_t{0};
    auto best_anchor = simplex.vertex_ids.front();
    auto best_gap = -std::numeric_limits<double>::infinity();

    for (const auto vertex_id : simplex.vertex_ids) {
        const auto &spectra = vertex_cache.get(vertex_id);
        auto vertex_occupation = size_t{0};
        auto min_gap = std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < ndof; ++band) {
            const auto d = signed_eigenvalue(spectra, band, mu);
            const auto gap = std::abs(d);
            if (gap <= tol) {
                return certificate(SimplexCertificateStatus::VisibleCut, vertex_occupation);
            }
            min_gap = std::min(min_gap, gap);
            if (d < -tol) {
                ++vertex_occupation;
            }
        }
        if (!has_reference_occupation) {
            reference_occupation = vertex_occupation;
            has_reference_occupation = true;
        } else if (vertex_occupation != reference_occupation) {
            return certificate(SimplexCertificateStatus::VisibleCut, reference_occupation);
        }
        if (min_gap > best_gap) {
            best_gap = min_gap;
            best_anchor = vertex_id;
        }
    }

    const auto &anchor = vertex_cache.get(best_anchor);
    std::vector<double> positive_gaps;
    std::vector<double> negative_gaps;
    positive_gaps.reserve(ndof);
    negative_gaps.reserve(ndof);

    for (size_t band = 0; band < ndof; ++band) {
        const auto d = signed_eigenvalue(anchor, band, mu);
        if (d > tol) {
            positive_gaps.push_back(d);
        } else if (d < -tol) {
            negative_gaps.push_back(-d);
        } else {
            return certificate(SimplexCertificateStatus::Inconclusive, reference_occupation);
        }
    }

    const auto npos = positive_gaps.size();
    const auto nneg = negative_gaps.size();
    const auto anchor_vectors =
        std::span<const Complex>(anchor.eigenvectors.data(), anchor.eigenvectors.size());

    std::vector<VertexBlocks> blocks;
    blocks.reserve(simplex.vertex_ids.size());
    std::vector<Complex> average_mixed(npos * nneg, Complex{0.0, 0.0});
    for (const auto vertex_id : simplex.vertex_ids) {
        blocks.push_back(build_vertex_blocks(
            anchor_vectors,
            ndof,
            npos,
            nneg,
            vertex_cache.get(vertex_id),
            mu
        ));
        for (size_t index = 0; index < average_mixed.size(); ++index) {
            average_mixed[index] += blocks.back().positive_negative_coupling[index];
        }
    }

    const auto average_scale = 1.0 / static_cast<double>(simplex.vertex_ids.size());
    for (auto &value : average_mixed) {
        value *= average_scale;
    }

    std::vector<Complex> rotation(npos * nneg, Complex{0.0, 0.0});
    for (size_t neg = 0; neg < nneg; ++neg) {
        for (size_t pos = 0; pos < npos; ++pos) {
            rotation[column_major_index(pos, neg, npos)] =
                average_mixed[column_major_index(pos, neg, npos)] /
                (positive_gaps[pos] + negative_gaps[neg]);
        }
    }

    const auto rotation_span = std::span<const Complex>(rotation.data(), rotation.size());
    const auto positive_gram_row_bounds =
        positive_frame_gram_row_bounds(rotation_span, npos, nneg);
    const auto negative_gram_row_bounds =
        negative_frame_gram_row_bounds(rotation_span, npos, nneg);
    auto positive_mu_radius = std::numeric_limits<double>::infinity();
    auto negative_mu_radius = std::numeric_limits<double>::infinity();
    for (const auto &vertex_blocks : blocks) {
        auto positive_block =
            rotated_positive_block(vertex_blocks, rotation_span, npos, nneg);
        subtract_positive_frame_margin(positive_block, rotation_span, npos, nneg, margin);
        positive_mu_radius = std::min(
            positive_mu_radius,
            side_mu_radius(
                positive_block,
                std::span<const double>(
                    positive_gram_row_bounds.data(),
                    positive_gram_row_bounds.size()
                ),
                npos,
                tol
            )
        );
        if (!positive_definite(std::move(positive_block), npos, tol)) {
            return inconclusive_certificate(
                mu,
                reference_occupation,
                ndof,
                npos,
                nneg,
                blocks,
                tol,
                estimate_occupation_bounds
            );
        }

        auto negative_block =
            rotated_negative_block(vertex_blocks, rotation_span, npos, nneg);
        negate_in_place(negative_block);
        subtract_negative_frame_margin(negative_block, rotation_span, npos, nneg, margin);
        negative_mu_radius = std::min(
            negative_mu_radius,
            side_mu_radius(
                negative_block,
                std::span<const double>(
                    negative_gram_row_bounds.data(),
                    negative_gram_row_bounds.size()
                ),
                nneg,
                tol
            )
        );
        if (!positive_definite(std::move(negative_block), nneg, tol)) {
            return inconclusive_certificate(
                mu,
                reference_occupation,
                ndof,
                npos,
                nneg,
                blocks,
                tol,
                estimate_occupation_bounds
            );
        }
    }

    auto result = certificate(SimplexCertificateStatus::CertifiedGapped, reference_occupation);
    result.lower_mu_bound = mu - negative_mu_radius;
    result.upper_mu_bound = mu + positive_mu_radius;
    return result;
}

}  // namespace lineartetrahedron::simplex_certificate
