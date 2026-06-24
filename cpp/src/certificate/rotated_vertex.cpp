#include "certificate/simplex_certificate.h"

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <span>
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

}  // namespace

SimplexCertificate certify_simplex_gap(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double margin,
    double tol
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
    for (const auto &vertex_blocks : blocks) {
        auto positive_block =
            rotated_positive_block(vertex_blocks, rotation_span, npos, nneg);
        subtract_positive_frame_margin(positive_block, rotation_span, npos, nneg, margin);
        if (!positive_definite(std::move(positive_block), npos, tol)) {
            return certificate(SimplexCertificateStatus::Inconclusive, reference_occupation);
        }

        auto negative_block =
            rotated_negative_block(vertex_blocks, rotation_span, npos, nneg);
        negate_in_place(negative_block);
        subtract_negative_frame_margin(negative_block, rotation_span, npos, nneg, margin);
        if (!positive_definite(std::move(negative_block), nneg, tol)) {
            return certificate(SimplexCertificateStatus::Inconclusive, reference_occupation);
        }
    }

    return certificate(SimplexCertificateStatus::CertifiedGapped, reference_occupation);
}

}  // namespace lineartetrahedron::simplex_certificate
