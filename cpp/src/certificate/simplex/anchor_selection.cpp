#include "certificate/simplex/anchor_selection.h"

#include "certificate/linalg/matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

OccupationBounds unconstrained_occupation(size_t ndof) {
    return OccupationBounds{.lower = 0, .upper = ndof};
}

}  // namespace

AnchorSelectionResult choose_anchor_vertex(
    double mu,
    const core::Simplex &simplex,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double tol
) {
    const auto &first = vertex_cache.get(simplex.vertex_ids.front());
    const auto ndof = first.eigenvalues.size();
    auto has_reference_nocc = false;
    auto reference_nocc = size_t{0};
    auto best_anchor = simplex.vertex_ids.front();
    auto best_gap = -std::numeric_limits<double>::infinity();

    for (const auto vertex_id : simplex.vertex_ids) {
        const auto &spectra = vertex_cache.get(vertex_id);
        auto nocc = size_t{0};
        auto min_gap = std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < ndof; ++band) {
            const auto d = signed_eigenvalue(spectra, band, mu);
            const auto gap = std::abs(d);
            if (gap <= tol) {
                return AnchorSelectionResult{
                    .early_certificate = SimplexCertificate{
                        .status = SimplexCertificateStatus::VisibleGapless,
                        .occupation_bounds = unconstrained_occupation(ndof),
                    },
                };
            }
            min_gap = std::min(min_gap, gap);
            if (d < -tol) {
                ++nocc;
            }
        }
        if (!has_reference_nocc) {
            reference_nocc = nocc;
            has_reference_nocc = true;
        } else if (nocc != reference_nocc) {
            return AnchorSelectionResult{
                .early_certificate = SimplexCertificate{
                    .status = SimplexCertificateStatus::VisibleGapless,
                    .occupation_bounds = unconstrained_occupation(ndof),
                },
            };
        }
        if (min_gap > best_gap) {
            best_gap = min_gap;
            best_anchor = vertex_id;
        }
    }

    return AnchorSelectionResult{
        .selection = AnchorSelection{
            .ndof = ndof,
            .nocc = reference_nocc,
            .vertex_id = best_anchor,
        },
    };
}

AnchorSplitResult split_anchor_spectrum(
    const VertexSpectra &anchor,
    double mu,
    double tol,
    size_t fallback_ndof
) {
    AnchorSplit split;
    split.unoccupied_gaps.reserve(anchor.eigenvalues.size());
    split.occupied_gaps.reserve(anchor.eigenvalues.size());

    for (size_t band = 0; band < anchor.eigenvalues.size(); ++band) {
        const auto d = signed_eigenvalue(anchor, band, mu);
        if (d > tol) {
            split.unoccupied_gaps.push_back(d);
        } else if (d < -tol) {
            split.occupied_gaps.push_back(-d);
        } else {
            return AnchorSplitResult{
                .early_certificate = SimplexCertificate{
                    .status = SimplexCertificateStatus::Inconclusive,
                    .occupation_bounds = unconstrained_occupation(fallback_ndof),
                },
            };
        }
    }
    return AnchorSplitResult{.split = std::move(split)};
}

}  // namespace lineartetrahedron::simplex_certificate::detail
