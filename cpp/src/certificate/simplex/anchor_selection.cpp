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

AnchorSelectionResult choose_anchor_spectrum(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    double tol
) {
    const auto ndof = eigenvalues.front().size();
    auto has_reference_nocc = false;
    auto reference_nocc = size_t{0};
    auto best_anchor = size_t{0};
    auto best_gap = -std::numeric_limits<double>::infinity();

    for (size_t vertex_index = 0; vertex_index < eigenvalues.size(); ++vertex_index) {
        const auto vertex_eigenvalues = eigenvalues[vertex_index];
        auto nocc = size_t{0};
        auto min_gap = std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < ndof; ++band) {
            const auto d = signed_eigenvalue(vertex_eigenvalues, band, mu);
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
            best_anchor = vertex_index;
        }
    }

    return AnchorSelectionResult{
        .selection = AnchorSelection{
            .ndof = ndof,
            .nocc = reference_nocc,
            .vertex_index = best_anchor,
        },
    };
}

AnchorSplitResult split_anchor_spectrum(
    std::span<const double> anchor_eigenvalues,
    double mu,
    double tol,
    size_t fallback_ndof
) {
    AnchorSplit split;
    split.unoccupied_gaps.reserve(anchor_eigenvalues.size());
    split.occupied_gaps.reserve(anchor_eigenvalues.size());

    for (size_t band = 0; band < anchor_eigenvalues.size(); ++band) {
        const auto d = signed_eigenvalue(anchor_eigenvalues, band, mu);
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
