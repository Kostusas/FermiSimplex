#include "certification/simplex/anchor_selection.h"

#include "certification/linalg/matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lineartetrahedron::certification::detail {
VertexSpectraAnalysis analyze_vertex_spectra(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    double tol
) {
    const auto ndof = eigenvalues.front().size();
    auto has_reference_nocc = false;
    auto reference_nocc = size_t{0};
    auto classification = VertexSpectraClassification::NeedsCertificate;
    auto best_anchor = size_t{0};
    auto best_anchor_nocc = size_t{0};
    auto best_gap = -std::numeric_limits<double>::infinity();

    for (size_t vertex_index = 0; vertex_index < eigenvalues.size(); ++vertex_index) {
        const auto vertex_eigenvalues = eigenvalues[vertex_index];
        auto nocc = size_t{0};
        auto min_gap = std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < ndof; ++band) {
            const auto d = signed_eigenvalue(vertex_eigenvalues, band, mu);
            const auto gap = std::abs(d);
            min_gap = std::min(min_gap, gap);
            if (gap <= tol) {
                classification = VertexSpectraClassification::VisibleGapless;
            }
            if (d < -tol) {
                ++nocc;
            }
        }
        if (!has_reference_nocc) {
            reference_nocc = nocc;
            has_reference_nocc = true;
        } else if (nocc != reference_nocc) {
            classification = VertexSpectraClassification::VisibleGapless;
        }
        if (min_gap > best_gap) {
            best_gap = min_gap;
            best_anchor = vertex_index;
            best_anchor_nocc = nocc;
        }
    }

    return VertexSpectraAnalysis{
        .classification = classification,
        .anchor = AnchorSelection{
            .nocc = best_anchor_nocc,
            .vertex_index = best_anchor,
        },
    };
}

AnchorSplit split_anchor_spectrum(
    std::span<const double> anchor_eigenvalues,
    double mu,
    size_t nocc
) {
    AnchorSplit split;
    split.occupied_gaps.reserve(nocc);
    split.unoccupied_gaps.reserve(anchor_eigenvalues.size() - nocc);

    for (size_t band = 0; band < nocc; ++band) {
        split.occupied_gaps.push_back(mu - anchor_eigenvalues[band]);
    }
    for (size_t band = nocc; band < anchor_eigenvalues.size(); ++band) {
        split.unoccupied_gaps.push_back(anchor_eigenvalues[band] - mu);
    }
    return split;
}

}  // namespace lineartetrahedron::certification::detail
