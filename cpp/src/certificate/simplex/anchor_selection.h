#pragma once

#include "certificate/simplex_certificate.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

struct AnchorSelection {
    size_t ndof = 0;
    size_t nocc = 0;
    core::VertexId vertex_id = 0;
};

struct AnchorSelectionResult {
    AnchorSelection selection;
    std::optional<SimplexCertificate> early_certificate;
};

struct AnchorSplit {
    std::vector<double> unoccupied_gaps;
    std::vector<double> occupied_gaps;
};

struct AnchorSplitResult {
    AnchorSplit split;
    std::optional<SimplexCertificate> early_certificate;
};

AnchorSelectionResult choose_anchor_vertex(
    double mu,
    const core::Simplex &simplex,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double tol
);

AnchorSplitResult split_anchor_spectrum(
    const VertexSpectra &anchor,
    double mu,
    double tol,
    size_t fallback_ndof
);

}  // namespace lineartetrahedron::simplex_certificate::detail
