#pragma once

#include "certificate/simplex_certificate.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

struct AnchorSelection {
    size_t ndof = 0;
    size_t nocc = 0;
    size_t vertex_index = 0;
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

AnchorSelectionResult choose_anchor_spectrum(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    double tol
);

AnchorSplitResult split_anchor_spectrum(
    std::span<const double> anchor_eigenvalues,
    double mu,
    double tol,
    size_t fallback_ndof
);

}  // namespace lineartetrahedron::simplex_certificate::detail
