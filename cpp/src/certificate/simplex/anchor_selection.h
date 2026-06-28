#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

enum class VertexSpectraClassification {
    NeedsCertificate,
    VisibleGapless,
};

struct AnchorSelection {
    size_t ndof = 0;
    size_t nocc = 0;
    size_t vertex_index = 0;
};

struct AnchorSplit {
    std::vector<double> unoccupied_gaps;
    std::vector<double> occupied_gaps;
};

VertexSpectraClassification classify_vertex_spectra(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    double tol
);

AnchorSelection choose_anchor_spectrum(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    double tol
);

std::optional<AnchorSplit> split_anchor_spectrum(
    std::span<const double> anchor_eigenvalues,
    double mu,
    double tol
);

}  // namespace lineartetrahedron::simplex_certificate::detail
