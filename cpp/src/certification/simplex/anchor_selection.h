#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace fermisimplex::certification::detail {

enum class VertexSpectraClassification {
    NeedsCertificate,
    VisibleGapless,
};

struct AnchorSelection {
    size_t nocc = 0;
    size_t vertex_index = 0;
};

struct VertexSpectraAnalysis {
    VertexSpectraClassification classification =
        VertexSpectraClassification::NeedsCertificate;
    AnchorSelection anchor;
};

struct AnchorSplit {
    std::vector<double> unoccupied_gaps;
    std::vector<double> occupied_gaps;
};

VertexSpectraAnalysis analyze_vertex_spectra(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    double tol
);

AnchorSplit split_anchor_spectrum(
    std::span<const double> anchor_eigenvalues,
    double mu,
    size_t nocc
);

}  // namespace fermisimplex::certification::detail
