#pragma once

#include "certification/linalg/matrix.h"
#include "certification/simplex/vertex_blocks.h"

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::certification::detail {

struct SimplexBlocks {
    std::vector<VertexBlocks> vertices;
    std::vector<Complex> average_coupling;
};

SimplexBlocks build_simplex_blocks(
    double mu,
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const Complex>> eigenvectors,
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t nocc,
    size_t anchor_vertex_index
);

std::vector<Complex> perturbative_rotation(
    std::span<const Complex> average_coupling,
    std::span<const double> unoccupied_gaps,
    std::span<const double> occupied_gaps
);

}  // namespace lineartetrahedron::certification::detail
