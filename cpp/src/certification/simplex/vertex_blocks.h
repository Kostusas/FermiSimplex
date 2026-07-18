#pragma once

#include "certification/linalg/matrix.h"

#include <cstddef>
#include <span>
#include <vector>

namespace fermisimplex::certification::detail {

struct VertexBlocks {
    std::vector<Complex> unoccupied_block;
    std::vector<Complex> occupied_block;
    std::vector<Complex> coupling_block;
};

VertexBlocks build_vertex_blocks(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t nocc,
    std::span<const double> target_eigenvalues,
    std::span<const Complex> target_eigenvectors,
    double mu
);

VertexBlocks build_anchor_vertex_blocks(
    size_t ndof,
    size_t nocc,
    std::span<const double> anchor_eigenvalues,
    double mu
);

}  // namespace fermisimplex::certification::detail
