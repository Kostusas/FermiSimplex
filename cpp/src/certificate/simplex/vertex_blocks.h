#pragma once

#include "certificate/linalg/matrix.h"

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

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

}  // namespace lineartetrahedron::simplex_certificate::detail
