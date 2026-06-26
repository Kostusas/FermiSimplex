#pragma once

#include "certificate/linalg/matrix.h"
#include "certificate/simplex_certificate.h"
#include "certificate/simplex/vertex_blocks.h"

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

struct SimplexBlocks {
    std::vector<VertexBlocks> vertices;
    std::vector<Complex> average_coupling;
};

SimplexBlocks build_simplex_blocks(
    double mu,
    const core::Simplex &simplex,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t nocc
);

std::vector<Complex> perturbative_rotation(
    std::span<const Complex> average_coupling,
    std::span<const double> unoccupied_gaps,
    std::span<const double> occupied_gaps
);

}  // namespace lineartetrahedron::simplex_certificate::detail
