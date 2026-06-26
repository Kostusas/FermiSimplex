#pragma once

#include "certificate/simplex_certificate.h"
#include "certificate/simplex/vertex_blocks.h"

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

enum class OccupationSector {
    Occupied,
    Unoccupied,
};

struct OccupationSectorCheck {
    bool passed = false;
    double mu_radius = 0.0;
};

size_t sector_size(OccupationSector sector, size_t ndof, size_t nocc);

size_t opposite_sector_size(OccupationSector sector, size_t ndof, size_t nocc);

OccupationSectorCheck check_occupation_sector(
    OccupationSector sector,
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    size_t size,
    size_t opposite_size,
    double margin,
    double tol
);

SimplexCertificate occupation_bounded_inconclusive(
    double mu,
    size_t ndof,
    size_t nocc,
    const std::vector<VertexBlocks> &blocks,
    double tol,
    bool estimate_occupation_bounds
);

}  // namespace lineartetrahedron::simplex_certificate::detail
