#pragma once

#include "certification/simplex/vertex_blocks.h"

#include <lineartetrahedron/certification.h>

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::certification::detail {

struct OccupationSectorCheck {
    bool passed = false;
    double mu_radius = 0.0;
};

OccupationSectorCheck check_unoccupied_sector(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    double linearization_error_bound,
    double tolerance
);

OccupationSectorCheck check_occupied_sector(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    double linearization_error_bound,
    double tolerance
);

SimplexCertificate make_unresolved_certificate(
    SimplexCertificateStatus status,
    double mu,
    const std::vector<VertexBlocks> &blocks,
    double linearization_error_bound,
    double tolerance
);

}  // namespace lineartetrahedron::certification::detail
