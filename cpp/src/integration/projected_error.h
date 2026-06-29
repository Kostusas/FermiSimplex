#pragma once

#include "core/vertex_spectra.h"
#include "integration/workspace.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>

namespace lineartetrahedron {

struct ProjectedErrorEstimate {
    double rho_down = 0.0;
    double rho_up = 0.0;
};

ProjectedErrorEstimate estimate_projected_error(
    const IntegrationWorkspace &workspace,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    size_t lower_band,
    size_t upper_band
);

}  // namespace lineartetrahedron
