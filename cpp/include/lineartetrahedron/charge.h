#pragma once

#include "lineartetrahedron/integration_workspace.h"
#include "lineartetrahedron/simplex_certificate.h"
#include "lineartetrahedron/types.h"
#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

namespace lineartetrahedron {

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    double certificate_error = 0.0
);

}  // namespace lineartetrahedron
