#pragma once

#include "lineartetrahedron/integration_workspace.h"
#include "lineartetrahedron/types.h"
#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <map>

namespace lineartetrahedron {

using ChargeWeylMarkerCache = std::map<adaptivesimplex::core::SimplexId, double>;

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    double weyl_indicator_error = 0.0,
    ChargeWeylMarkerCache *weyl_marker_cache = nullptr
);

}  // namespace lineartetrahedron
