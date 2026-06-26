#pragma once

#include "core/types.h"
#include "core/vertex_spectra.h"
#include "integration/charge_certificate_cache.h"
#include "integration/workspace.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

namespace lineartetrahedron {

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    bool certify = true,
    ChargeCertificateCache *certificate_cache = nullptr
);

}  // namespace lineartetrahedron
