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
    ChargeCertificateCache *certificate_cache = nullptr,
    double hessian_bound = 0.0,
    double anharmonicity_bound = 0.0,
    InconclusiveChargeErrorMode inconclusive_error_mode =
        InconclusiveChargeErrorMode::Projected
);

ChargeValue charge_on_simplex_with_energy_bound(
    double mu,
    const IntegrationWorkspace &workspace,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    bool certify,
    ChargeCertificateCache *certificate_cache,
    double energy_bound,
    InconclusiveChargeErrorMode inconclusive_error_mode =
        InconclusiveChargeErrorMode::Projected
);

}  // namespace lineartetrahedron
