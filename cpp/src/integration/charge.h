#pragma once

#include <fermisimplex/certification.h>
#include <fermisimplex/integration.h>

#include "integration/projected_error.h"

#include <adaptivesimplex/core/geometry.h>

#include <cstdint>

namespace fermisimplex::integration_detail {

struct ChargeContribution {
    double value = 0.0;
    double dcharge_dmu = 0.0;
    double projected_error = 0.0;
    double certified_error_bound = 0.0;
    std::int64_t visible_gapless_simplices = 0;
    std::int64_t inconclusive_simplices = 0;

    ChargeContribution &operator+=(const ChargeContribution &other) noexcept;
    ChargeContribution &operator-=(const ChargeContribution &other) noexcept;
};

double projected_occupation_shell(
    double mu,
    const SpectralMesh &mesh,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    certification::OccupationBounds occupation_bounds,
    ProjectedErrorEstimate projected_error
);

double projected_charge_error(
    double mu,
    const SpectralMesh &mesh,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    certification::OccupationBounds occupation_bounds,
    ProjectedErrorCache *projected_error_cache = nullptr
);

ChargeContribution band_charge_on_simplex(
    double mu,
    const SpectralMesh &mesh,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id
);

ChargeContribution charge_on_simplex(
    double mu,
    const SpectralMesh &mesh,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    double curvature_bound,
    ProjectedErrorCache *projected_error_cache = nullptr
);

}  // namespace fermisimplex::integration_detail
