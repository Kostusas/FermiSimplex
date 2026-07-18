#pragma once

#include <lineartetrahedron/integration.h>

#include <adaptivesimplex/core/geometry.h>

#include <cstdint>

namespace lineartetrahedron::integration_detail {

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

ChargeContribution charge_on_simplex(
    double mu,
    const SpectralMesh &mesh,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    double curvature_bound
);

}  // namespace lineartetrahedron::integration_detail
