#pragma once

#include <fermisimplex/occupation.h>

namespace fermisimplex::occupation_detail {

enum class VertexOccupation { uniform, touches_level, crossing };

// A crossing requires disjoint vertex occupation intervals: near-level
// eigenvalues alone cannot rule out a fixed occupation over the simplex.
VertexOccupation vertex_occupation(const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id, double mu);

// The shared enclosure traversal, consuming only its strict occupation bounds.
bool fixed_occupation(const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id, double mu, std::uint32_t depth,
    ChargeErrorStats &stats, std::optional<double> interpolation_error_bound = std::nullopt);

}  // namespace fermisimplex::occupation_detail
