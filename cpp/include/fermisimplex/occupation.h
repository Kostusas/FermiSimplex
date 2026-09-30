#pragma once

#include <fermisimplex/integration.h>
#include <optional>

namespace fermisimplex {

// Uniform implications are conditional on the quadratic interpolation allowance.
// With no supplied bound it is estimated from samples, and may miss features.
struct OccupationEnclosure {
    double charge_lower = 0;
    double charge_upper = 0;
    double density_cut_error = 0;
    double interpolation_error = 0;
    double model_error = 0;
    double safe_gap = 0;
    std::size_t active_dimension = 0;
    std::size_t occupation_lower = 0;
    std::size_t occupation_upper = 0;
    bool remainder_is_sampled = true;
    bool fixed_occupation() const { return occupation_lower == occupation_upper; }
};

// All vertex eigensystems of the selected simplex must already be cached.
OccupationEnclosure enclose_occupation(
    const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id,
    double mu,
    std::uint32_t depth,
    ChargeErrorStats &stats,
    std::optional<double> interpolation_error_bound = std::nullopt
);

}  // namespace fermisimplex
