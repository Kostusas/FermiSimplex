#pragma once

#include <fermisimplex/spectral_mesh.h>

#include <cstddef>
#include <limits>

namespace fermisimplex {

struct ProjectedResidualEstimate {
    // These are sampled estimates, not certified bounds over the simplex.
    // The ordered Ritz eigenvalues in the interpolated vertex subspace differ
    // from the barycentrically interpolated vertex eigenvalues approximately
    // within [-negative_estimate, positive_estimate].
    double negative_estimate = 0.0;
    double positive_estimate = 0.0;
    std::size_t union_dimension = 0;
    double minimum_projector_gap = std::numeric_limits<double>::infinity();
};

ProjectedResidualEstimate estimate_projected_residual(
    const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id,
    size_t lower_band,
    size_t upper_band
);

}  // namespace fermisimplex
