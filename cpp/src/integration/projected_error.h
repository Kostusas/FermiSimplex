#pragma once

#include <lineartetrahedron/spectral_mesh.h>

#include <cstddef>

namespace lineartetrahedron {

struct ProjectedResidualEstimate {
    // These are sampled estimates, not certified bounds over the simplex.
    // For R = H_actual - H_linear, sampled eigenvalues lie approximately in
    // [-negative_estimate, positive_estimate].
    double negative_estimate = 0.0;
    double positive_estimate = 0.0;
};

ProjectedResidualEstimate estimate_projected_residual(
    const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id,
    size_t lower_band,
    size_t upper_band
);

}  // namespace lineartetrahedron
