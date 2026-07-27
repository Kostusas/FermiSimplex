#pragma once

#include <fermisimplex/spectral_mesh.h>

#include <cstddef>
#include <map>
#include <tuple>

namespace fermisimplex {

struct ProjectedErrorEstimate {
    // These are sampled estimates, not certified bounds over the simplex.
    // Ordered projected eigenvalues differ from barycentrically interpolated
    // vertex eigenvalues approximately within
    // [-negative_estimate, positive_estimate].
    double negative_estimate = 0.0;
    double positive_estimate = 0.0;
};

// Scope one cache to one integration over an immutable mesh. Shared simplex
// edges are keyed canonically together with their selected band interval.
struct ProjectedErrorCache {
    using EdgeKey = std::tuple<
        adaptivesimplex::core::VertexId,
        adaptivesimplex::core::VertexId,
        std::size_t,
        std::size_t
    >;
    std::map<EdgeKey, ProjectedErrorEstimate> edge_estimates;
};

ProjectedErrorEstimate estimate_projected_error(
    const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id,
    std::size_t lower_band,
    std::size_t upper_band,
    ProjectedErrorCache *cache = nullptr
);

}  // namespace fermisimplex
