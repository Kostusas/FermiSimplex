#pragma once

#include "core/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>
#include <adaptivesimplex/core/vertex_cache.h>

namespace lineartetrahedron::simplex_certificate {

namespace core = adaptivesimplex::core;

enum class InertiaDecision {
    CertifiedSafe,
    VisibleCut,
    Inconclusive,
};

InertiaDecision classify_rotated_vertex_frame_simplex(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double margin,
    double tol
);

}  // namespace lineartetrahedron::simplex_certificate
