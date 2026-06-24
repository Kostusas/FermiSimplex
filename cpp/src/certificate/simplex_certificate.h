#pragma once

#include "core/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>

namespace lineartetrahedron::simplex_certificate {

namespace core = adaptivesimplex::core;

enum class SimplexCertificateStatus {
    CertifiedGapped,
    VisibleCut,
    Inconclusive,
};

struct SimplexCertificate {
    SimplexCertificateStatus status = SimplexCertificateStatus::Inconclusive;
    size_t vertex_occupation = 0;
};

SimplexCertificate certify_simplex_gap(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double margin,
    double tol
);

}  // namespace lineartetrahedron::simplex_certificate
