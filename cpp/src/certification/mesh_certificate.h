#pragma once

#include <lineartetrahedron/certification.h>
#include <lineartetrahedron/spectral_mesh.h>

namespace lineartetrahedron::certification {

namespace core = adaptivesimplex::core;

SimplexCertificate certify_mesh_simplex(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
    double linearization_error_bound,
    double tolerance = kDefaultTolerance
);

}  // namespace lineartetrahedron::certification
