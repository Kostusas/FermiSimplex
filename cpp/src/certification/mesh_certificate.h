#pragma once

#include <fermisimplex/certification.h>
#include <fermisimplex/spectral_mesh.h>

namespace fermisimplex::certification {

namespace core = adaptivesimplex::core;

SimplexCertificate certify_mesh_simplex(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
    double linearization_error_bound,
    double tolerance = kDefaultTolerance
);

}  // namespace fermisimplex::certification
