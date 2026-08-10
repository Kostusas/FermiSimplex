#pragma once

#include <fermisimplex/certification.h>
#include <fermisimplex/spectral_mesh.h>

#include <memory>

namespace fermisimplex::certification {

namespace core = adaptivesimplex::core;

class PreparedSimplexCertificate {
public:
    struct Impl;

    explicit PreparedSimplexCertificate(std::shared_ptr<const Impl> impl);

    SimplexCertificate certify(double linearization_error_bound) const;
    OccupationBounds occupation_bounds(double linearization_error_bound) const;

private:
    std::shared_ptr<const Impl> impl_;
};

PreparedSimplexCertificate prepare_mesh_simplex_certificate(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
    double tolerance = kDefaultTolerance
);

PreparedSimplexCertificate prepare_simplex_certificate(
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors,
    double mu,
    double tolerance = kDefaultTolerance
);

SimplexCertificate certify_mesh_simplex(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
    double linearization_error_bound,
    double tolerance = kDefaultTolerance
);

}  // namespace fermisimplex::certification
