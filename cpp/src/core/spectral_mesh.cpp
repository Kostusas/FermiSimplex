#include <fermisimplex/spectral_mesh.h>

#include "core/simplex_geometry.h"
#include "linalg/blas_lapack.h"

#include <adaptivesimplex/core/root_mesh.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>

namespace fermisimplex {
namespace core = adaptivesimplex::core;
namespace {

std::shared_ptr<const HamiltonianModel> validate_model(
    std::shared_ptr<const HamiltonianModel> model
) {
    if (!model) {
        throw std::runtime_error("SpectralMesh: model must not be null");
    }
    if (model->ndim() == 0 || model->ndof() == 0) {
        throw std::runtime_error("SpectralMesh: dimensions must be positive");
    }
    return model;
}

double validate_tolerance(double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::runtime_error("SpectralMesh: tolerance must be finite and non-negative");
    }
    return tolerance;
}

}  // namespace

SpectralMesh::SpectralMesh(
    std::shared_ptr<const HamiltonianModel> model,
    double tolerance,
    std::uint32_t root_level
) : model_(validate_model(std::move(model))),
    geometry_(core::root_geometry(model_->ndim(), root_level)),
    tolerance_(validate_tolerance(tolerance)) {}

Eigensystem SpectralMesh::spectrum(std::span<const double> reduced_point) const {
    auto matrix = hamiltonian(reduced_point);
    auto result = Eigensystem{};
    linalg::diagonalize_hermitian_in_place(
        matrix,
        result.eigenvalues,
        model_->ndof(),
        true,
        "SpectralMesh"
    );
    result.eigenvectors = std::move(matrix);
    return result;
}

std::vector<std::complex<double>> SpectralMesh::hamiltonian(
    std::span<const double> reduced_point
) const {
    return model_->evaluate(reduced_point);
}

double SpectralMesh::linearization_error_bound(
    core::SimplexId simplex_id,
    double curvature_bound
) const {
    if (!std::isfinite(curvature_bound) || curvature_bound < 0.0) {
        throw std::runtime_error(
            "curvature_bound must be finite and non-negative"
        );
    }
    return symmetric_linearization_error_bound(
        curvature_bound,
        simplex_diameter(geometry_, simplex_id)
    );
}

}  // namespace fermisimplex
