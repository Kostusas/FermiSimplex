#include <fermisimplex/spectral_mesh.h>

#include "core/simplex_geometry.h"
#include "linalg/blas_lapack.h"

#include <adaptivesimplex/core/root_mesh.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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

bool is_finite(std::complex<double> value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void validate_hamiltonian(
    std::span<const std::complex<double>> matrix,
    std::size_t size
) {
    auto scale = 1.0;
    for (const auto value : matrix) {
        if (!is_finite(value)) {
            throw std::runtime_error("SpectralMesh: Hamiltonian entries must be finite");
        }
        scale = std::max({scale, std::abs(value.real()), std::abs(value.imag())});
    }

    // Accept only roundoff-sized asymmetry. The unit scale floor also permits
    // machine-level noise when larger terms cancel to a nearly zero matrix.
    // LAPACK reads one triangle, so a larger mismatch must be rejected instead
    // of silently discarding the other triangle.
    const auto roundoff =
        128.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max<std::size_t>(size, 1)) * scale;
    for (std::size_t column = 0; column < size; ++column) {
        const auto diagonal = matrix[column * size + column];
        if (std::abs(diagonal.imag()) > roundoff) {
            throw std::runtime_error("SpectralMesh: Hamiltonian must be Hermitian");
        }
        for (std::size_t row = column + 1; row < size; ++row) {
            const auto lower = matrix[column * size + row];
            const auto upper = matrix[row * size + column];
            if (std::abs(lower - std::conj(upper)) > roundoff) {
                throw std::runtime_error("SpectralMesh: Hamiltonian must be Hermitian");
            }
        }
    }
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
    if (reduced_point.size() != model_->ndim()) {
        throw std::runtime_error("SpectralMesh: point dimension mismatch");
    }
    if (!std::all_of(reduced_point.begin(), reduced_point.end(), [](double coordinate) {
            return std::isfinite(coordinate);
        })) {
        throw std::runtime_error("SpectralMesh: point coordinates must be finite");
    }
    auto matrix = model_->evaluate(reduced_point);
    if (matrix.size() != model_->ndof() * model_->ndof()) {
        throw std::runtime_error("SpectralMesh: Hamiltonian matrix shape mismatch");
    }
    validate_hamiltonian(matrix, model_->ndof());
    return matrix;
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
