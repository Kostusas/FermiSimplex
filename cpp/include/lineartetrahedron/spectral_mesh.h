#pragma once

#include <lineartetrahedron/hamiltonian.h>

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace lineartetrahedron {

struct Eigensystem {
    std::vector<double> eigenvalues;
    // Eigenvectors are the columns of this column-major square matrix.
    std::vector<std::complex<double>> eigenvectors;
};

using EigensystemCache = adaptivesimplex::core::VertexCache<Eigensystem>;

class SpectralMesh {
public:
    // Owns the AdaptiveSimplex geometry and the spectrum cache shared by all
    // charge, density-matrix, and Fermi-surface calculations.
    explicit SpectralMesh(
        std::shared_ptr<const HamiltonianModel> model,
        double tolerance = 1e-14,
        std::uint32_t root_level = 1
    );

    std::size_t ndim() const noexcept { return model_->ndim(); }
    std::size_t ndof() const noexcept { return model_->ndof(); }
    double tolerance() const noexcept { return tolerance_; }
    std::size_t cached_vertices() const noexcept { return eigensystems_.size(); }
    std::int64_t active_simplices() const noexcept {
        return static_cast<std::int64_t>(geometry_.simplices().n_active());
    }
    std::int64_t active_vertices() const {
        return static_cast<std::int64_t>(geometry_.n_active_vertices());
    }

    const HamiltonianModel &model() const noexcept { return *model_; }

    adaptivesimplex::core::Geometry &geometry() noexcept { return geometry_; }
    const adaptivesimplex::core::Geometry &geometry() const noexcept {
        return geometry_;
    }
    EigensystemCache &eigensystems() noexcept { return eigensystems_; }
    const EigensystemCache &eigensystems() const noexcept { return eigensystems_; }

    // Validates the Hamiltonian at one reduced coordinate.
    std::vector<std::complex<double>> hamiltonian(
        std::span<const double> reduced_point
    ) const;

    // Validates and diagonalizes the Hamiltonian at one reduced coordinate.
    Eigensystem spectrum(std::span<const double> reduced_point) const;

    double linearization_error_bound(
        adaptivesimplex::core::SimplexId simplex_id,
        double curvature_bound
    ) const;

private:
    std::shared_ptr<const HamiltonianModel> model_;
    adaptivesimplex::core::Geometry geometry_;
    EigensystemCache eigensystems_;
    double tolerance_ = 1e-14;
};

}  // namespace lineartetrahedron
