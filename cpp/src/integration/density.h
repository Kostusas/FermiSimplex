#pragma once

#include <lineartetrahedron/spectral_mesh.h>

#include <adaptivesimplex/adaptive/dense_value.h>
#include <adaptivesimplex/core/geometry.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron::integration_detail {

class DensityMatrixRule {
public:
    using Value = adaptivesimplex::adaptive::DenseValue<std::complex<double>>;

    DensityMatrixRule(
        std::size_t ndim,
        std::size_t ndof,
        std::vector<LatticeVector> lattice_vectors
    );

    std::size_t lattice_vector_count() const noexcept {
        return lattice_vector_count_;
    }
    std::size_t ndof() const noexcept { return ndof_; }

    Value on_simplex(
        double mu,
        const SpectralMesh &mesh,
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::SimplexId simplex_id
    ) const;

private:
    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
    std::size_t lattice_vector_count_ = 0;
    std::vector<std::int64_t> lattice_vectors_;
};

}  // namespace lineartetrahedron::integration_detail
