#pragma once

#include <fermisimplex/integration.h>

#include <adaptivesimplex/adaptive/dense_value.h>
#include <adaptivesimplex/core/geometry.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fermisimplex::integration_detail {

class DensityRule {
public:
    using Value = adaptivesimplex::adaptive::DenseValue<std::complex<double>>;

    DensityRule(
        std::size_t ndim,
        std::size_t ndof,
        std::vector<LatticeVector> lattice_vectors
    );
    DensityRule(
        std::size_t ndim,
        std::size_t ndof,
        std::vector<LatticeVector> lattice_vectors,
        std::vector<DensityComponent> components
    );

    std::size_t lattice_vector_count() const noexcept {
        return lattice_vector_count_;
    }
    std::size_t ndof() const noexcept { return ndof_; }
    std::size_t output_size() const noexcept { return output_size_; }

    Value on_simplex(
        double mu,
        const SpectralMesh &mesh,
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::SimplexId simplex_id
    ) const;

private:
    struct Contribution {
        std::size_t output_index = 0;
        std::size_t lattice_vector_index = 0;
    };

    struct ComponentPair {
        std::size_t row = 0;
        std::size_t column = 0;
        std::size_t contribution_begin = 0;
        std::size_t contribution_end = 0;
    };

    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
    std::size_t lattice_vector_count_ = 0;
    std::size_t output_size_ = 0;
    std::vector<std::int64_t> lattice_vectors_;
    std::vector<ComponentPair> pairs_;
    std::vector<Contribution> contributions_;
};

}  // namespace fermisimplex::integration_detail
