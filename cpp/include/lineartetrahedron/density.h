#pragma once

#include "lineartetrahedron/integration_workspace.h"
#include "lineartetrahedron/types.h"
#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/adaptive/dense_value.h>
#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

class DensityComponents {
public:
    using DensityValue = adaptivesimplex::adaptive::DenseValue<std::complex<double>>;

    DensityComponents(
        size_t ndim,
        size_t ndof,
        KeyArray keys,
        ComponentIndexArray component_rows,
        ComponentIndexArray component_cols,
        ComponentIndexArray component_key_indices
    );

    size_t size() const noexcept { return component_count_; }

    DensityValue on_simplex(
        double mu,
        const IntegrationWorkspace &workspace,
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::SimplexId simplex_id,
        const adaptivesimplex::core::VertexCache<VertexSpectra> &cache
    ) const;

private:
    struct ComponentContribution {
        size_t component_index = 0;
        size_t key_index = 0;
    };
    struct ComponentPair {
        size_t row = 0;
        size_t col = 0;
        std::vector<ComponentContribution> contributions;
    };

    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t key_count_ = 0;
    size_t component_count_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<ComponentPair> pairs_;
};

}  // namespace lineartetrahedron
