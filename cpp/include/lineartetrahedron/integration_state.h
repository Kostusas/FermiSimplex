#pragma once

#include "lineartetrahedron/types.h"
#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/root_mesh.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace lineartetrahedron {

class IntegrationState {
public:
    struct DensityComponent {
        size_t row = 0;
        size_t col = 0;
        size_t key_index = 0;
    };

    IntegrationState(
        std::shared_ptr<TightBindingModel> model,
        KeyArray keys,
        ComponentIndexArray component_rows,
        ComponentIndexArray component_cols,
        ComponentIndexArray component_key_indices,
        double tol
    );

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    size_t density_component_count() const noexcept { return components_.size(); }
    size_t n_cached_nodes() const noexcept { return cache_.size(); }
    std::uint64_t n_kernel_evals() const noexcept { return evaluator_.n_kernel_evals(); }
    std::int64_t n_active_simplices() const noexcept {
        return static_cast<std::int64_t>(geometry_.simplices().n_active());
    }
    std::int64_t n_active_vertices() const {
        return static_cast<std::int64_t>(geometry_.n_active_vertices());
    }

    adaptivesimplex::core::Geometry &geometry() noexcept { return geometry_; }
    const adaptivesimplex::core::Geometry &geometry() const noexcept { return geometry_; }
    adaptivesimplex::core::VertexCache<VertexSpectra> &cache() noexcept { return cache_; }
    VertexSpectra evaluate_vertex(
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::VertexId vertex_id
    ) {
        return evaluator_.evaluate(geometry, vertex_id);
    }

    const std::vector<std::int64_t> &keys() const noexcept { return keys_; }
    const std::vector<DensityComponent> &components() const noexcept { return components_; }
    double tol() const noexcept { return tol_; }

private:
    std::shared_ptr<TightBindingModel> model_;
    adaptivesimplex::core::Geometry geometry_;
    adaptivesimplex::core::VertexCache<VertexSpectra> cache_;
    VertexSpectraEvaluator evaluator_;
    double tol_ = 1e-14;
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t n_keys_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<DensityComponent> components_;
};

}  // namespace lineartetrahedron
