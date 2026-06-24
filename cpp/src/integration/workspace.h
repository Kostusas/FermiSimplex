#pragma once

#include "core/tight_binding.h"
#include "core/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/root_mesh.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace lineartetrahedron {

class IntegrationWorkspace {
public:
    explicit IntegrationWorkspace(
        std::shared_ptr<TightBindingModel> model,
        double tol = 1e-14
    );

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    double tol() const noexcept { return tol_; }
    size_t n_cached_nodes() const noexcept { return cache_.size(); }
    std::int64_t n_active_simplices() const noexcept {
        return static_cast<std::int64_t>(geometry_.simplices().n_active());
    }
    std::int64_t n_active_vertices() const {
        return static_cast<std::int64_t>(geometry_.n_active_vertices());
    }

    adaptivesimplex::core::Geometry &geometry() noexcept { return geometry_; }
    const adaptivesimplex::core::Geometry &geometry() const noexcept { return geometry_; }
    adaptivesimplex::core::VertexCache<VertexSpectra> &cache() noexcept { return cache_; }

    VertexSpectra evaluate_vertex(std::span<const double> reduced_point) const;

private:
    std::shared_ptr<TightBindingModel> model_;
    adaptivesimplex::core::Geometry geometry_;
    adaptivesimplex::core::VertexCache<VertexSpectra> cache_;
    VertexSpectraEvaluator evaluator_;
    double tol_ = 1e-14;
    size_t ndim_ = 0;
    size_t ndof_ = 0;
};

}  // namespace lineartetrahedron
