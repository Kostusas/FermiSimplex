#pragma once

#include "core/vertex_spectra.h"
#include "fermi_surface/fermi_surface.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {

namespace core = adaptivesimplex::core;

using SpectraCache = core::VertexCache<VertexSpectra>;

extern thread_local FermiSurfaceStats fermi_surface_stats_;

core::Geometry make_fermi_geometry(size_t ndim);

std::vector<core::SimplexId> active_simplices(const core::Geometry &geometry);

template <class Cache>
std::vector<core::VertexId> missing_vertices_for(
    core::Geometry &geometry,
    Cache &cache,
    const std::vector<core::SimplexId> &simplex_ids
) {
    return geometry.missing_vertices(
        std::span<const core::SimplexId>(simplex_ids.data(), simplex_ids.size()),
        cache,
        0
    );
}

template <class Cache, class Evaluator>
void evaluate_vertices(
    core::Geometry &geometry,
    Cache &cache,
    const Evaluator &evaluator,
    std::span<const core::VertexId> vertex_ids
) {
    fermi_surface_stats_.evaluated_vertices += vertex_ids.size();
    for (const auto vertex_id : vertex_ids) {
        const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        cache.insert(
            vertex_id,
            evaluator.evaluate_reduced_point(
                std::span<const double>(reduced_point.data(), reduced_point.size())
            )
        );
    }
}

template <class Cache, class Evaluator>
void evaluate_missing_vertices(
    core::Geometry &geometry,
    Cache &cache,
    const Evaluator &evaluator,
    const std::vector<core::SimplexId> &simplex_ids
) {
    const auto missing = missing_vertices_for(geometry, cache, simplex_ids);
    evaluate_vertices(
        geometry,
        cache,
        evaluator,
        std::span<const core::VertexId>(missing.data(), missing.size())
    );
}

}  // namespace lineartetrahedron::fermi_surface_detail
