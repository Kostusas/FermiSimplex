#pragma once

#include "lineartetrahedron/fermi_surface.h"
#include "lineartetrahedron/simplex_certificate.h"
#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {

namespace core = adaptivesimplex::core;

using SpectraCache = core::VertexCache<VertexSpectra>;
using Clock = std::chrono::steady_clock;

extern thread_local FermiSurfaceStats fermi_surface_stats_;

std::uint64_t nanoseconds_since(Clock::time_point start);

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
    const auto start = Clock::now();
    ++fermi_surface_stats_.vertex_evaluation_calls;
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
    fermi_surface_stats_.vertex_evaluation_nanoseconds += nanoseconds_since(start);
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

struct MarkResult {
    std::vector<core::SimplexId> marked;
    std::vector<core::SimplexId> surface_terminal;
    std::int64_t safe = 0;
    std::int64_t cut = 0;
    std::int64_t feature_size = 0;
    std::int64_t unresolved = 0;
};

void extract_surface(
    const HamiltonianModel &model,
    const core::Geometry &geometry,
    const SpectraCache &cache,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    double tol,
    bool return_nearest_vertex_states,
    FermiSurfaceResult &result
);

}  // namespace lineartetrahedron::fermi_surface_detail
