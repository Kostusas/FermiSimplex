#include "lineartetrahedron/fermi_surface.h"

#include "internal.h"

#include <adaptivesimplex/core/root_mesh.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace lineartetrahedron {

namespace core = adaptivesimplex::core;

namespace fermi_surface_detail {

namespace {

double max_reduced_edge_length(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto result = 0.0;
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        const auto left_point =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[left]).to_point();
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            const auto right_point =
                geometry.vertices().dyadic_vertex(simplex.vertex_ids[right]).to_point();
            auto squared = 0.0;
            for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
                const auto delta = left_point[axis] - right_point[axis];
                squared += delta * delta;
            }
            result = std::max(result, std::sqrt(squared));
        }
    }
    return result;
}

MarkResult classify_frontier(
    const core::Geometry &geometry,
    const SpectraCache &cache,
    const std::vector<core::SimplexId> &frontier,
    double mu,
    double min_feature_size,
    double margin,
    double tol
) {
    MarkResult result;
    ++fermi_surface_stats_.marking_passes;
    for (const auto simplex_id : frontier) {
        ++fermi_surface_stats_.active_simplex_visits;
        ++fermi_surface_stats_.classified_simplices;
        const auto refinable = max_reduced_edge_length(geometry, simplex_id) > min_feature_size;
        const auto decision = simplex_certificate::classify_rotated_vertex_frame_simplex(
            mu,
            geometry,
            simplex_id,
            cache,
            margin,
            tol
        );

        if (decision == simplex_certificate::InertiaDecision::CertifiedSafe) {
            ++result.safe;
        } else if (decision == simplex_certificate::InertiaDecision::VisibleCut) {
            if (refinable) {
                result.marked.push_back(simplex_id);
                ++fermi_surface_stats_.marked_simplices;
            } else {
                result.surface_terminal.push_back(simplex_id);
                ++result.cut;
            }
        } else if (refinable) {
            result.marked.push_back(simplex_id);
            ++fermi_surface_stats_.marked_simplices;
        } else {
            result.surface_terminal.push_back(simplex_id);
            ++result.feature_size;
        }
    }
    return result;
}

std::set<core::SimplexId> simplex_set(const std::vector<core::SimplexId> &simplex_ids) {
    return std::set<core::SimplexId>(simplex_ids.begin(), simplex_ids.end());
}

std::vector<core::SimplexId> next_frontier(
    const core::Geometry &geometry,
    const std::set<core::SimplexId> &previous_active,
    const std::vector<core::SimplexId> &marked,
    const std::vector<core::SimplexId> &refined
) {
    const auto refined_set = simplex_set(refined);
    std::set<core::SimplexId> seen;
    std::vector<core::SimplexId> result;

    for (const auto simplex_id : active_simplices(geometry)) {
        if (!previous_active.contains(simplex_id)) {
            seen.insert(simplex_id);
            result.push_back(simplex_id);
        }
    }
    for (const auto simplex_id : marked) {
        if (!refined_set.contains(simplex_id) && seen.insert(simplex_id).second) {
            result.push_back(simplex_id);
        }
    }
    return result;
}

}  // namespace

thread_local FermiSurfaceStats fermi_surface_stats_;

std::uint64_t nanoseconds_since(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
    );
}

core::Geometry make_fermi_geometry(size_t ndim) {
    return core::root_geometry(ndim, ndim == 1 ? 2U : 1U);
}

std::vector<core::SimplexId> active_simplices(const core::Geometry &geometry) {
    const auto active = geometry.simplices().active_simplices();
    return std::vector<core::SimplexId>(active.begin(), active.end());
}

}  // namespace fermi_surface_detail

void reset_fermi_surface_stats() {
    fermi_surface_detail::fermi_surface_stats_ = FermiSurfaceStats{};
}

FermiSurfaceStats fermi_surface_stats() {
    return fermi_surface_detail::fermi_surface_stats_;
}

FermiSurfaceResult fermi_surface(
    std::shared_ptr<TightBindingModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_nearest_vertex_states
) {
    return fermi_surface_from_model(
        std::static_pointer_cast<const HamiltonianModel>(std::move(model)),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_nearest_vertex_states
    );
}

FermiSurfaceResult fermi_surface_from_model(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_nearest_vertex_states
) {
    using namespace fermi_surface_detail;

    const auto total_start = Clock::now();
    if (!model) {
        throw std::runtime_error("fermi_surface: model must not be null");
    }
    if (model->ndim() < 1) {
        throw std::runtime_error("fermi_surface: dimension must be positive");
    }
    if (!(min_feature_size > 0.0) || !std::isfinite(min_feature_size)) {
        throw std::runtime_error("fermi_surface: min_feature_size must be positive");
    }
    if (margin < 0.0 || !std::isfinite(margin)) {
        throw std::runtime_error("fermi_surface: margin must be finite and non-negative");
    }

    FermiSurfaceResult result;
    result.ndim = model->ndim();
    result.ndof = model->ndof();
    result.has_states = return_nearest_vertex_states;
    result.min_feature_size = min_feature_size;

    auto geometry = make_fermi_geometry(model->ndim());
    auto cache = SpectraCache{};
    auto evaluator = VertexSpectraEvaluator(model);
    auto frontier = active_simplices(geometry);
    std::vector<core::SimplexId> terminal_surface_simplices;

    while (!frontier.empty()) {
        const auto missing = missing_vertices_for(geometry, cache, frontier);
        if (
            max_diagonalizations >= 0 &&
            fermi_surface_stats_.evaluated_vertices + missing.size() >
                static_cast<std::uint64_t>(max_diagonalizations)
        ) {
            result.converged = false;
            result.n_unresolved_simplices += static_cast<std::int64_t>(frontier.size());
            break;
        }

        evaluate_vertices(
            geometry,
            cache,
            evaluator,
            std::span<const core::VertexId>(missing.data(), missing.size())
        );
        const auto marking_start = Clock::now();
        auto marks = classify_frontier(
            geometry,
            cache,
            frontier,
            mu,
            min_feature_size,
            margin,
            tol
        );
        fermi_surface_stats_.marking_nanoseconds += nanoseconds_since(marking_start);
        result.n_safe_simplices += marks.safe;
        result.n_cut_simplices += marks.cut;
        result.n_feature_size_simplices += marks.feature_size;
        result.n_unresolved_simplices += marks.unresolved;
        terminal_surface_simplices.insert(
            terminal_surface_simplices.end(),
            marks.surface_terminal.begin(),
            marks.surface_terminal.end()
        );

        if (marks.marked.empty()) {
            frontier.clear();
            result.converged = result.n_unresolved_simplices == 0;
            break;
        }

        const auto previous_active = simplex_set(active_simplices(geometry));
        auto refined = marks.marked;

        const auto refinement_start = Clock::now();
        geometry.refine_active(refined, 1);
        fermi_surface_stats_.refinement_nanoseconds += nanoseconds_since(refinement_start);
        ++fermi_surface_stats_.refinement_calls;
        result.refinements += static_cast<std::int64_t>(refined.size());
        frontier = next_frontier(geometry, previous_active, marks.marked, refined);
    }

    const auto extraction_start = Clock::now();
    extract_surface(
        *model,
        geometry,
        cache,
        std::span<const core::SimplexId>(
            terminal_surface_simplices.data(),
            terminal_surface_simplices.size()
        ),
        mu,
        tol,
        return_nearest_vertex_states,
        result
    );
    fermi_surface_stats_.extraction_nanoseconds += nanoseconds_since(extraction_start);

    result.n_active_simplices = static_cast<std::int64_t>(geometry.simplices().n_active());
    result.n_active_vertices = static_cast<std::int64_t>(geometry.n_active_vertices());
    fermi_surface_stats_.total_nanoseconds += nanoseconds_since(total_start);
    return result;
}

}  // namespace lineartetrahedron
