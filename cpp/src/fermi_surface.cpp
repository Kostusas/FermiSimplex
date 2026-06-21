#include "lineartetrahedron/fermi_surface.h"

#include "lineartetrahedron/vertex_spectra.h"
#include "lineartetrahedron/weyl.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/root_mesh.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;

namespace {

using EigenvalueCache = core::VertexCache<std::vector<double>>;
using Clock = std::chrono::steady_clock;

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

void evaluate_missing_vertices(
    core::Geometry &geometry,
    EigenvalueCache &cache,
    const VertexEigenvaluesEvaluator &evaluator,
    const std::vector<core::SimplexId> &simplex_ids
) {
    const auto start = Clock::now();
    const auto missing = geometry.missing_vertices(
        std::span<const core::SimplexId>(simplex_ids.data(), simplex_ids.size()),
        cache,
        0
    );
    ++fermi_surface_stats_.vertex_evaluation_calls;
    fermi_surface_stats_.evaluated_vertices += missing.size();
    for (const auto vertex_id : missing) {
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

double eigenvalue_at(
    const EigenvalueCache &cache,
    core::VertexId vertex_id,
    size_t band
) {
    return cache.get(vertex_id)[band];
}

struct MarkResult {
    std::vector<core::SimplexId> marked;
    std::int64_t unresolved = 0;
};

using WeylDecisionCache = std::map<core::SimplexId, weyl::WeylSimplexDecision>;

MarkResult mark_simplices(
    const TightBindingModel &model,
    const core::Geometry &geometry,
    const EigenvalueCache &cache,
    double mu,
    double min_feature_size,
    bool use_weyl_bounds,
    double tol,
    WeylDecisionCache &decision_cache
) {
    MarkResult result;
    ++fermi_surface_stats_.marking_passes;
    for (const auto simplex_id : geometry.simplices().active_simplices()) {
        ++fermi_surface_stats_.active_simplex_visits;
        if (const auto cached = decision_cache.find(simplex_id);
            cached != decision_cache.end()) {
            ++fermi_surface_stats_.cached_decisions;
            if (cached->second.unresolved) {
                ++result.unresolved;
            }
            continue;
        }

        const auto edge_length = weyl::max_physical_edge_length(geometry, simplex_id);
        const auto refinable = edge_length > min_feature_size;
        ++fermi_surface_stats_.classified_simplices;
        const auto decision = weyl::classify_simplex_for_refinement(
            mu,
            model,
            geometry,
            simplex_id,
            refinable,
            use_weyl_bounds,
            tol,
            [&](core::VertexId vertex_id, size_t band_index) {
                return eigenvalue_at(cache, vertex_id, band_index);
            }
        );

        if (decision.should_refine) {
            result.marked.push_back(simplex_id);
            ++fermi_surface_stats_.marked_simplices;
        } else {
            decision_cache.emplace(simplex_id, decision);
            ++fermi_surface_stats_.terminal_cached_simplices;
            if (decision.unresolved) {
                ++result.unresolved;
            }
        }
    }
    return result;
}

std::vector<double> physical_vertex_point(
    const core::Geometry &geometry,
    core::VertexId vertex_id
) {
    const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
    return weyl::physical_point(std::span<const double>(reduced_point.data(), reduced_point.size()));
}

std::vector<double> interpolate_crossing(
    std::span<const double> left,
    std::span<const double> right,
    double left_value,
    double right_value
) {
    const auto t = -left_value / (right_value - left_value);
    std::vector<double> point(left.size());
    for (size_t axis = 0; axis < left.size(); ++axis) {
        point[axis] = (1.0 - t) * left[axis] + t * right[axis];
    }
    return point;
}

void append_product_cells(
    size_t negative_count,
    size_t positive_count,
    const std::vector<std::int64_t> &crossing_indices,
    FermiSurfaceResult &result
) {
    const auto local_cells =
        product_simplex_triangulation_cells(negative_count, positive_count);
    const auto ndim = negative_count + positive_count - 1;
    for (const auto local_index : local_cells) {
        result.cells.push_back(crossing_indices[static_cast<size_t>(local_index)]);
    }
    if (local_cells.size() % ndim != 0) {
        throw std::runtime_error("Fermi surface triangulation emitted invalid cell data");
    }
}

void extract_band_surface(
    const core::Geometry &geometry,
    const EigenvalueCache &cache,
    core::SimplexId simplex_id,
    size_t band,
    double mu,
    double tol,
    FermiSurfaceResult &result
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<size_t> negative;
    std::vector<size_t> positive;
    std::vector<double> signed_values(simplex.vertex_ids.size());
    std::vector<std::vector<double>> physical_points(simplex.vertex_ids.size());

    for (size_t local_vertex = 0; local_vertex < simplex.vertex_ids.size(); ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto signed_value = cache.get(vertex_id)[band] - mu;
        signed_values[local_vertex] = signed_value;
        physical_points[local_vertex] = physical_vertex_point(geometry, vertex_id);
        if (signed_value < -tol) {
            negative.push_back(local_vertex);
        } else if (signed_value > tol) {
            positive.push_back(local_vertex);
        }
    }

    if (negative.empty() || positive.empty()) {
        return;
    }

    std::vector<std::int64_t> crossing_indices(negative.size() * positive.size());
    for (size_t neg_index = 0; neg_index < negative.size(); ++neg_index) {
        for (size_t pos_index = 0; pos_index < positive.size(); ++pos_index) {
            const auto left = negative[neg_index];
            const auto right = positive[pos_index];
            const auto point = interpolate_crossing(
                std::span<const double>(physical_points[left].data(), physical_points[left].size()),
                std::span<const double>(physical_points[right].data(), physical_points[right].size()),
                signed_values[left],
                signed_values[right]
            );
            const auto point_index =
                static_cast<std::int64_t>(result.points.size() / result.ndim);
            result.points.insert(result.points.end(), point.begin(), point.end());
            crossing_indices[neg_index * positive.size() + pos_index] = point_index;
        }
    }

    append_product_cells(negative.size(), positive.size(), crossing_indices, result);
}

void extract_surface(
    const TightBindingModel &model,
    const core::Geometry &geometry,
    const EigenvalueCache &cache,
    double mu,
    double tol,
    FermiSurfaceResult &result
) {
    for (const auto simplex_id : geometry.simplices().active_simplices()) {
        for (size_t band = 0; band < model.ndof(); ++band) {
            extract_band_surface(geometry, cache, simplex_id, band, mu, tol, result);
        }
    }
}

void append_shuffle_cells(
    size_t negative_count,
    size_t positive_count,
    size_t negative_position,
    size_t positive_position,
    std::vector<std::int64_t> &path,
    std::vector<std::int64_t> &cells
) {
    if (negative_position + 1 == negative_count &&
        positive_position + 1 == positive_count) {
        cells.insert(cells.end(), path.begin(), path.end());
        return;
    }

    if (negative_position + 1 < negative_count) {
        const auto next = (negative_position + 1) * positive_count + positive_position;
        path.push_back(static_cast<std::int64_t>(next));
        append_shuffle_cells(
            negative_count,
            positive_count,
            negative_position + 1,
            positive_position,
            path,
            cells
        );
        path.pop_back();
    }
    if (positive_position + 1 < positive_count) {
        const auto next = negative_position * positive_count + positive_position + 1;
        path.push_back(static_cast<std::int64_t>(next));
        append_shuffle_cells(
            negative_count,
            positive_count,
            negative_position,
            positive_position + 1,
            path,
            cells
        );
        path.pop_back();
    }
}

}  // namespace

void reset_fermi_surface_stats() {
    fermi_surface_stats_ = FermiSurfaceStats{};
}

FermiSurfaceStats fermi_surface_stats() {
    return fermi_surface_stats_;
}

std::vector<std::int64_t> product_simplex_triangulation_cells(
    size_t negative_count,
    size_t positive_count
) {
    if (negative_count == 0 || positive_count == 0) {
        return {};
    }
    std::vector<std::int64_t> path{0};
    std::vector<std::int64_t> cells;
    append_shuffle_cells(negative_count, positive_count, 0, 0, path, cells);
    return cells;
}

FermiSurfaceResult fermi_surface(
    std::shared_ptr<TightBindingModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_refinements,
    bool use_weyl_bounds,
    double tol
) {
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

    auto geometry = make_fermi_geometry(model->ndim());
    auto cache = EigenvalueCache{};
    auto evaluator = VertexEigenvaluesEvaluator(model);
    FermiSurfaceResult result;
    result.ndim = model->ndim();
    result.min_feature_size = min_feature_size;

    auto decision_cache = WeylDecisionCache{};
    auto remaining = max_refinements;
    while (true) {
        if (remaining == 0) {
            result.converged = false;
            result.n_unresolved_simplices =
                static_cast<std::int64_t>(geometry.simplices().n_active());
            break;
        }

        const auto active = active_simplices(geometry);
        evaluate_missing_vertices(geometry, cache, evaluator, active);
        const auto marking_start = Clock::now();
        auto marks = mark_simplices(
            *model,
            geometry,
            cache,
            mu,
            min_feature_size,
            use_weyl_bounds,
            tol,
            decision_cache
        );
        fermi_surface_stats_.marking_nanoseconds += nanoseconds_since(marking_start);
        if (marks.marked.empty()) {
            result.converged = marks.unresolved == 0;
            result.n_unresolved_simplices = marks.unresolved;
            break;
        }
        if (remaining > 0 && static_cast<std::int64_t>(marks.marked.size()) > remaining) {
            marks.marked.resize(static_cast<size_t>(remaining));
        }
        const auto refined_count = static_cast<std::int64_t>(marks.marked.size());
        const auto refinement_start = Clock::now();
        geometry.refine_active(marks.marked, 1);
        fermi_surface_stats_.refinement_nanoseconds += nanoseconds_since(refinement_start);
        ++fermi_surface_stats_.refinement_calls;
        result.refinements += refined_count;
        if (remaining > 0) {
            remaining -= refined_count;
        }
    }

    const auto active = active_simplices(geometry);
    evaluate_missing_vertices(geometry, cache, evaluator, active);
    const auto extraction_start = Clock::now();
    extract_surface(*model, geometry, cache, mu, tol, result);
    fermi_surface_stats_.extraction_nanoseconds += nanoseconds_since(extraction_start);
    result.n_active_simplices = static_cast<std::int64_t>(geometry.simplices().n_active());
    result.n_active_vertices = static_cast<std::int64_t>(geometry.n_active_vertices());
    fermi_surface_stats_.total_nanoseconds += nanoseconds_since(total_start);
    return result;
}

}  // namespace lineartetrahedron
