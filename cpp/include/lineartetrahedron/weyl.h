#pragma once

#include "lineartetrahedron/tight_binding.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace lineartetrahedron::weyl {

namespace core = adaptivesimplex::core;

struct SimplexGeometry {
    std::vector<double> center_physical;
    std::vector<double> coordinate_radii;
    std::vector<double> vertex_offsets;
    size_t nvertices = 0;
};

struct BandVertexRange {
    double min = 0.0;
    double max = 0.0;
    bool same_side = false;
    bool on_level = false;
};

struct WeylSimplexDecision {
    bool should_refine = false;
    bool unresolved = false;
};

inline std::vector<double> simplex_center(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<double> center(geometry.ndim(), 0.0);
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        for (size_t axis = 0; axis < center.size(); ++axis) {
            center[axis] += point[axis];
        }
    }
    const auto scale = 1.0 / static_cast<double>(simplex.vertex_ids.size());
    for (auto &coordinate : center) {
        coordinate *= scale;
    }
    return center;
}

inline std::vector<double> physical_point(std::span<const double> reduced_point) {
    std::vector<double> result(reduced_point.size());
    for (size_t axis = 0; axis < reduced_point.size(); ++axis) {
        result[axis] = 2.0 * kPi * reduced_point[axis] - kPi;
    }
    return result;
}

inline SimplexGeometry make_simplex_geometry(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto center_reduced = simplex_center(geometry, simplex_id);
    SimplexGeometry result;
    result.nvertices = simplex.vertex_ids.size();
    result.center_physical = physical_point(
        std::span<const double>(center_reduced.data(), center_reduced.size())
    );
    result.coordinate_radii.assign(geometry.ndim(), 0.0);
    result.vertex_offsets.assign(result.nvertices * geometry.ndim(), 0.0);

    for (size_t local_vertex = 0; local_vertex < simplex.vertex_ids.size(); ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        for (size_t axis = 0; axis < point.size(); ++axis) {
            const auto offset = 2.0 * kPi * std::abs(point[axis] - center_reduced[axis]);
            result.vertex_offsets[local_vertex * geometry.ndim() + axis] = offset;
            result.coordinate_radii[axis] = std::max(result.coordinate_radii[axis], offset);
        }
    }
    return result;
}

inline double max_physical_edge_length(
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
                const auto delta = 2.0 * kPi * (left_point[axis] - right_point[axis]);
                squared += delta * delta;
            }
            result = std::max(result, std::sqrt(squared));
        }
    }
    return result;
}

inline double max_weighted_vertex_offset(
    const SimplexGeometry &geometry,
    std::span<const double> axis_weights,
    size_t ndim
) {
    auto result = 0.0;
    for (size_t local_vertex = 0; local_vertex < geometry.nvertices; ++local_vertex) {
        auto distance = 0.0;
        for (size_t axis = 0; axis < ndim; ++axis) {
            distance +=
                axis_weights[axis] * geometry.vertex_offsets[local_vertex * ndim + axis];
        }
        result = std::max(result, distance);
    }
    return result;
}

inline bool interval_contains(double value, double lower, double upper, double tol) {
    return value >= lower - tol && value <= upper + tol;
}

template <class EigenvalueAt>
BandVertexRange band_vertex_range(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    size_t band,
    double tol,
    EigenvalueAt eigenvalue_at
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto all_above = true;
    auto all_below = true;
    auto on_level = false;
    auto minimum = std::numeric_limits<double>::infinity();
    auto maximum = -std::numeric_limits<double>::infinity();
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto energy = eigenvalue_at(vertex_id, band);
        minimum = std::min(minimum, energy);
        maximum = std::max(maximum, energy);
        on_level = on_level || std::abs(energy - mu) <= tol;
        all_above = all_above && energy > mu + tol;
        all_below = all_below && energy < mu - tol;
    }
    return BandVertexRange{
        .min = minimum,
        .max = maximum,
        .same_side = all_above || all_below,
        .on_level = on_level,
    };
}

inline double local_hessian_delta(
    const TightBindingModel &model,
    const SimplexGeometry &geometry
) {
    constexpr auto kLanczosHessianToleranceFraction = 0.25;
    constexpr auto kLanczosNumericalFloorMultiplier = 64.0;
    std::vector<double> axis_weights(model.ndim(), 0.0);
    for (size_t axis = 0; axis < model.ndim(); ++axis) {
        auto hessian_correction = 0.0;
        for (size_t coordinate = 0; coordinate < model.ndim(); ++coordinate) {
            hessian_correction +=
                model.hessian_bound(axis, coordinate) * geometry.coordinate_radii[coordinate];
        }
        const auto uncertainty_target = std::max(
            kLanczosNumericalFloorMultiplier * std::numeric_limits<double>::epsilon() *
                std::max(model.global_derivative_bounds()[axis], 1.0),
            kLanczosHessianToleranceFraction * hessian_correction
        );
        axis_weights[axis] =
            model.derivative_spectral_norm(
                geometry.center_physical.data(),
                axis,
                uncertainty_target
            ) +
            hessian_correction;
    }
    return 2.0 * max_weighted_vertex_offset(
        geometry,
        std::span<const double>(axis_weights.data(), axis_weights.size()),
        model.ndim()
    );
}

inline bool hessian_marker_applies_to_range(
    double mu,
    const TightBindingModel &model,
    const BandVertexRange &range,
    const SimplexGeometry &weyl_geometry,
    double global_delta,
    double &local_delta,
    bool &has_local_delta,
    double tol
) {
    if (!range.same_side) {
        return false;
    }

    if (!interval_contains(mu, range.min - global_delta, range.max + global_delta, tol)) {
        return false;
    }

    if (!has_local_delta) {
        local_delta = local_hessian_delta(model, weyl_geometry);
        has_local_delta = true;
    }
    return interval_contains(mu, range.min - local_delta, range.max + local_delta, tol);
}

inline bool has_straddling_vertices(const BandVertexRange &range) {
    return !range.same_side && !range.on_level;
}

template <class EigenvalueAt>
WeylSimplexDecision classify_simplex_for_refinement(
    double mu,
    const TightBindingModel &model,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    bool refinable,
    bool use_weyl_bounds,
    double tol,
    EigenvalueAt eigenvalue_at
) {
    auto decision = WeylSimplexDecision{};
    SimplexGeometry weyl_geometry;
    auto global_delta = 0.0;
    auto local_delta = 0.0;
    auto has_weyl_geometry = false;
    auto has_local_delta = false;

    const auto ensure_weyl_geometry = [&]() {
        if (has_weyl_geometry) {
            return;
        }
        weyl_geometry = make_simplex_geometry(geometry, simplex_id);
        global_delta = 2.0 * max_weighted_vertex_offset(
            weyl_geometry,
            model.global_derivative_bounds(),
            model.ndim()
        );
        has_weyl_geometry = true;
    };

    for (size_t band = 0; band < model.ndof(); ++band) {
        const auto range = band_vertex_range(
            mu,
            geometry,
            simplex_id,
            band,
            tol,
            eigenvalue_at
        );

        if (range.on_level || has_straddling_vertices(range)) {
            if (refinable) {
                decision.should_refine = true;
                return decision;
            }
            if (range.on_level) {
                decision.unresolved = true;
            }
            continue;
        }

        if (!use_weyl_bounds || !range.same_side) {
            continue;
        }

        ensure_weyl_geometry();
        if (!hessian_marker_applies_to_range(
                mu,
                model,
                range,
                weyl_geometry,
                global_delta,
                local_delta,
                has_local_delta,
                tol
            )) {
            continue;
        }

        if (refinable) {
            decision.should_refine = true;
            return decision;
        }
        decision.unresolved = true;
    }

    return decision;
}

template <class EigenvalueAt>
bool simplex_has_weyl_marker(
    double mu,
    const TightBindingModel &model,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    double tol,
    EigenvalueAt eigenvalue_at
) {
    SimplexGeometry weyl_geometry = make_simplex_geometry(geometry, simplex_id);
    const auto global_delta = 2.0 * max_weighted_vertex_offset(
        weyl_geometry,
        model.global_derivative_bounds(),
        model.ndim()
    );
    auto local_delta = 0.0;
    auto has_local_delta = false;

    for (size_t band = 0; band < model.ndof(); ++band) {
        const auto range = band_vertex_range(
            mu,
            geometry,
            simplex_id,
            band,
            tol,
            eigenvalue_at
        );
        if (hessian_marker_applies_to_range(
                mu,
                model,
                range,
                weyl_geometry,
                global_delta,
                local_delta,
                has_local_delta,
                tol
            )) {
            return true;
        }
    }

    return false;
}

}  // namespace lineartetrahedron::weyl
