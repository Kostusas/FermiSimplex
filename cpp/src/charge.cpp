#include "lineartetrahedron/charge.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

double occupied_fraction_derivative(
    std::span<const double> energies,
    double mu
) {
    const size_t dimension = energies.size() - 1;
    double result = 0.0;
    for (size_t i = 0; i < energies.size(); ++i) {
        const double distance = mu - energies[i];
        if (distance <= 0.0) {
            continue;
        }
        double denominator = 1.0;
        for (size_t j = 0; j < energies.size(); ++j) {
            if (j != i) {
                denominator *= energies[j] - energies[i];
            }
        }
        result += std::pow(distance, static_cast<double>(dimension - 1)) / denominator;
    }
    return static_cast<double>(dimension) * result;
}

bool strictly_ordered(std::span<const double> energies, double tol) {
    for (size_t index = 1; index < energies.size(); ++index) {
        if (energies[index] - energies[index - 1] <= tol) {
            return false;
        }
    }
    return true;
}

std::vector<double> sorted_band_energies(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t band
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<double> energies;
    energies.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        energies.push_back(cache.get(vertex_id).eigenvalues[band]);
    }
    std::stable_sort(energies.begin(), energies.end());
    return energies;
}

std::vector<double> simplex_center(
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

struct WeylSimplexGeometry {
    std::vector<double> center_physical;
    std::vector<double> coordinate_radii;
    std::vector<double> vertex_offsets;
    size_t nvertices = 0;
};

WeylSimplexGeometry make_weyl_geometry(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto center_reduced = simplex_center(geometry, simplex_id);
    WeylSimplexGeometry result;
    result.nvertices = simplex.vertex_ids.size();
    result.center_physical.resize(geometry.ndim());
    result.coordinate_radii.assign(geometry.ndim(), 0.0);
    result.vertex_offsets.assign(result.nvertices * geometry.ndim(), 0.0);

    for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
        result.center_physical[axis] = 2.0 * kPi * center_reduced[axis] - kPi;
    }

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

double max_weighted_vertex_offset(
    const WeylSimplexGeometry &geometry,
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

bool interval_contains(double value, double lower, double upper, double tol) {
    return value >= lower - tol && value <= upper + tol;
}

struct BandVertexRange {
    double min = 0.0;
    double max = 0.0;
    bool same_side = false;
};

BandVertexRange band_vertex_range(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t band,
    double tol
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto all_above = true;
    auto all_below = true;
    auto minimum = std::numeric_limits<double>::infinity();
    auto maximum = -std::numeric_limits<double>::infinity();
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto energy = cache.get(vertex_id).eigenvalues[band];
        minimum = std::min(minimum, energy);
        maximum = std::max(maximum, energy);
        all_above = all_above && energy > mu + tol;
        all_below = all_below && energy < mu - tol;
    }
    return BandVertexRange{
        .min = minimum,
        .max = maximum,
        .same_side = all_above || all_below,
    };
}

double local_hessian_weyl_delta(
    const IntegrationWorkspace &workspace,
    const WeylSimplexGeometry &geometry
) {
    std::vector<double> axis_weights(workspace.ndim(), 0.0);
    for (size_t axis = 0; axis < workspace.ndim(); ++axis) {
        auto bound = workspace.derivative_spectral_norm(
            std::span<const double>(
                geometry.center_physical.data(),
                geometry.center_physical.size()
            ),
            axis
        );
        for (size_t coordinate = 0; coordinate < workspace.ndim(); ++coordinate) {
            bound +=
                workspace.hessian_bound(axis, coordinate) *
                geometry.coordinate_radii[coordinate];
        }
        axis_weights[axis] = bound;
    }
    return 2.0 * max_weighted_vertex_offset(
        geometry,
        std::span<const double>(axis_weights.data(), axis_weights.size()),
        workspace.ndim()
    );
}

bool weyl_marker_applies(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t band,
    const WeylSimplexGeometry &weyl_geometry,
    double global_delta,
    double &local_delta,
    bool &has_local_delta
) {
    const auto range =
        band_vertex_range(mu, geometry, simplex_id, cache, band, workspace.tol());
    if (!range.same_side) {
        return false;
    }

    if (!interval_contains(
            mu,
            range.min - global_delta,
            range.max + global_delta,
            workspace.tol()
        )) {
        return false;
    }

    if (!has_local_delta) {
        local_delta = local_hessian_weyl_delta(workspace, weyl_geometry);
        has_local_delta = true;
    }
    return interval_contains(
        mu,
        range.min - local_delta,
        range.max + local_delta,
        workspace.tol()
    );
}

}  // namespace

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    double weyl_indicator_error
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    ChargeValue result;
    const auto use_weyl_marker =
        weyl_indicator_error > 0.0 && workspace.reduced_lipschitz_bound() > 0.0;
    WeylSimplexGeometry weyl_geometry;
    auto global_delta = 0.0;
    auto local_delta = 0.0;
    auto has_local_delta = false;
    if (use_weyl_marker) {
        weyl_geometry = make_weyl_geometry(geometry, simplex_id);
        const auto global_bounds = workspace.global_derivative_bounds();
        global_delta = 2.0 * max_weighted_vertex_offset(
            weyl_geometry,
            global_bounds,
            workspace.ndim()
        );
    }

    for (size_t band = 0; band < workspace.ndof(); ++band) {
        if (use_weyl_marker &&
            weyl_marker_applies(
                mu,
                workspace,
                geometry,
                simplex_id,
                cache,
                band,
                weyl_geometry,
                global_delta,
                local_delta,
                has_local_delta
            )) {
            result.weyl_indicator_error = weyl_indicator_error;
        }

        const auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{.level = mu, .level_tolerance = workspace.tol()}
        );

        if (moments.kind == cut::SimplexCutKind::on_level) {
            result.charge += 0.5 * simplex.volume;
            continue;
        }

        result.charge += moments.volume;
        if (moments.kind != cut::SimplexCutKind::partial) {
            continue;
        }

        const auto energies = sorted_band_energies(geometry, simplex_id, cache, band);
        const auto energy_span = std::span<const double>(energies.data(), energies.size());
        if (strictly_ordered(energy_span, workspace.tol())) {
            result.derivative +=
                simplex.volume * occupied_fraction_derivative(energy_span, mu);
        }
    }
    return result;
}

}  // namespace lineartetrahedron
