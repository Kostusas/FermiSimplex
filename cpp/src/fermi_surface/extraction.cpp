#include "internal.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {

namespace {

double eigenvalue_at(
    const SpectraCache &cache,
    core::VertexId vertex_id,
    size_t band
) {
    return cache.get(vertex_id).eigenvalues[band];
}

std::vector<double> reduced_vertex_point(
    const core::Geometry &geometry,
    core::VertexId vertex_id
) {
    const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
    return std::vector<double>(reduced_point.begin(), reduced_point.end());
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

void append_nearest_vertex_state(
    const SpectraCache &cache,
    core::VertexId left_vertex_id,
    core::VertexId right_vertex_id,
    size_t band,
    double mu,
    FermiSurfaceResult &result
) {
    const auto &left_spectra = cache.get(left_vertex_id);
    const auto &right_spectra = cache.get(right_vertex_id);
    const auto left_distance = std::abs(left_spectra.eigenvalues[band] - mu);
    const auto right_distance = std::abs(right_spectra.eigenvalues[band] - mu);
    const auto use_left = left_distance <= right_distance;
    const auto &spectra = use_left ? left_spectra : right_spectra;

    result.state_band_indices.push_back(static_cast<std::int64_t>(band));
    result.state_eigenvalues.push_back(spectra.eigenvalues[band]);
    for (size_t row = 0; row < result.ndof; ++row) {
        result.state_eigenvectors.push_back(spectra.eigenvectors[band * result.ndof + row]);
    }
}

void extract_band_surface(
    const core::Geometry &geometry,
    const SpectraCache &cache,
    core::SimplexId simplex_id,
    size_t band,
    double mu,
    double tol,
    bool return_states,
    FermiSurfaceResult &result
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<size_t> negative;
    std::vector<size_t> positive;
    std::vector<double> signed_values(simplex.vertex_ids.size());
    std::vector<std::vector<double>> reduced_points(simplex.vertex_ids.size());

    for (size_t local_vertex = 0; local_vertex < simplex.vertex_ids.size(); ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto signed_value = eigenvalue_at(cache, vertex_id, band) - mu;
        signed_values[local_vertex] = signed_value;
        reduced_points[local_vertex] = reduced_vertex_point(geometry, vertex_id);
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
                std::span<const double>(reduced_points[left].data(), reduced_points[left].size()),
                std::span<const double>(reduced_points[right].data(), reduced_points[right].size()),
                signed_values[left],
                signed_values[right]
            );
            const auto point_index =
                static_cast<std::int64_t>(result.points.size() / result.ndim);
            result.points.insert(result.points.end(), point.begin(), point.end());
            if (return_states) {
                append_nearest_vertex_state(
                    cache,
                    simplex.vertex_ids[left],
                    simplex.vertex_ids[right],
                    band,
                    mu,
                    result
                );
            }
            crossing_indices[neg_index * positive.size() + pos_index] = point_index;
        }
    }

    append_product_cells(negative.size(), positive.size(), crossing_indices, result);
}

void extract_surface_impl(
    const HamiltonianModel &model,
    const core::Geometry &geometry,
    const SpectraCache &cache,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    double tol,
    bool return_states,
    FermiSurfaceResult &result
) {
    for (const auto simplex_id : simplex_ids) {
        for (size_t band = 0; band < model.ndof(); ++band) {
            extract_band_surface(
                geometry,
                cache,
                simplex_id,
                band,
                mu,
                tol,
                return_states,
                result
            );
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

void extract_surface(
    const HamiltonianModel &model,
    const core::Geometry &geometry,
    const SpectraCache &cache,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    double tol,
    bool return_states,
    FermiSurfaceResult &result
) {
    extract_surface_impl(
        model,
        geometry,
        cache,
        simplex_ids,
        mu,
        tol,
        return_states,
        result
    );
}

}  // namespace lineartetrahedron::fermi_surface_detail

namespace lineartetrahedron {

std::vector<std::int64_t> product_simplex_triangulation_cells(
    size_t negative_count,
    size_t positive_count
) {
    if (negative_count == 0 || positive_count == 0) {
        return {};
    }
    std::vector<std::int64_t> path{0};
    std::vector<std::int64_t> cells;
    fermi_surface_detail::append_shuffle_cells(
        negative_count,
        positive_count,
        0,
        0,
        path,
        cells
    );
    return cells;
}

}  // namespace lineartetrahedron
