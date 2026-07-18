#include "fermi_surface/surface_extraction.h"

#include "fermi_surface/surface_triangulation.h"

#include <algorithm>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {
namespace {

using PointKey = std::tuple<size_t, core::VertexId, core::VertexId>;
using CellKey = std::pair<size_t, std::vector<std::int64_t>>;

double eigenvalue_at(
    const EigensystemCache &cache,
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

class SurfaceBuilder {
public:
    explicit SurfaceBuilder(FermiSurfaceResult &result) : result_(result) {}

    std::int64_t vertex_point(
        size_t band,
        const core::Geometry &geometry,
        core::VertexId vertex_id
    ) {
        return point(
            PointKey{band, vertex_id, vertex_id},
            reduced_vertex_point(geometry, vertex_id)
        );
    }

    std::int64_t edge_crossing(
        size_t band,
        const core::Geometry &geometry,
        core::VertexId left_id,
        core::VertexId right_id,
        double left_value,
        double right_value
    ) {
        const auto key = PointKey{
            band,
            std::min(left_id, right_id),
            std::max(left_id, right_id),
        };
        const auto left = reduced_vertex_point(geometry, left_id);
        const auto right = reduced_vertex_point(geometry, right_id);
        return point(
            key,
            interpolate_crossing(left, right, left_value, right_value)
        );
    }

    void append_cell(size_t band, std::vector<std::int64_t> indices) {
        if (indices.size() != result_.ndim) {
            throw std::runtime_error("Fermi surface triangulation emitted invalid cell data");
        }

        auto canonical = indices;
        std::sort(canonical.begin(), canonical.end());
        if (!cells_.insert(CellKey{band, std::move(canonical)}).second) {
            return;
        }
        used_points_.insert(indices.begin(), indices.end());
        result_.cells.insert(result_.cells.end(), indices.begin(), indices.end());
        result_.cell_bands.push_back(static_cast<std::int64_t>(band));
    }

    void finish() {
        if (used_points_.size() != points_.size()) {
            // A lower-dimensional contact was found, but the result format
            // can represent only codimension-one cells.
            result_.coverage_certified = false;
        }
    }

private:
    std::int64_t point(const PointKey &key, std::vector<double> coordinates) {
        if (const auto existing = points_.find(key); existing != points_.end()) {
            return existing->second;
        }

        const auto index = static_cast<std::int64_t>(result_.points.size() / result_.ndim);
        result_.points.insert(
            result_.points.end(),
            coordinates.begin(),
            coordinates.end()
        );
        points_.emplace(key, index);
        return index;
    }

    FermiSurfaceResult &result_;
    std::map<PointKey, std::int64_t> points_;
    std::set<CellKey> cells_;
    std::set<std::int64_t> used_points_;
};

void append_join_cells(
    size_t band,
    std::span<const std::int64_t> on_level_indices,
    size_t negative_count,
    size_t positive_count,
    const std::vector<std::int64_t> &crossing_indices,
    SurfaceBuilder &builder
) {
    const auto local_cells =
        product_simplex_triangulation_cells(negative_count, positive_count);
    const auto product_cell_size = negative_count + positive_count - 1;
    if (local_cells.size() % product_cell_size != 0) {
        throw std::runtime_error("Fermi surface triangulation emitted invalid cell data");
    }
    for (size_t offset = 0; offset < local_cells.size(); offset += product_cell_size) {
        auto cell = std::vector<std::int64_t>(
            on_level_indices.begin(),
            on_level_indices.end()
        );
        cell.reserve(on_level_indices.size() + product_cell_size);
        for (size_t index = 0; index < product_cell_size; ++index) {
            cell.push_back(
                crossing_indices[static_cast<size_t>(local_cells[offset + index])]
            );
        }
        builder.append_cell(band, std::move(cell));
    }
}

void extract_band_surface(
    const core::Geometry &geometry,
    const EigensystemCache &cache,
    core::SimplexId simplex_id,
    size_t band,
    double mu,
    double tol,
    SurfaceBuilder &builder,
    FermiSurfaceResult &result
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<size_t> negative;
    std::vector<size_t> positive;
    std::vector<size_t> on_level;
    std::vector<double> signed_values(simplex.vertex_ids.size());

    for (size_t local_vertex = 0; local_vertex < simplex.vertex_ids.size(); ++local_vertex) {
        const auto vertex_id = simplex.vertex_ids[local_vertex];
        const auto signed_value = eigenvalue_at(cache, vertex_id, band) - mu;
        signed_values[local_vertex] = signed_value;
        if (signed_value < -tol) {
            negative.push_back(local_vertex);
        } else if (signed_value > tol) {
            positive.push_back(local_vertex);
        } else {
            on_level.push_back(local_vertex);
        }
    }

    if (on_level.size() == simplex.vertex_ids.size()) {
        // A flat band at mu fills the simplex rather than defining a
        // codimension-one surface, so no surface mesh can certify it.
        result.coverage_certified = false;
        return;
    }

    std::vector<std::int64_t> on_level_indices;
    on_level_indices.reserve(on_level.size());
    for (const auto local_vertex : on_level) {
        on_level_indices.push_back(
            builder.vertex_point(band, geometry, simplex.vertex_ids[local_vertex])
        );
    }

    if (negative.empty() || positive.empty()) {
        // The zero set is the on-level face itself. It is a surface cell only
        // when that face has codimension one; lower-dimensional tangencies do
        // not have a representation in the fixed-width cell array.
        if (on_level_indices.size() == result.ndim) {
            builder.append_cell(band, std::move(on_level_indices));
        }
        return;
    }

    std::vector<std::int64_t> crossing_indices(negative.size() * positive.size());
    for (size_t neg_index = 0; neg_index < negative.size(); ++neg_index) {
        for (size_t pos_index = 0; pos_index < positive.size(); ++pos_index) {
            const auto left = negative[neg_index];
            const auto right = positive[pos_index];
            crossing_indices[neg_index * positive.size() + pos_index] =
                builder.edge_crossing(
                    band,
                    geometry,
                    simplex.vertex_ids[left],
                    simplex.vertex_ids[right],
                    signed_values[left],
                    signed_values[right]
                );
        }
    }

    append_join_cells(
        band,
        on_level_indices,
        negative.size(),
        positive.size(),
        crossing_indices,
        builder
    );
}

}  // namespace

void extract_terminal_surface(
    const SpectralMesh &mesh,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    FermiSurfaceResult &result
) {
    auto builder = SurfaceBuilder(result);
    for (const auto simplex_id : simplex_ids) {
        for (size_t band = 0; band < mesh.ndof(); ++band) {
            extract_band_surface(
                mesh.geometry(),
                mesh.eigensystems(),
                simplex_id,
                band,
                mu,
                mesh.tolerance(),
                builder,
                result
            );
        }
    }
    builder.finish();
}

}  // namespace lineartetrahedron::fermi_surface_detail
