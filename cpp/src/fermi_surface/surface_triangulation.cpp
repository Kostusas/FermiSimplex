#include "fermi_surface/surface_triangulation.h"

#include <cstddef>

namespace fermisimplex {
namespace {

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

std::vector<std::int64_t> product_simplex_triangulation_cells(
    size_t negative_count,
    size_t positive_count
) {
    if (negative_count == 0 || positive_count == 0) {
        return {};
    }
    std::vector<std::int64_t> path{0};
    std::vector<std::int64_t> cells;
    append_shuffle_cells(
        negative_count,
        positive_count,
        0,
        0,
        path,
        cells
    );
    return cells;
}

}  // namespace fermisimplex
