#pragma once

#include "adaptivesimplex/core/simplex_table.h"
#include "adaptivesimplex/core/vertex_table.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace adaptivesimplex::core {

class Geometry {
public:
    Geometry(size_t ndim, VertexTable vertices, SimplexTable simplices)
        : ndim_(ndim),
          vertices_(std::move(vertices)),
          simplices_(std::move(simplices)) {}

    size_t ndim() const noexcept { return ndim_; }
    const VertexTable &vertices() const noexcept { return vertices_; }
    const SimplexTable &simplices() const noexcept { return simplices_; }
    size_t n_active_vertices() const {
        std::unordered_set<VertexId> used;
        for (const auto simplex_id : simplices_.active_simplices()) {
            const auto &vertex_ids = simplices_.simplex(simplex_id).vertex_ids;
            used.insert(vertex_ids.begin(), vertex_ids.end());
        }
        return used.size();
    }

    std::vector<SimplexId> preview_active(SimplexId simplex_id, std::uint32_t depth) {
        if (!simplices_.is_active(simplex_id)) {
            throw std::runtime_error("Geometry: simplex is not active");
        }

        std::vector<SimplexId> result;
        subdivide(simplex_id, depth, result);
        return result;
    }

    std::vector<VertexId> required_vertices(
        std::span<const SimplexId> simplex_ids,
        std::uint32_t preview_depth
    ) {
        std::unordered_set<VertexId> seen;
        std::vector<VertexId> result;

        const auto add_simplex_vertices = [&](SimplexId simplex_id) {
            const auto &vertex_ids = simplices_.simplex(simplex_id).vertex_ids;
            for (const auto vertex_id : vertex_ids) {
                if (seen.insert(vertex_id).second) {
                    result.push_back(vertex_id);
                }
            }
        };

        for (const auto simplex_id : simplex_ids) {
            if (!simplices_.is_active(simplex_id)) {
                throw std::runtime_error("Geometry: simplex is not active");
            }
            add_simplex_vertices(simplex_id);
            for (const auto preview_id : preview_active(simplex_id, preview_depth)) {
                add_simplex_vertices(preview_id);
            }
        }

        return result;
    }

    template <class Cache>
    std::vector<VertexId> missing_vertices(
        std::span<const SimplexId> simplex_ids,
        const Cache &cache,
        std::uint32_t preview_depth
    ) {
        std::vector<VertexId> missing;
        for (const auto vertex_id : required_vertices(simplex_ids, preview_depth)) {
            if (!cache.contains(vertex_id)) {
                missing.push_back(vertex_id);
            }
        }
        return missing;
    }
    std::vector<SimplexId> refine_active(
        const std::vector<SimplexId> &simplex_ids,
        std::uint32_t depth = 1
    ) {
        std::unordered_set<SimplexId> selected(simplex_ids.begin(), simplex_ids.end());
        if (selected.empty()) {
            return {};
        }

        const auto subdivision_depth = std::max<std::uint32_t>(depth, 1);
        const auto current_active = simplices_.active_simplices();

        if (!simplices_.all_active(simplex_ids)) {
            throw std::runtime_error("Geometry: selected simplex is not active");
        }

        std::vector<SimplexId> active_simplices;
        active_simplices.reserve(current_active.size() * 2);
        std::vector<SimplexId> replacements;

        for (const auto simplex_id : current_active) {
            if (!selected.contains(simplex_id)) {
                active_simplices.push_back(simplex_id);
                continue;
            }

            const auto offset = replacements.size();
            subdivide(simplex_id, subdivision_depth, replacements);
            active_simplices.insert(
                active_simplices.end(),
                replacements.begin() + static_cast<std::ptrdiff_t>(offset),
                replacements.end()
            );
        }

        simplices_.replace_active_simplices(std::move(active_simplices));
        return replacements;
    }

private:
    std::array<SimplexId, 2> bisect(SimplexId simplex_id) {
        const auto &existing = simplices_.simplex(simplex_id);
        if (existing.children.has_value()) {
            return existing.children->simplex_ids;
        }

        const auto parent_vertex_ids = existing.vertex_ids;
        const auto child_volume = 0.5 * existing.volume;
        const auto split_edge = longest_edge(existing);
        const auto midpoint_id = vertices_.midpoint(
            parent_vertex_ids[split_edge[0]],
            parent_vertex_ids[split_edge[1]]
        );

        auto child_a = parent_vertex_ids;
        auto child_b = parent_vertex_ids;
        child_a[split_edge[0]] = midpoint_id;
        child_b[split_edge[1]] = midpoint_id;

        const auto child_a_id = simplices_.add(child_a, child_volume);
        const auto child_b_id = simplices_.add(child_b, child_volume);

        auto &parent = simplices_.mutable_simplex(simplex_id);
        parent.children = SimplexChildren{
            split_edge,
            midpoint_id,
            std::array<SimplexId, 2>{child_a_id, child_b_id},
        };
        return parent.children->simplex_ids;
    }

    void subdivide(SimplexId simplex_id, std::uint32_t depth, std::vector<SimplexId> &out) {
        if (depth == 0) {
            out.push_back(simplex_id);
            return;
        }

        const auto children = bisect(simplex_id);
        if (depth == 1) {
            out.insert(out.end(), children.begin(), children.end());
            return;
        }
        for (const auto child_id : children) {
            subdivide(child_id, depth - 1, out);
        }
    }

    std::array<size_t, 2> longest_edge(const Simplex &simplex) const {
        if (simplex.vertex_ids.size() < 2) {
            throw std::runtime_error("Geometry: simplex has no edge to bisect");
        }

        std::array<size_t, 2> best{0, 1};
        auto best_length = -1.0L;
        for (auto i = size_t{0}; i < simplex.vertex_ids.size(); ++i) {
            const auto &left = vertices_.dyadic_vertex(simplex.vertex_ids[i]);
            for (auto j = i + 1; j < simplex.vertex_ids.size(); ++j) {
                const auto &right = vertices_.dyadic_vertex(simplex.vertex_ids[j]);
                const auto length = left.squared_distance_to(right);
                if (length > best_length) {
                    best_length = length;
                    best = {i, j};
                }
            }
        }
        return best;
    }

    size_t ndim_ = 0;
    VertexTable vertices_;
    SimplexTable simplices_;
};

}  // namespace adaptivesimplex::core
