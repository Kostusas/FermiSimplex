#pragma once

#include <adaptivesimplex/core/geometry.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace fermisimplex::detail {

using adaptivesimplex::core::Geometry;
using adaptivesimplex::core::SimplexId;
using adaptivesimplex::core::VertexId;

struct EvaluatedSimplex {
    SimplexId simplex_id;
    SimplexId active_ancestor_id;

    bool operator==(const EvaluatedSimplex &) const = default;
};

// Select a complete evaluated covering from existing active/preview trees.
// Cache must supply const contains(VertexId); no value evaluation is performed.
template <class Cache>
std::vector<EvaluatedSimplex> evaluated_partition(const Geometry &geometry, const Cache &cache) {
    std::vector<EvaluatedSimplex> result;
    const auto select = [&](const auto &self, SimplexId id, SimplexId active_id) -> bool {
        const auto &simplex = geometry.simplices().simplex(id);
        if (!std::all_of(simplex.vertex_ids.begin(), simplex.vertex_ids.end(),
                         [&](VertexId vertex) { return cache.contains(vertex); })) {
            return false;
        }

        if (simplex.children) {
            const auto offset = result.size();
            const auto children = simplex.children->simplex_ids;
            if (self(self, children[0], active_id) && self(self, children[1], active_id)) {
                return true;
            }
            result.resize(offset);
        }
        result.push_back({id, active_id});
        return true;
    };

    for (const auto id : geometry.simplices().active_simplices()) {
        if (!select(select, id, id)) {
            throw std::runtime_error("evaluated_partition: no complete evaluated covering");
        }
    }
    return result;
}

}  // namespace fermisimplex::detail
