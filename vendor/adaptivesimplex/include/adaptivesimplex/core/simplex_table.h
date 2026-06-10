#pragma once

#include "adaptivesimplex/core/types.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adaptivesimplex::core {

struct SimplexChildren {
    std::array<size_t, 2> split_edge;
    VertexId midpoint;
    std::array<SimplexId, 2> simplex_ids;
};

struct Simplex {
    std::vector<VertexId> vertex_ids;
    double volume = 0.0;
    std::optional<SimplexChildren> children;
};

class SimplexTable {
public:
    size_t size() const noexcept { return simplices_.size(); }
    size_t n_active() const noexcept { return active_simplices_.size(); }

    SimplexId add(std::vector<VertexId> vertex_ids, double volume) {
        const auto simplex_id = simplices_.size();
        Simplex simplex;
        simplex.vertex_ids = std::move(vertex_ids);
        simplex.volume = volume;
        simplices_.push_back(std::move(simplex));
        return simplex_id;
    }

    const Simplex &simplex(SimplexId simplex_id) const {
        return simplices_[index(simplex_id)];
    }

    Simplex &mutable_simplex(SimplexId simplex_id) {
        return simplices_[index(simplex_id)];
    }

    std::span<const SimplexId> active_simplices() const noexcept { return active_simplices_; }
    bool is_active(SimplexId simplex_id) const {
        return std::find(active_simplices_.begin(), active_simplices_.end(), simplex_id) !=
               active_simplices_.end();
    }
    bool all_active(std::span<const SimplexId> simplex_ids) const {
        for (const auto simplex_id : simplex_ids) {
            if (!is_active(simplex_id)) {
                return false;
            }
        }
        return true;
    }

    void replace_active_simplices(std::vector<SimplexId> active_simplices) {
        active_simplices_ = std::move(active_simplices);
    }

private:
    size_t index(SimplexId simplex_id) const {
        if (simplex_id >= simplices_.size()) {
            throw std::runtime_error("SimplexTable: simplex_id out of range");
        }
        return simplex_id;
    }

    std::vector<Simplex> simplices_;
    std::vector<SimplexId> active_simplices_;
};

}  // namespace adaptivesimplex::core
