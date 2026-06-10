#pragma once

#include "adaptivesimplex/core/dyadic_vertex.h"
#include "adaptivesimplex/core/types.h"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace adaptivesimplex::core {

class VertexTable {
public:
    explicit VertexTable(size_t ndim)
        : ndim_(ndim) {}

    size_t ndim() const noexcept { return ndim_; }
    size_t size() const noexcept { return vertices_.size(); }

    VertexId get_or_add(DyadicVertex vertex) {
        if (vertex.coords().size() != ndim_) {
            throw std::runtime_error("VertexTable: vertex dimension mismatch");
        }

        auto it = lookup_.find(vertex);
        if (it != lookup_.end()) {
            return it->second;
        }

        const auto vertex_id = vertices_.size();
        vertices_.push_back(std::move(vertex));
        lookup_.emplace(vertices_.back(), vertex_id);
        return vertex_id;
    }

    VertexId midpoint(VertexId left, VertexId right) {
        return get_or_add(DyadicVertex::midpoint(dyadic_vertex(left), dyadic_vertex(right)));
    }

    const DyadicVertex &dyadic_vertex(VertexId vertex_id) const {
        if (vertex_id >= vertices_.size()) {
            throw std::runtime_error("VertexTable: vertex_id out of range");
        }
        return vertices_[vertex_id];
    }

private:
    size_t ndim_ = 0;
    std::vector<DyadicVertex> vertices_;
    std::unordered_map<DyadicVertex, VertexId, DyadicVertex::Hash> lookup_;
};

}  // namespace adaptivesimplex::core
