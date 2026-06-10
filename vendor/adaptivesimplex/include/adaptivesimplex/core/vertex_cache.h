#pragma once

#include "adaptivesimplex/core/types.h"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adaptivesimplex::core {

template <class Value>
class VertexCache {
public:
    using value_type = Value;

    explicit VertexCache(size_t initial_reserve = 1'000'000) {
        entries_.reserve(initial_reserve);
    }

    VertexCache(const VertexCache &) = delete;
    VertexCache &operator=(const VertexCache &) = delete;
    VertexCache(VertexCache &&) noexcept = default;
    VertexCache &operator=(VertexCache &&) noexcept = default;

    void clear() {
        entries_.clear();
        cached_count_ = 0;
    }

    size_t size() const noexcept { return cached_count_; }

    bool contains(VertexId vertex_id) const noexcept {
        return vertex_id < entries_.size() && entries_[vertex_id] != nullptr;
    }

    const value_type &get(VertexId vertex_id) const {
        if (!contains(vertex_id)) {
            throw std::runtime_error("VertexCache: vertex is not cached");
        }
        return *entries_[vertex_id];
    }

    void insert(VertexId vertex_id, value_type value) {
        if (vertex_id >= entries_.size()) {
            entries_.resize(vertex_id + 1);
        }

        auto &entry = entries_[vertex_id];
        if (entry == nullptr) {
            ++cached_count_;
        }
        entry = std::make_unique<value_type>(std::move(value));
    }

    void insert_many(std::span<const VertexId> vertex_ids, std::vector<value_type> values) {
        if (vertex_ids.size() != values.size()) {
            throw std::runtime_error("VertexCache: insert_many size mismatch");
        }
        for (size_t index = 0; index < vertex_ids.size(); ++index) {
            insert(vertex_ids[index], std::move(values[index]));
        }
    }

private:
    size_t cached_count_ = 0;
    std::vector<std::unique_ptr<value_type>> entries_;
};

}  // namespace adaptivesimplex::core
