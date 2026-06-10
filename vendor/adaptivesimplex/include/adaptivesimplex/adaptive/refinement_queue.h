#pragma once

#include "adaptivesimplex/core/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace adaptivesimplex::adaptive {

struct RefinementCandidate {
    double error = 0.0;
    core::SimplexId simplex_id = 0;

    bool operator<(const RefinementCandidate &other) const noexcept {
        return error < other.error;
    }
};

class RefinementQueue {
public:
    void push(core::SimplexId simplex_id, double error) {
        if (error > 0.0) {
            heap_.push(RefinementCandidate{error, simplex_id});
        }
    }

    std::vector<core::SimplexId> select_for_reduction(
        double target_reduction,
        std::int64_t remaining_refinements,
        size_t min_batch_size,
        size_t max_batch_size
    ) {
        min_batch_size = std::max<size_t>(min_batch_size, 1);
        max_batch_size = std::max(max_batch_size, min_batch_size);
        if (remaining_refinements > 0) {
            max_batch_size = std::min(max_batch_size, static_cast<size_t>(remaining_refinements));
        }

        std::vector<core::SimplexId> selected;
        double accumulated = 0.0;
        while (!heap_.empty() && selected.size() < max_batch_size) {
            if (selected.size() >= min_batch_size && accumulated >= target_reduction) {
                break;
            }
            const auto top = heap_.top();
            heap_.pop();
            if (top.error <= 0.0) {
                break;
            }
            selected.push_back(top.simplex_id);
            accumulated += top.error;
        }
        return selected;
    }

private:
    std::priority_queue<RefinementCandidate> heap_;
};

}  // namespace adaptivesimplex::adaptive
