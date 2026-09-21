#pragma once

#include <algorithm>
#include <cmath>

namespace fermisimplex::integration_detail {

struct DensityGlobalError {
    template <class Value>
    using state_type = double;

    bool has_preview = true;

    template <class Value>
    state_type<Value> zero() const {
        return 0.0;
    }

    template <class Value>
    void add_local_estimate(
        state_type<Value> &state,
        const Value &local_estimate
    ) const {
        if (has_preview) {
            const auto error = local_estimate.max_abs();
            state += error * error;
        }
    }

    template <class Value>
    void remove_local_estimate(
        state_type<Value> &state,
        const Value &local_estimate
    ) const {
        if (has_preview) {
            const auto error = local_estimate.max_abs();
            state -= error * error;
        }
    }

    template <class Value>
    double error(
        const state_type<Value> &state,
        const Value &global_correction
    ) const {
        return std::max(
            std::sqrt(std::max(0.0, state)),
            global_correction.max_abs()
        );
    }
};

}  // namespace fermisimplex::integration_detail
