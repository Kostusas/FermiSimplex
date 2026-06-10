#pragma once

#include <cstddef>
#include <cstdint>

namespace adaptivesimplex::adaptive {

struct Options {
    double target_error = 0.0;
    std::int64_t max_refinements = 0;
    std::uint32_t preview_depth = 1;
    size_t min_refinement_batch_size = 1;
    size_t max_refinement_batch_size = 100;
};

template <class Value>
struct Estimate {
    Value value;
    Value correction;
    double indicator = 0.0;
};

template <class Value>
struct Result {
    Value value;
    Value correction;
    double error_scalar = 0.0;
    std::int64_t work = 0;
    std::int64_t refinements = 0;
    bool converged = false;
};

}  // namespace adaptivesimplex::adaptive
