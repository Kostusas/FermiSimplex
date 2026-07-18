#pragma once

#include "certification/linalg/matrix.h"

#include <cstddef>
#include <vector>

namespace fermisimplex::certification::detail {

struct PositiveDefiniteResult {
    bool passed = false;
    size_t accepted_rank = 0;
};

double certificate_margin(double tol);

PositiveDefiniteResult positive_definite(
    std::vector<Complex> block,
    size_t size,
    double margin = 0.0
);

}  // namespace fermisimplex::certification::detail
