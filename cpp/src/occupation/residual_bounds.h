#pragma once

#include "occupation/polynomial.h"

namespace fermisimplex::occupation_detail {

// P(k)-lower-remainder*I <= S(k) <= P(k)+upper+remainder*I.
// These two constant matrices are shared by all temporary subdivisions.
struct ResidualBounds {
    Matrix lower, upper;
    double remainder = 0;

    explicit ResidualBounds(std::size_t size)
        : lower(size * size), upper(size * size) {}
};

}  // namespace fermisimplex::occupation_detail
