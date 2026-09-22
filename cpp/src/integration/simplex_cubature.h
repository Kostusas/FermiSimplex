#pragma once

#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <vector>

namespace fermisimplex::integration_detail {

// Integer homogeneous barycentric coordinates, reduced by their common gcd.
// Exact keys reuse coincident nodes across orders, including the centroid.
using BarycentricNode = std::vector<unsigned>;
using Cubature = std::map<BarycentricNode, long double>;

inline BarycentricNode canonical_node(BarycentricNode node) {
    unsigned divisor = 0;
    for (auto entry : node) divisor = std::gcd(divisor, entry);
    for (auto &entry : node) entry /= divisor;
    return node;
}

// Normalized weights (sum=1). Grundmann & Moeller (1978),
// doi:10.1137/0715019. Index s is exact through degree 2*s+1.
// The positive odd barycentric lattices at levels 0..s are nested.
inline Cubature grundmann_moeller(unsigned dimension, unsigned s) {
    Cubature result;
    BarycentricNode node(dimension + 1);
    for (unsigned t = 0; t <= s; ++t) {
        const auto denominator = dimension + 1 + 2 * t;
        auto weight = std::exp(
            std::lgamma(static_cast<long double>(dimension + 1)) +
            (2 * s + 1) * std::log(static_cast<long double>(denominator)) -
            2 * s * std::log(2.L) -
            std::lgamma(static_cast<long double>(s - t + 1)) -
            std::lgamma(static_cast<long double>(dimension + 2 + s + t))
        );
        if ((s - t) % 2) weight = -weight;
        const auto enumerate = [&](auto &&self, unsigned axis, unsigned left) -> void {
            if (axis == dimension) {
                node[axis] = 2 * left + 1;
                result[canonical_node(node)] += weight;
                return;
            }
            for (unsigned beta = 0; beta <= left; ++beta) {
                node[axis] = 2 * beta + 1;
                self(self, axis + 1, left - beta);
            }
        };
        enumerate(enumerate, 0, t);
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
