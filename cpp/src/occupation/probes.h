#pragma once

#include "occupation/polynomial.h"

#include <map>

namespace fermisimplex::occupation_detail {

// Complete simplex lattice, excluding the vertices and edge midpoints already
// used by the quadratic interpolant. The rule depends only on vertex count.
inline const std::vector<Weights> &probe_weights(std::size_t vertices) {
    static thread_local std::map<std::size_t, std::vector<Weights>> rules;
    if (const auto found = rules.find(vertices); found != rules.end())
        return found->second;
    constexpr unsigned degree = 4;
    std::vector<Weights> probes;
    Weights weights(vertices);
    const auto visit = [&](auto &&self, std::size_t axis, unsigned remaining,
                           unsigned support, bool half_only) -> void {
        const auto append = [&](unsigned count) {
            weights[axis] = static_cast<double>(count) / degree;
            const auto next_support = support + (count != 0);
            const auto next_half_only = half_only && (count == 0 || 2 * count == degree);
            if (axis + 1 == vertices) {
                if (next_support != 1 && !next_half_only) probes.push_back(weights);
            } else {
                self(self, axis + 1, remaining - count, next_support, next_half_only);
            }
        };
        if (axis + 1 == vertices) append(remaining);
        else for (unsigned count = 0; count <= remaining; ++count) append(count);
    };
    visit(visit, 0, degree, 0, true);
    return rules.emplace(vertices, std::move(probes)).first->second;
}

inline double probe_remainder_factor(std::size_t dimension) {
    // Rational subdivision proves 2^d through d=4. In any dimension the
    // degree-four residual's Bernstein coefficient norm is at most 32:
    // every coefficient has support on at most four vertices. See the proof
    // in benchmarks/verify_remainder_factor.py and docs/occupation-enclosure.md.
    return std::ldexp(1., static_cast<int>(std::min<std::size_t>(dimension, 5)));
}

}  // namespace fermisimplex::occupation_detail
