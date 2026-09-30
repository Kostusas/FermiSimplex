#include "occupation/probes.h"
#include "test_helpers.h"

#include <iostream>
#include <numeric>
#include <set>

using namespace fermisimplex::occupation_detail;
using namespace fermisimplex::test;

int main() {
    try {
        for (std::size_t d = 1; d <= 8; ++d) {
            const auto &probes = probe_weights(d + 1);
            const auto count = (d+4)*(d+3)*(d+2)*(d+1)/24 - (d+2)*(d+1)/2;
            expect_eq(probes.size(), count, "complete quartic lattice without quadratic nodes");
            const std::set<Weights> unique(probes.begin(), probes.end());
            expect_eq(unique.size(), count, "no repeated probes");
            for (auto weights : probes) {
                expect(std::accumulate(weights.begin(), weights.end(), 0.) == 1.,
                       "barycentric weights sum to one");
                // Transpositions generate every vertex permutation.
                for (std::size_t i = 1; i <= d; ++i) {
                    std::swap(weights[0], weights[i]);
                    expect(unique.contains(weights), "rule is invariant under vertex relabeling");
                    std::swap(weights[0], weights[i]);
                }
            }
            if (d >= 3) {
                Weights face_center(d + 1);
                std::fill_n(face_center.begin(), 4, .25);
                expect(unique.contains(face_center), "every four-vertex face is probed");
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
