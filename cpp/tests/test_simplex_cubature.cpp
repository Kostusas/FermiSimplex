#include <complex>
#include "integration/simplex_cubature.h"
#include "integration/density_error.h"
#include <adaptivesimplex/adaptive/dense_value.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fermisimplex::integration_detail;

// Check every barycentric monomial through the rule's advertised degree.
// Exact normalized simplex moment: d! prod(alpha_i!) / (d+sum(alpha_i))!.
void moments(const Cubature &rule, unsigned d, unsigned degree) {
    std::vector<unsigned> alpha(d + 1);
    const auto check = [&]() {
        unsigned total = 0;
        long double log_exact = std::lgamma(static_cast<long double>(d + 1));
        for (auto a : alpha) {
            total += a;
            log_exact += std::lgamma(static_cast<long double>(a + 1));
        }
        const auto exact = std::exp(log_exact - std::lgamma(static_cast<long double>(d + total + 1)));
        long double actual = 0;
        for (const auto &[node, weight] : rule) {
            const auto denominator = std::accumulate(node.begin(), node.end(), 0U);
            auto term = weight;
            for (unsigned i = 0; i <= d; ++i) {
                term *= std::pow(static_cast<long double>(node[i]) / denominator, alpha[i]);
            }
            actual += term;
        }
        if (std::abs(actual - exact) > 2e-11L * exact + 1e-15L) {
            throw std::runtime_error("incorrect simplex monomial moment");
        }
    };
    const auto enumerate = [&](auto &&self, unsigned axis, unsigned left) -> void {
        if (axis == d) { alpha[axis] = left; check(); return; }
        for (unsigned i = 0; i <= left; ++i) {
            alpha[axis] = i;
            self(self, axis + 1, left - i);
        }
    };
    for (unsigned total = 0; total <= degree; ++total) enumerate(enumerate, 0, total);
}

int main() {
    using Value = adaptivesimplex::adaptive::DenseValue<std::complex<double>>;
    DensityGlobalError policy;
    double state = 0;
    Value positive(2), negative(2), coherent(2), cancelled(2);
    positive[0] = 3.; negative[0] = -3.; coherent[0] = 6.;
    policy.add_local_estimate(state, positive);
    policy.add_local_estimate(state, negative);
    if (std::abs(policy.error(state, cancelled)-std::sqrt(18.)) > 1e-14 ||
        policy.error(state, coherent) != 6.) {
        throw std::runtime_error("statistical/coherent density error mismatch");
    }
    policy.remove_local_estimate(state, negative);
    if (policy.error(state, positive) != 3.) {
        throw std::runtime_error("density error removal mismatch");
    }
    for (unsigned d = 1; d <= 3; ++d) {
        moments(vertices_centroid(d), d, 2);
        Cubature previous;
        for (unsigned s = 0; s <= 10; ++s) {
            const auto rule = grundmann_moeller(d, s);
            for (const auto &[node, weight] : previous) {
                if (!rule.contains(node)) throw std::runtime_error("rules are not nested");
            }
            moments(rule, d, 2 * s + 1);
            previous = rule;
        }
    }
    std::cout << "All simplex cubature moments and nesting passed\n";
}
