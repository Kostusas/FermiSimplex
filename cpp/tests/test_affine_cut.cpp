#include "occupation/affine_cut.h"
#include "test_helpers.h"

#include <adaptivesimplex/cut/simplex_moments.h>
#include <iostream>
#include <random>

using namespace fermisimplex::occupation_detail;
using namespace fermisimplex::test;

int main() {
    try {
        double worst_volume_error = 0, worst_derivative_error = 0;
        std::mt19937 generator(718);
        std::uniform_real_distribution<double> random(-1, 1);
        for (std::size_t dimension = 1; dimension <= 7; ++dimension)
            for (int trial = 0; trial < 50; ++trial) {
                std::vector<double> values(dimension + 1);
                for (auto &value : values)
                    value = random(generator) * (trial % 2 ? 1. : 4e-14);
                const auto reference = adaptivesimplex::cut::simplex_moments(1., values);
                const AffineCut cut(values, 1e-14);
                const auto fraction = reference.kind == adaptivesimplex::cut::SimplexCutKind::on_level
                    ? .5 : reference.volume;
                const auto error = std::abs(cut.fraction() - fraction);
                worst_volume_error = std::max(worst_volume_error, error);
                expect(error < 2e-12, "scalar fraction agrees with independent geometric clipping");
            }

        // One nonzero barycentric coordinate has a Beta(1,d) distribution.
        // This checks repeated knots, bandwidth scaling and endpoint limits.
        for (std::size_t dimension = 1; dimension <= 12; ++dimension)
            for (const auto width : {1e-10, 1., 1e10})
                for (const auto level : {0., .125, .5, .875, 1.}) {
                    std::vector<double> values(dimension + 1, -width * level);
                    values.back() += width;
                    const AffineCut cut(values, 1e-14);
                    const auto fraction = 1 - std::pow(1 - level, dimension);
                    const auto derivative = level == 0 ? 0. :
                        dimension * std::pow(1 - level, dimension - 1);
                    const auto error = std::abs(width * cut.derivative() - derivative);
                    worst_derivative_error = std::max(worst_derivative_error, error);
                    expect(std::abs(cut.fraction() - fraction) < 2e-12,
                           "repeated-knot fraction agrees with exact Beta CDF");
                    expect(error < 2e-12,
                           "scaled derivative agrees with exact Beta density");
                }
        std::cout << "maximum volume error=" << worst_volume_error
                  << " scaled derivative error=" << worst_derivative_error << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
