#include "occupation/polynomial.h"
#include "test_helpers.h"

#include <iostream>
#include <numeric>
#include <random>

using namespace fermisimplex::occupation_detail;
using namespace fermisimplex::test;

int main() {
    try {
        std::mt19937 generator(4291);
        std::uniform_real_distribution<double> random(0., 1.);
        double maximum_error = 0;
        for (std::size_t dimension = 1; dimension <= 8; ++dimension)
            for (const std::size_t size : {1, 3, 12}) {
                Polynomial parent(dimension + 1, size);
                for (auto &control : parent.controls) {
                    control.resize(size * size);
                    for (std::size_t j = 0; j < size; ++j)
                        for (std::size_t i = 0; i <= j; ++i) {
                            const auto value = std::complex<double>{2 * random(generator) - 1,
                                i == j ? 0. : 2 * random(generator) - 1};
                            control[i + j * size] = value;
                            control[j + i * size] = std::conj(value);
                        }
                }
                for (std::size_t left = 0; left <= dimension; ++left)
                    for (std::size_t right = left + 1; right <= dimension; ++right)
                        for (const auto replaced : {left, right}) {
                            const auto child = parent.bisected(left, right, replaced);
                            for (int trial = 0; trial < 5; ++trial) {
                                Weights weights(dimension + 1);
                                for (auto &value : weights) value = random(generator);
                                const auto total = std::accumulate(weights.begin(), weights.end(), 0.);
                                for (auto &value : weights) value /= total;
                                auto original = weights;
                                original[replaced] = 0;
                                original[left] += .5 * weights[replaced];
                                original[right] += .5 * weights[replaced];
                                const auto reference = parent.blossom(original, original);
                                const auto value = child.blossom(weights, weights);
                                for (std::size_t k = 0; k < value.size(); ++k)
                                    maximum_error = std::max(maximum_error,
                                        std::abs(reference[k] - value[k]));
                            }
                        }
            }
        expect(maximum_error < 2e-14, "bisection preserves the matrix polynomial in 1D through 8D");
        std::cout << "maximum bisection error=" << maximum_error << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
