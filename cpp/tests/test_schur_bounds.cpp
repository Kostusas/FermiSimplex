#include "occupation/schur.h"
#include "test_helpers.h"

#include <iostream>
#include <numeric>
#include <random>

using namespace fermisimplex;
using namespace fermisimplex::occupation_detail;
using namespace fermisimplex::test;

namespace {

Matrix hermitian(std::size_t n, std::mt19937 &generator) {
    std::normal_distribution<double> normal;
    Matrix result(n * n);
    for (std::size_t col = 0; col < n; ++col)
        for (std::size_t row = 0; row <= col; ++row) {
            const Complex value{normal(generator), row == col ? 0. : normal(generator)};
            result[row + col * n] = value;
            result[col + row * n] = std::conj(value);
        }
    return result;
}

double minimum_eigenvalue(Matrix matrix, std::size_t n) {
    std::vector<double> values;
    linalg::diagonalize_hermitian_in_place(matrix, values, n, false, "reference bound");
    return values.front();
}

Matrix envelope_value(const ResidualMatrices &bounds, const Weights &point, bool upper) {
    Matrix result(bounds.entries);
    const auto &controls = upper ? bounds.upper : bounds.lower;
    for (std::size_t c = 0; c < bounds.layout->indices.size(); ++c) {
        const auto &index = bounds.layout->indices[c];
        auto weight = multinomial(index);
        for (std::size_t i = 0; i < point.size(); ++i)
            weight *= std::pow(point[i], index[i]);
        for (std::size_t k = 0; k < result.size(); ++k)
            result[k] += weight * controls[c * bounds.entries + k];
    }
    return result;
}

}  // namespace

int main() {
    try {
        std::mt19937 generator(6138);
        std::uniform_real_distribution<double> random(0., 1.);
        const std::vector<std::size_t> safe{0, 1, 4, 5}, active{2, 3};
        const std::size_t n = 6, q = 2, s = 4;
        double maximum_violation = 0, maximum_schur_error = 0;
        std::size_t checks = 0;
        for (std::size_t dimension = 1; dimension <= 3; ++dimension)
            for (std::size_t negative : {0, 2, 4})
                for (double eta : {0., 1e-4, 1e-3}) {
                    const auto v = dimension + 1;
                    std::vector<double> reference{.04, .3, 5., 30.};
                    for (std::size_t i = 0; i < negative; ++i) reference[i] *= -1;
                    Polynomial full(v, n);
                    double gap = std::numeric_limits<double>::infinity();
                    for (auto &control : full.controls) {
                        control = hermitian(n, generator);
                        for (std::size_t col = 0; col < n; ++col)
                            for (std::size_t row = 0; row < n; ++row)
                                control[row + col * n] *= .002;
                        for (std::size_t i = 0; i < s; ++i) {
                            control[safe[i] + safe[i] * n] += reference[i];
                            for (std::size_t j = 0; j < q; ++j) {
                                control[safe[i] + active[j] * n] *= 80 * std::sqrt(std::abs(reference[i]));
                                control[active[j] + safe[i] * n] =
                                    std::conj(control[safe[i] + active[j] * n]);
                            }
                        }
                        // Strong coupling between opposite safe signs is permitted.
                        for (std::size_t i = 0; i < negative; ++i)
                            for (std::size_t j = negative; j < s; ++j) {
                                control[safe[i] + safe[j] * n] *= 200;
                                control[safe[j] + safe[i] * n] =
                                    std::conj(control[safe[i] + safe[j] * n]);
                            }
                        for (const auto sign : {-1, 1}) {
                            const auto begin = sign == -1 ? 0 : negative;
                            const auto end = sign == -1 ? negative : s;
                            if (begin == end) continue;
                            std::vector<std::size_t> sector(safe.begin() + begin, safe.begin() + end);
                            auto matrix = block(control, n, sector, sector);
                            for (auto &value : matrix) value *= sign;
                            gap = std::min(gap, minimum_eigenvalue(matrix, sector.size()) - eta);
                        }
                    }
                    expect(gap > 0, "test safe sectors have certified margins");
                    std::vector<Matrix> solutions;
                    for (std::size_t i = 0; i < v; ++i) {
                        auto d = block(full.at(i, i), n, safe, safe);
                        auto b = block(full.at(i, i), n, safe, active);
                        expect(linalg::solve_linear_system_in_place(d, b, s, q,
                            "test vertex solve"), "safe block invertible");
                        solutions.push_back(std::move(b));
                    }
                    Polynomial fitted(v, q);
                    ResidualMatrices bounds(v, q);
                    ChargeErrorStats stats;
                    const auto epsilon = schur_allowance(full, safe, active, reference,
                        solutions, negative, eta, gap, stats, fitted, bounds);
                    for (std::size_t trial = 0; trial < 100; ++trial) {
                        Weights point(v);
                        for (auto &weight : point) weight = random(generator);
                        const auto total = std::accumulate(point.begin(), point.end(), 0.);
                        for (auto &weight : point) weight /= total;
                        auto actual = full.blossom(point, point);
                        auto perturbation = hermitian(n, generator);
                        const auto scale = eta / norm(perturbation);
                        for (std::size_t i = 0; i < actual.size(); ++i)
                            actual[i] += scale * perturbation[i];
                        auto d = block(actual, n, safe, safe);
                        const auto b = block(actual, n, safe, active);
                        auto solution = b;
                        expect(linalg::solve_linear_system_in_place(d, solution, s, q,
                            "exact Schur solve"), "perturbed safe block invertible");
                        auto exact = block(actual, n, active, active);
                        linalg::matrix_multiply('C', 'N', q, q, s, -1., b.data(), s,
                            solution.data(), s, 1., exact.data(), q);
                        const auto approximate = fitted.blossom(point, point);
                        auto low = envelope_value(bounds, point, false);
                        auto high = envelope_value(bounds, point, true);
                        auto error = exact;
                        for (std::size_t k = 0; k < exact.size(); ++k) {
                            low[k] = exact[k] - low[k];
                            high[k] -= exact[k];
                            error[k] -= approximate[k];
                        }
                        for (std::size_t band = 0; band < q; ++band) {
                            low[band + band * q] += bounds.remainder;
                            high[band + band * q] += bounds.remainder;
                        }
                        maximum_violation = std::max({maximum_violation,
                            -minimum_eigenvalue(low, q), -minimum_eigenvalue(high, q)});
                        const auto error_norm = norm(error);
                        // Frobenius <= sqrt(q) times the operator-norm allowance.
                        expect(error_norm <= std::sqrt(double(q)) * epsilon + 1e-12,
                            "scalar Schur bound contains direct solve");
                        maximum_schur_error = std::max(maximum_schur_error, error_norm);
                        ++checks;
                    }
                }
        expect(maximum_violation < 1e-12, "matrix envelopes contain perturbed exact Schur matrices");
        std::cout << "Schur envelope checks=" << checks
                  << " maximum violation=" << maximum_violation
                  << " maximum Schur Frobenius error=" << maximum_schur_error << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
