#include "certificate/internal.h"
#include "certificate/simplex_certificate.h"
#include "core/tight_binding.h"
#include "integration/runtime.h"

#include <adaptivesimplex/adaptive/types.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;
namespace detail = lineartetrahedron::simplex_certificate::detail;

constexpr double kTol = 1e-12;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_eq(size_t actual, size_t expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void expect_near(double actual, double expected, double tol, const std::string &message) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

std::vector<Complex> diagonal_matrix(std::initializer_list<double> diagonal) {
    const auto size = diagonal.size();
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    auto index = size_t{0};
    for (const auto value : diagonal) {
        result[detail::column_major_index(index, index, size)] = Complex{value, 0.0};
        ++index;
    }
    return result;
}

std::vector<Complex> hermitian_2x2(double a, double b, double c) {
    return {
        Complex{a, 0.0},
        Complex{b, 0.0},
        Complex{b, 0.0},
        Complex{c, 0.0},
    };
}

std::vector<Complex> winding_hamiltonian(int winding, double reduced_k) {
    const auto phase = 2.0 * lineartetrahedron::kPi * static_cast<double>(winding) * reduced_k;
    const auto c = std::cos(phase);
    const auto s = std::sin(phase);
    return hermitian_2x2(c, s, -c);
}

std::vector<Complex> negated(std::vector<Complex> matrix) {
    for (auto &value : matrix) {
        value = -value;
    }
    return matrix;
}

std::pair<size_t, size_t> winding_bounds(int winding, std::initializer_list<double> points) {
    std::vector<std::vector<Complex>> plus_blocks;
    std::vector<std::vector<Complex>> minus_blocks;
    for (const auto point : points) {
        auto h = winding_hamiltonian(winding, point);
        plus_blocks.push_back(h);
        minus_blocks.push_back(negated(std::move(h)));
    }
    const auto r_minus = detail::estimate_common_rank(minus_blocks, 2, kTol);
    const auto r_plus = detail::estimate_common_rank(plus_blocks, 2, kTol);
    return {r_minus, 2 - r_plus};
}

std::shared_ptr<lineartetrahedron::TightBindingModel> winding_model(int winding) {
    std::vector<std::int64_t> keys{
        static_cast<std::int64_t>(winding),
        static_cast<std::int64_t>(-winding),
    };
    const auto sigma_z = diagonal_matrix({1.0, -1.0});
    const auto sigma_x = hermitian_2x2(0.0, 1.0, 0.0);

    std::vector<Complex> matrices;
    matrices.reserve(2 * 4);
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        matrices.push_back(0.5 * sigma_z[index] + Complex{0.0, 0.5} * sigma_x[index]);
    }
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        matrices.push_back(0.5 * sigma_z[index] - Complex{0.0, 0.5} * sigma_x[index]);
    }
    return std::make_shared<lineartetrahedron::TightBindingModel>(
        1,
        2,
        std::move(keys),
        std::move(matrices)
    );
}

void test_explicit_common_rank_cases() {
    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({2.0, 3.0}),
                diagonal_matrix({1.5, 2.5}),
                diagonal_matrix({4.0, 1.2}),
            },
            2,
            kTol
        ),
        2,
        "all-positive blocks should certify full rank"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({2.0, -1.0}),
                diagonal_matrix({1.5, -0.5}),
            },
            2,
            kTol
        ),
        1,
        "single common positive direction should certify rank one"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({1.0, -1.0}),
                diagonal_matrix({-1.0, 1.0}),
            },
            2,
            kTol
        ),
        0,
        "antipodal positive cones should certify no common direction"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({3.0, 2.0, -1.0}),
                diagonal_matrix({2.0, 3.0, -0.5}),
                diagonal_matrix({4.0, 1.5, -2.0}),
            },
            3,
            kTol
        ),
        2,
        "binary search should return the largest passing nested rank"
    );
}

void test_winding_model_bounds() {
    const auto short_bounds = winding_bounds(2, {0.0, 0.125});
    expect_eq(short_bounds.first, 1, "short winding interval lower bound");
    expect_eq(short_bounds.second, 1, "short winding interval upper bound");

    const auto antipodal_bounds = winding_bounds(2, {0.0, 0.25});
    expect_eq(antipodal_bounds.first, 0, "antipodal winding interval lower bound");
    expect_eq(antipodal_bounds.second, 2, "antipodal winding interval upper bound");
}

void test_charge_path_uses_occupation_bounds() {
    const auto options = adaptivesimplex::adaptive::Options{
        .target_error = 0.75,
        .max_refinements = 0,
        .preview_depth = 1,
        .min_refinement_batch_size = 1,
        .max_refinement_batch_size = 100,
    };

    auto bounded_runtime = lineartetrahedron::IntegrationRuntime(
        winding_model(3),
        {0},
        {},
        {},
        {},
        kTol
    );
    const auto bounded_result = bounded_runtime.evaluate_charge(0.0, options);
    expect_near(
        bounded_result.charge_error,
        2.0,
        1e-12,
        "charge path should use the certified [0,2] occupation-width error"
    );

    auto visible_runtime = lineartetrahedron::IntegrationRuntime(
        winding_model(1),
        {0},
        {},
        {},
        {},
        kTol
    );
    const auto visible_default = visible_runtime.evaluate_charge(0.0, options);
    expect_near(
        visible_default.charge_error,
        0.0,
        1e-12,
        "visible/certified charge cases should not gain bound-based error"
    );
}

}  // namespace

int main() {
    try {
        test_explicit_common_rank_cases();
        test_winding_model_bounds();
        test_charge_path_uses_occupation_bounds();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
