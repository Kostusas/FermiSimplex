#include "certification/linalg/cholesky.h"
#include "certification/bounds/occupation_bounds.h"
#include "test_helpers.h"

#include <complex>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace fermisimplex::test;
namespace detail = fermisimplex::certification::detail;

std::pair<size_t, size_t> winding_bounds(int winding, std::initializer_list<double> points) {
    std::vector<std::vector<Complex>> plus_blocks;
    std::vector<std::vector<Complex>> minus_blocks;
    for (const auto point : points) {
        auto h = winding_hamiltonian(winding, point);
        plus_blocks.push_back(diagonal_matrix({
            std::real(h[detail::column_major_index(0, 0, 2)]),
        }));
        minus_blocks.push_back(diagonal_matrix({
            -std::real(h[detail::column_major_index(1, 1, 2)]),
        }));
    }
    const auto r_minus =
        detail::estimate_ordered_subset_rank(minus_blocks, 1, 0.0, kTol);
    const auto r_plus =
        detail::estimate_ordered_subset_rank(plus_blocks, 1, 0.0, kTol);
    return {r_minus, 2 - r_plus};
}

void test_explicit_occupation_bound_cases() {
    expect_eq(
        detail::estimate_ordered_subset_rank(
            {
                diagonal_matrix({2.0, 3.0}),
                diagonal_matrix({1.5, 2.5}),
                diagonal_matrix({4.0, 1.2}),
            },
            2,
            0.0,
            kTol
        ),
        2,
        "all-positive blocks should certify full rank"
    );

    expect_eq(
        detail::estimate_ordered_subset_rank(
            {
                diagonal_matrix({2.0, -1.0}),
                diagonal_matrix({1.5, -0.5}),
            },
            2,
            0.0,
            kTol
        ),
        1,
        "single common positive direction should certify rank one"
    );

    expect_eq(
        detail::estimate_ordered_subset_rank(
            {
                diagonal_matrix({1.0, -1.0}),
                diagonal_matrix({-1.0, 1.0}),
            },
            2,
            0.0,
            kTol
        ),
        0,
        "antipodal positive cones should certify no common direction"
    );

    expect_eq(
        detail::estimate_ordered_subset_rank(
            {
                diagonal_matrix({3.0, 2.0, -1.0}),
                diagonal_matrix({2.0, 3.0, -0.5}),
                diagonal_matrix({4.0, 1.5, -2.0}),
            },
            3,
            0.0,
            kTol
        ),
        2,
        "ordered-subset test should return the largest passing rank"
    );

    expect_eq(
        detail::estimate_ordered_subset_rank(
            {
                diagonal_matrix({1.0, 3.0, 2.0}),
                diagonal_matrix({1.0, -1.0, 2.0}),
            },
            3,
            0.0,
            kTol
        ),
        2,
        "ordered-subset test should sort directions by worst vertex margin"
    );

    const auto full_rank_estimate = detail::estimate_ordered_subset_rank_with_mu_radius(
        {
            diagonal_matrix({2.0, 3.0}),
            diagonal_matrix({1.5, 2.5}),
        },
        2,
        0.0,
        kTol
    );
    expect_eq(full_rank_estimate.rank, 2, "rank estimate should report full rank");
    expect(
        full_rank_estimate.mu_radius > 1.0 && full_rank_estimate.mu_radius < 1.6,
        "rank estimate should report a finite one-sided mu radius"
    );

    const auto zero_rank_estimate = detail::estimate_ordered_subset_rank_with_mu_radius(
        {
            diagonal_matrix({1.0, -1.0}),
            diagonal_matrix({-1.0, 1.0}),
        },
        2,
        0.0,
        kTol
    );
    expect_eq(zero_rank_estimate.rank, 0, "zero-rank estimate should report zero rank");
    expect(
        std::isinf(zero_rank_estimate.mu_radius),
        "zero-rank estimate should impose no one-sided mu constraint"
    );
}

void test_positive_definite_reports_accepted_rank() {
    const auto zero_size = detail::positive_definite({}, 0);
    expect(zero_size.passed, "zero-size Cholesky test should pass");
    expect_eq(zero_size.accepted_rank, 0, "zero-size accepted rank");

    const auto full = detail::positive_definite(diagonal_matrix({2.0, 3.0}), 2);
    expect(full.passed, "positive diagonal matrix should pass Cholesky");
    expect_eq(full.accepted_rank, 2, "full Cholesky accepted rank");

    const auto partial = detail::positive_definite(diagonal_matrix({2.0, -1.0, 3.0}), 3);
    expect(!partial.passed, "matrix with negative second pivot should fail Cholesky");
    expect_eq(partial.accepted_rank, 1, "failed Cholesky accepted rank");
}

void test_winding_model_bounds() {
    const auto short_bounds = winding_bounds(2, {0.0, 0.0625});
    expect_eq(short_bounds.first, 1, "short winding interval lower bound");
    expect_eq(short_bounds.second, 1, "short winding interval upper bound");

    const auto antipodal_bounds = winding_bounds(2, {0.0, 0.25});
    expect_eq(antipodal_bounds.first, 0, "antipodal winding interval lower bound");
    expect_eq(antipodal_bounds.second, 2, "antipodal winding interval upper bound");
}

}  // namespace

int main() {
    try {
        test_explicit_occupation_bound_cases();
        test_positive_definite_reports_accepted_rank();
        test_winding_model_bounds();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
