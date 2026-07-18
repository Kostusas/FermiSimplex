#include "certification/bounds/occupation_bounds.h"
#include "test_helpers.h"

#include <fermisimplex/certification.h>

#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace fermisimplex::test;
namespace certificate = fermisimplex::certification;
namespace detail = fermisimplex::certification::detail;

void test_zero_width_mu_bounds_are_valid_only_at_the_endpoint() {
    const auto estimate = detail::estimate_ordered_subset_rank_with_mu_radius(
        {
            {
                Complex{1.0, 0.0},
                Complex{1.0, 0.0},
                Complex{1.0, 0.0},
                Complex{2.0, 0.0},
            },
        },
        2,
        0.0,
        kTol
    );
    expect_eq(estimate.rank, 2, "positive block should certify rank two");
    expect_near(
        estimate.mu_radius,
        0.0,
        1e-14,
        "non-diagonally-dominant positive block should have zero Gershgorin mu radius"
    );

    auto result = certificate::SimplexCertificate{};
    result.status = certificate::SimplexCertificateStatus::Inconclusive;
    result.occupation_bounds = certificate::OccupationBounds{.lower = 2, .upper = 2};
    result.mu_interval = certificate::MuInterval{.lower = 0.0, .upper = 0.0};
    expect(result.mu_interval.has_value(), "zero-width mu interval is still an interval");
    expect(
        certificate::occupation_bounds_valid_at(result, 0.0),
        "zero-width interval should include its endpoint"
    );
    expect(
        !certificate::occupation_bounds_valid_at(result, 1e-8),
        "zero-width interval should exclude nearby mu"
    );
}

void test_empty_simplex_spectra_throws() {
    auto threw = false;
    try {
        (void)certificate::certify_simplex(
            std::span<const std::span<const double>>{},
            std::span<const std::span<const Complex>>{},
            0.0,
            0.0
        );
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect(threw, "empty simplex spectra should throw");
}

void test_invalid_linearization_error_bound_throws() {
    auto rejected_error = false;
    try {
        const auto spectra = std::vector<fermisimplex::Eigensystem>{
            diagonal_spectra({-1.0, 1.0}),
        };
        (void)certify_direct(spectra, 0.0, -1.0);
    } catch (const std::runtime_error &) {
        rejected_error = true;
    }
    expect(rejected_error, "negative linearization error should throw");
}

void test_nonfinite_parameters_and_invalid_eigenvalues_throw() {
    const auto valid = std::vector<fermisimplex::Eigensystem>{
        diagonal_spectra({-1.0, 1.0}),
    };

    auto rejected_mu = false;
    try {
        (void)certify_direct(valid, std::numeric_limits<double>::infinity());
    } catch (const std::runtime_error &) {
        rejected_mu = true;
    }
    expect(rejected_mu, "non-finite mu should throw");

    auto rejected_tolerance = false;
    try {
        (void)certify_direct(valid, 0.0, 0.0, -1.0);
    } catch (const std::runtime_error &) {
        rejected_tolerance = true;
    }
    expect(rejected_tolerance, "negative tolerance should throw");

    auto rejected_nonfinite_tolerance = false;
    try {
        (void)certify_direct(
            valid,
            0.0,
            0.0,
            std::numeric_limits<double>::quiet_NaN()
        );
    } catch (const std::runtime_error &) {
        rejected_nonfinite_tolerance = true;
    }
    expect(rejected_nonfinite_tolerance, "non-finite tolerance should throw");

    auto rejected_nonfinite_eigenvalue = false;
    try {
        (void)certify_direct({diagonal_spectra({
            -1.0,
            std::numeric_limits<double>::infinity(),
        })});
    } catch (const std::runtime_error &) {
        rejected_nonfinite_eigenvalue = true;
    }
    expect(rejected_nonfinite_eigenvalue, "non-finite eigenvalues should throw");

    auto rejected_unsorted_eigenvalues = false;
    try {
        (void)certify_direct({diagonal_spectra({1.0, -1.0})});
    } catch (const std::runtime_error &) {
        rejected_unsorted_eigenvalues = true;
    }
    expect(rejected_unsorted_eigenvalues, "decreasing eigenvalues should throw");
}

void test_invalid_eigenvectors_throw() {
    auto nonfinite = diagonal_spectra({-1.0, 1.0});
    nonfinite.eigenvectors[0] = Complex{
        1.0,
        std::numeric_limits<double>::quiet_NaN(),
    };
    auto rejected_nonfinite = false;
    try {
        (void)certify_direct({nonfinite});
    } catch (const std::runtime_error &) {
        rejected_nonfinite = true;
    }
    expect(rejected_nonfinite, "non-finite eigenvector entries should throw");

    auto nonorthonormal = diagonal_spectra({-1.0, 1.0});
    nonorthonormal.eigenvectors = {
        Complex{1.0, 0.0},
        Complex{0.0, 0.0},
        Complex{1.0, 0.0},
        Complex{0.0, 0.0},
    };
    auto rejected_nonorthonormal = false;
    try {
        (void)certify_direct({nonorthonormal});
    } catch (const std::runtime_error &) {
        rejected_nonorthonormal = true;
    }
    expect(rejected_nonorthonormal, "non-orthonormal eigenvector columns should throw");
}

}  // namespace

int main() {
    try {
        test_zero_width_mu_bounds_are_valid_only_at_the_endpoint();
        test_empty_simplex_spectra_throws();
        test_invalid_linearization_error_bound_throws();
        test_nonfinite_parameters_and_invalid_eigenvalues_throw();
        test_invalid_eigenvectors_throw();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
