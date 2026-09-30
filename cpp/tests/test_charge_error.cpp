#include "test_helpers.h"
#include "integration/charge.h"
#include "occupation/cut_disagreement.h"
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace fermisimplex;
using namespace fermisimplex::test;

void test_density_cut_error_result_boundary() {
    using fermisimplex::integration_detail::validated_density_cut_error;
    expect_eq(validated_density_cut_error(-1e-17, 1.0), 0.0,
              "roundoff-negative cut error must normalize to zero");
    expect_eq(validated_density_cut_error(0.25, 1.0), 0.25,
              "positive cut error must be preserved");
    for (const auto invalid : {
             -1e-6,
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()
         }) {
        auto rejected = false;
        try {
            (void)validated_density_cut_error(invalid, 1.0);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        expect(rejected, "invalid cut error must be rejected");
    }
}

void test_two_affine_cut_disagreement() {
    using occupation_detail::cut_disagreement;
    constexpr auto tolerance = 1e-14;
    const auto first_1d = std::array{-0.25, 0.75};
    const auto second_1d = std::array{0.75, -0.25};
    expect_near(
        cut_disagreement(1.0, first_1d, second_1d, tolerance),
        0.5, 1e-13, "opposed interval cuts"
    );
    expect_near(
        cut_disagreement(1.0, first_1d, first_1d, tolerance),
        0.0, 1e-13, "identical interval cuts"
    );
    const auto nested_1d = std::array{-0.5, 0.5};
    expect_near(
        cut_disagreement(1.0, first_1d, nested_1d, tolerance),
        0.25, 1e-13, "nested interval cuts"
    );

    const auto x_2d = std::array{-0.5, 0.5, -0.5};
    const auto y_2d = std::array{-0.5, -0.5, 0.5};
    expect_near(
        cut_disagreement(0.5, x_2d, y_2d, tolerance),
        0.25, 1e-13, "crossing triangular cuts"
    );

    const auto x_3d = std::array{-0.5, 0.5, -0.5, -0.5};
    const auto y_3d = std::array{-0.5, -0.5, 0.5, -0.5};
    expect_near(
        cut_disagreement(1.0 / 6.0, x_3d, y_3d, tolerance),
        1.0 / 24.0, 1e-13, "crossing tetrahedral cuts"
    );
    const auto flat = std::array{0.0, 0.0};
    expect_near(
        cut_disagreement(1.0, flat, first_1d, tolerance),
        0.5, 1e-13, "half-occupied flat cut"
    );
}

int main() {
    try {
        test_density_cut_error_result_boundary();
        test_two_affine_cut_disagreement();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
