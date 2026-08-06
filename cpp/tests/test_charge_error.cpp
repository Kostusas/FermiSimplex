#include "test_helpers.h"

#include <fermisimplex/hamiltonian.h>
#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fermisimplex;
using namespace fermisimplex::test;

using Matrix = std::vector<Complex>;
using Evaluator = std::function<Matrix(std::span<const double>)>;

std::size_t cm(std::size_t row, std::size_t column, std::size_t size) {
    return row + column * size;
}

class TestModel final : public HamiltonianModel {
public:
    TestModel(
        std::size_t ndim,
        std::size_t ndof,
        Evaluator evaluator
    ) : ndim_(ndim), ndof_(ndof), evaluator_(std::move(evaluator)) {}

    std::size_t ndim() const noexcept override { return ndim_; }
    std::size_t ndof() const noexcept override { return ndof_; }

    Matrix evaluate(std::span<const double> point) const override {
        return evaluator_(point);
    }

private:
    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
    Evaluator evaluator_;
};

std::shared_ptr<const HamiltonianModel> scalar_model(
    std::function<double(double)> energy
) {
    return std::make_shared<TestModel>(
        1,
        1,
        [energy = std::move(energy)](std::span<const double> point) {
            return Matrix{Complex{energy(point[0]), 0.0}};
        }
    );
}

ChargeResult estimate(
    std::shared_ptr<const HamiltonianModel> model,
    std::uint32_t error_depth
) {
    auto mesh = SpectralMesh(std::move(model), kTol, 0);
    return estimate_charge_on_current_mesh(mesh, 0.0, 10.0, error_depth);
}

void expect_positive(double value, const std::string &message) {
    expect(value > 1e-12, message);
}

double linear_triangle_occupied_volume(
    std::array<double, 3> energies,
    double level,
    double volume
) {
    std::sort(energies.begin(), energies.end());
    if (level <= energies[0]) {
        return 0.0;
    }
    if (level >= energies[2]) {
        return volume;
    }
    if (level < energies[1]) {
        return volume *
            (level - energies[0]) * (level - energies[0]) /
            ((energies[1] - energies[0]) *
             (energies[2] - energies[0]));
    }
    return volume * (
        1.0 -
        (energies[2] - level) * (energies[2] - level) /
        ((energies[2] - energies[0]) *
         (energies[2] - energies[1]))
    );
}

double linear_tetrahedron_occupied_volume(
    std::array<double, 4> energies,
    double level,
    double volume
) {
    std::sort(energies.begin(), energies.end());
    if (level <= energies.front()) {
        return 0.0;
    }
    if (level >= energies.back()) {
        return volume;
    }

    auto fraction = 0.0;
    for (std::size_t band = 0; band < energies.size(); ++band) {
        if (level <= energies[band]) {
            continue;
        }
        auto denominator = 1.0;
        for (std::size_t other = 0; other < energies.size(); ++other) {
            if (other != band) {
                denominator *= energies[other] - energies[band];
            }
        }
        fraction += std::pow(level - energies[band], 3) / denominator;
    }
    return volume * fraction;
}

void test_affine_scalar_has_zero_error() {
    const auto result = estimate(
        scalar_model([](double x) { return x - 0.25; }),
        1
    );

    expect_near(
        result.stopping_error,
        0.0,
        1e-12,
        "affine scalar interpolation error"
    );
    expect_eq(
        result.error_stats.conservative_fallbacks,
        0,
        "affine scalar fallback count"
    );
    expect(
        result.error_stats.micro_simplices >
            result.error_stats.root_simplices,
        "an uncertain affine scalar root should use its micro-simplices"
    );
}

void test_fixed_scalar_uses_the_cheap_terminal_path() {
    const auto result = estimate(
        scalar_model([](double) { return -1.0; }),
        1
    );

    expect_near(
        result.stopping_error,
        0.0,
        1e-12,
        "fixed scalar interpolation error"
    );
    expect_eq(
        result.error_stats.full_eigensystems,
        0,
        "a fixed leaf should not diagonalize its midpoint"
    );
    expect_eq(
        result.error_stats.norm_eigensystems,
        0,
        "a fixed leaf should use the Frobenius matrix norm"
    );
}

void test_curved_scalar_uses_depth_and_midpoints() {
    const auto model =
        scalar_model([](double x) { return x * x - 0.1; });
    const auto shallow = estimate(model, 0);
    const auto recursive = estimate(model, 1);

    expect_positive(
        shallow.stopping_error,
        "quadratic scalar should have a nonzero midpoint estimate"
    );
    expect_positive(
        recursive.stopping_error,
        "recursive quadratic scalar should retain a nonzero estimate"
    );
    expect(
        recursive.error_stats.micro_simplices >
            shallow.error_stats.micro_simplices,
        "depth one should visit more micro-simplices than depth zero"
    );
    expect(
        recursive.error_stats.hamiltonian_evaluations >
            shallow.error_stats.hamiltonian_evaluations,
        "depth one should sample more Hamiltonians than depth zero"
    );
    expect(
        recursive.error_stats.norm_eigensystems > 0,
        "terminal midpoint defects should evaluate matrix norms"
    );
    expect_eq(
        recursive.error_stats.root_simplices,
        recursive.stats.active_simplices,
        "one charge-error root per active mesh simplex"
    );
}

void test_two_band_space_bypasses_root_gate() {
    const auto model = std::make_shared<TestModel>(
        1,
        2,
        [](std::span<const double> point) {
            const auto x = point[0];
            return diagonal_matrix({x - 0.2, x - 0.3});
        }
    );
    const auto result = estimate(model, 1);

    expect_eq(
        result.error_stats.conservative_fallbacks,
        0,
        "q=2 must bypass the large-active-space gate"
    );
    expect(
        result.error_stats.micro_simplices >
            result.error_stats.root_simplices,
        "q=2 should recurse instead of using a root fallback"
    );
}

void test_large_active_space_uses_tight_fallback() {
    const auto model = std::make_shared<TestModel>(
        1,
        6,
        [](std::span<const double> point) {
            const auto x = point[0];
            return diagonal_matrix({
                -2.0,
                x - 0.1,
                x - 0.2,
                x - 0.3,
                2.0,
                3.0,
            });
        }
    );
    const auto result = estimate(model, 1);

    expect_near(result.value, 1.6, 1e-12, "large-q linear charge");
    expect_near(
        result.stopping_error,
        2.4,
        1e-12,
        "large-q distance to the tighter occupation interval"
    );
    expect_eq(
        result.error_stats.conservative_fallbacks,
        1,
        "large-q root fallback count"
    );
}

std::shared_ptr<const HamiltonianModel> embedded_quadratic_model() {
    return std::make_shared<TestModel>(
        1,
        8,
        [](std::span<const double> point) {
            const auto x = point[0];
            return diagonal_matrix({
                -4.0,
                -3.0,
                -2.0,
                x * x - 0.1,
                2.0,
                3.0,
                4.0,
                5.0,
            });
        }
    );
}

void test_safe_spectators_reduce_to_the_active_band() {
    const auto scalar = estimate(
        scalar_model([](double x) { return x * x - 0.1; }),
        1
    );
    const auto embedded = estimate(embedded_quadratic_model(), 1);

    expect_near(
        embedded.value - 3.0,
        scalar.value,
        1e-12,
        "safe spectators should only add integer charge"
    );
    expect_near(
        embedded.stopping_error,
        scalar.stopping_error,
        1e-12,
        "safe spectators should not change the active-band estimate"
    );
    expect(
        embedded.error_stats.schur_reductions > 0,
        "embedded active band should trigger a Schur reduction"
    );
    expect_eq(
        embedded.error_stats.minimum_active_dimension,
        1,
        "embedded model minimum active dimension"
    );
    expect(
        embedded.error_stats.safe_block_solves > 0,
        "Schur reduction should use direct safe-block solves"
    );
    expect_eq(
        embedded.error_stats.conservative_fallbacks,
        0,
        "separated spectators should not trigger a fallback"
    );
}

void test_nonsingular_coupled_reduction_uses_pointwise_schur_solves() {
    const auto model = std::make_shared<TestModel>(
        1,
        2,
        [](std::span<const double> point) {
            const auto x = point[0];
            const auto active = x - 0.25;
            constexpr auto safe = 2.0;
            const auto angle = 0.35 * x;
            const auto cosine = std::cos(angle);
            const auto sine = std::sin(angle);
            auto result = Matrix(4, Complex{0.0, 0.0});
            result[cm(0, 0, 2)] =
                Complex{cosine * cosine * active + sine * sine * safe, 0.0};
            result[cm(1, 1, 2)] =
                Complex{sine * sine * active + cosine * cosine * safe, 0.0};
            result[cm(0, 1, 2)] =
                Complex{cosine * sine * (active - safe), 0.0};
            result[cm(1, 0, 2)] = result[cm(0, 1, 2)];
            return result;
        }
    );
    const auto result = estimate(model, 0);

    expect_near(result.value, 0.25, 1e-12, "rotated affine-band charge");
    expect_eq(result.error_stats.schur_reductions, 1, "coupled Schur reductions");
    expect_eq(result.error_stats.safe_block_solves, 3, "coupled safe-block solves");
    expect_eq(result.error_stats.minimum_active_dimension, 1, "coupled active dimension");
    expect_eq(result.error_stats.singular_schur_failures, 0, "coupled singular blocks");
    expect_eq(result.error_stats.conservative_fallbacks, 0, "coupled fallbacks");
}

void test_new_microvertices_detect_a_visible_harmonic() {
    const auto model = scalar_model([](double x) {
        return x - 0.25 +
            0.2 * std::sin(2.0 * std::numbers::pi_v<double> * x);
    });
    const auto shallow = estimate(model, 0);
    const auto recursive = estimate(model, 1);

    expect_near(
        shallow.stopping_error,
        0.0,
        1e-12,
        "the chosen harmonic aliases at depth-zero samples"
    );
    expect_positive(
        recursive.stopping_error,
        "depth-one microvertices should reveal the harmonic"
    );
    expect(
        recursive.error_stats.hamiltonian_evaluations >
            shallow.error_stats.hamiltonian_evaluations,
        "harmonic detection should come from additional Hamiltonian samples"
    );
}

void test_certified_root_rechecks_reopened_band_curvature() {
    constexpr auto angle = 2.0;
    constexpr auto shift = 0.6;
    const auto sine = std::sin(angle);
    const auto cosine = std::cos(angle);
    const auto model = std::make_shared<TestModel>(
        1,
        2,
        [=](std::span<const double> point) {
            const auto x = point[0];
            const auto bump = 4.0 * shift * x * (1.0 - x);
            auto result = Matrix(4, Complex{0.0, 0.0});
            result[cm(0, 0, 2)] =
                Complex{(1.0 - x) + x * cosine + bump, 0.0};
            result[cm(1, 1, 2)] =
                Complex{-(1.0 - x) - x * cosine + bump, 0.0};
            result[cm(0, 1, 2)] = Complex{x * sine, 0.0};
            result[cm(1, 0, 2)] = Complex{x * sine, 0.0};
            return result;
        }
    );
    const auto result = estimate(model, 0);

    const auto one_minus_cosine = 1.0 - cosine;
    const auto discriminant =
        4.0 * one_minus_cosine * one_minus_cosine +
        64.0 * shift * shift;
    const auto crossing_product =
        (-2.0 * one_minus_cosine + std::sqrt(discriminant)) /
        (32.0 * shift * shift);
    const auto exact_charge =
        1.0 - std::sqrt(1.0 - 4.0 * crossing_product);
    const auto true_error = std::abs(result.value - exact_charge);

    expect_near(result.value, 1.0, 1e-12, "mixed-band linear charge");
    expect_eq(
        result.error_stats.initial_active_dimension_sum,
        0,
        "the mixed-band root should begin certified"
    );
    expect(
        result.stopping_error >= true_error,
        "a matrix defect that reopens bands must also sample band curvature"
    );
    expect(
        result.error_stats.full_eigensystems > 0,
        "reopened bands should trigger midpoint eigensystems"
    );
}

void test_singular_safe_block_falls_back_without_throwing() {
    const auto model = std::make_shared<TestModel>(
        1,
        3,
        [](std::span<const double> point) {
            const auto x = point[0];
            const auto active = x - 0.5;
            const auto safe = 16.0 * (x - 0.25) * (x - 0.25);
            const auto coupling = x * (x - 1.0);
            auto result = Matrix(9, Complex{0.0, 0.0});
            result[cm(0, 0, 3)] = Complex{-2.0, 0.0};
            result[cm(1, 1, 3)] = Complex{active, 0.0};
            result[cm(2, 2, 3)] = Complex{safe, 0.0};
            result[cm(1, 2, 3)] = Complex{coupling, 0.0};
            result[cm(2, 1, 3)] = Complex{coupling, 0.0};
            return result;
        }
    );
    const auto result = estimate(model, 1);

    expect(
        result.error_stats.singular_schur_failures > 0,
        "a singular pointwise safe block should be detected"
    );
    expect(
        result.error_stats.conservative_fallbacks > 0,
        "a singular pointwise safe block should use a conservative leaf"
    );
    expect(std::isfinite(result.stopping_error), "singular fallback estimate");
}

void test_quadratic_2d_uses_complete_depth_one_geometry() {
    constexpr auto offset = 0.3;
    const auto model = std::make_shared<TestModel>(
        2,
        1,
        [](std::span<const double> point) {
            return Matrix{
                Complex{
                    point[0] * point[0] +
                        point[1] * point[1] -
                        offset,
                    0.0,
                },
            };
        }
    );
    const auto result = estimate(model, 1);

    // The other root has the same children under x/y exchange.
    const auto shifted_charge = [&](double level) {
        constexpr auto child_volume = 1.0 / 8.0;
        return 2.0 * (
            linear_triangle_occupied_volume(
                {0.5 - offset, 1.25 - offset, 2.0 - offset},
                level,
                child_volume
            ) +
            linear_triangle_occupied_volume(
                {0.5 - offset, 1.0 - offset, 1.25 - offset},
                level,
                child_volume
            ) +
            linear_triangle_occupied_volume(
                {0.25 - offset, 1.0 - offset, 0.5 - offset},
                level,
                child_volume
            ) +
            linear_triangle_occupied_volume(
                {-offset, 0.25 - offset, 0.5 - offset},
                level,
                child_volume
            )
        );
    };

    // The longest child edge has squared length 1/2, hence defect 1/8.
    constexpr auto edge_midpoint_defect = 1.0 / 8.0;
    constexpr auto beta = (4.0 / 3.0) * edge_midpoint_defect;
    const auto lower_charge = shifted_charge(-beta);
    const auto upper_charge = shifted_charge(beta);
    const auto linear_charge =
        2.0 * linear_triangle_occupied_volume(
            {-offset, 1.0 - offset, 2.0 - offset},
            0.0,
            0.5
        );
    const auto expected_error = std::max(
        std::abs(linear_charge - lower_charge),
        std::abs(upper_charge - linear_charge)
    );

    expect_near(result.value, linear_charge, 1e-12, "2D quadratic charge");
    expect_near(
        result.stopping_error,
        expected_error,
        1e-12,
        "2D quadratic shifted-volume estimate"
    );
    expect(
        lower_charge < linear_charge && linear_charge < upper_charge,
        "the shifted levels should widen charge in the correct direction"
    );
    expect_eq(result.error_stats.root_simplices, 2, "2D root triangles");
    expect_eq(
        result.error_stats.micro_simplices,
        10,
        "each 2D root should visit itself and four children"
    );
    expect_eq(
        result.error_stats.terminal_simplices,
        8,
        "depth one should terminate on eight child triangles"
    );
    expect_eq(
        result.error_stats.conservative_fallbacks,
        0,
        "2D scalar quadratic should not use a fallback"
    );
}

void test_quadratic_3d_uses_beta_geometry_and_eight_children() {
    constexpr auto offset = 0.3;
    const auto model = std::make_shared<TestModel>(
        3,
        1,
        [](std::span<const double> point) {
            return Matrix{
                Complex{
                    point[0] * point[0] +
                        point[1] * point[1] +
                        point[2] * point[2] -
                        offset,
                    0.0,
                },
            };
        }
    );
    const auto shallow = estimate(model, 0);
    const auto recursive = estimate(model, 1);

    constexpr auto root_volume = 1.0 / 6.0;
    constexpr auto edge_midpoint_defect = 3.0 / 4.0;
    constexpr auto beta = (6.0 / 4.0) * edge_midpoint_defect;
    constexpr auto energies = std::array{
        -offset,
        1.0 - offset,
        2.0 - offset,
        3.0 - offset,
    };
    const auto linear_charge = 6.0 * linear_tetrahedron_occupied_volume(
        energies,
        0.0,
        root_volume
    );
    const auto lower_charge = 6.0 * linear_tetrahedron_occupied_volume(
        energies,
        -beta,
        root_volume
    );
    const auto upper_charge = 6.0 * linear_tetrahedron_occupied_volume(
        energies,
        beta,
        root_volume
    );
    const auto expected_error = std::max(
        std::abs(linear_charge - lower_charge),
        std::abs(upper_charge - linear_charge)
    );

    expect_near(shallow.value, linear_charge, 1e-12, "3D quadratic charge");
    expect_near(
        shallow.stopping_error,
        expected_error,
        1e-12,
        "3D quadratic shifted-volume estimate"
    );
    expect_eq(recursive.error_stats.root_simplices, 6, "3D root tetrahedra");
    expect_eq(recursive.error_stats.micro_simplices, 54, "3D root and child visits");
    expect_eq(recursive.error_stats.terminal_simplices, 48, "3D terminal children");
    expect_eq(recursive.error_stats.conservative_fallbacks, 0, "3D fallbacks");
}

}  // namespace

int main() {
    try {
        test_affine_scalar_has_zero_error();
        test_fixed_scalar_uses_the_cheap_terminal_path();
        test_curved_scalar_uses_depth_and_midpoints();
        test_quadratic_2d_uses_complete_depth_one_geometry();
        test_quadratic_3d_uses_beta_geometry_and_eight_children();
        test_two_band_space_bypasses_root_gate();
        test_large_active_space_uses_tight_fallback();
        test_safe_spectators_reduce_to_the_active_band();
        test_nonsingular_coupled_reduction_uses_pointwise_schur_solves();
        test_new_microvertices_detect_a_visible_harmonic();
        test_certified_root_rechecks_reopened_band_curvature();
        test_singular_safe_block_falls_back_without_throwing();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
