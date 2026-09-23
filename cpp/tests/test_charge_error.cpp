#include "test_helpers.h"

#include "integration/charge.h"
#include "integration/charge_error/cached_model.h"
#include "integration/charge_error/cut_disagreement.h"
#include "integration/charge_error/projected_schur.h"
#include "integration/charge_profile.h"

#include <fermisimplex/hamiltonian.h>
#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include <adaptivesimplex/adaptive/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fermisimplex;
using namespace fermisimplex::test;

using Matrix = std::vector<Complex>;
using Evaluator = std::function<Matrix(std::span<const double>)>;

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

class DelegatingModel final : public HamiltonianModel {
public:
    explicit DelegatingModel(
        std::shared_ptr<const HamiltonianModel> delegate
    ) : delegate_(std::move(delegate)) {}

    std::size_t ndim() const noexcept override {
        return delegate_->ndim();
    }

    std::size_t ndof() const noexcept override {
        return delegate_->ndof();
    }

    Matrix evaluate(std::span<const double> point) const override {
        return delegate_->evaluate(point);
    }

private:
    std::shared_ptr<const HamiltonianModel> delegate_;
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
    std::uint32_t error_depth,
    double mu = 0.0
) {
    auto mesh = SpectralMesh(std::move(model), kTol, 0);
    return integrate_charge(
        mesh,
        mu,
        adaptivesimplex::adaptive::Options{
            .target_error = 10.0,
            .max_refinements = 0,
            .preview_depth = 0,
            .min_refinement_batch_size = 1,
            .max_refinement_batch_size = 100,
        },
        error_depth
    );
}

Matrix adjoint(Matrix matrix, std::size_t size) {
    auto result = Matrix(matrix.size());
    for (std::size_t column = 0; column < size; ++column) {
        for (std::size_t row = 0; row < size; ++row) {
            result[cm(row, column, size)] =
                std::conj(matrix[cm(column, row, size)]);
        }
    }
    return result;
}

std::shared_ptr<const TightBindingModel> dense_tight_binding_model(
    std::size_t active
) {
    constexpr auto size = std::size_t{20};
    if (active == 0 || active > 8) {
        throw std::runtime_error("invalid test active-space size");
    }
    const auto occupied = (size - active) / 2;
    const auto empty_begin = occupied + active;
    auto onsite = Matrix(size * size);
    for (std::size_t band = 0; band < occupied; ++band) {
        onsite[cm(band, band, size)] = Complex{
            -20.0 - static_cast<double>(band), 0.0
        };
    }
    const auto center = 0.5 * static_cast<double>(active - 1);
    for (std::size_t band = 0; band < active; ++band) {
        onsite[cm(occupied + band, occupied + band, size)] = Complex{
            -0.2 + 0.015 * (static_cast<double>(band) - center), 0.0
        };
    }
    for (std::size_t band = empty_begin; band < size; ++band) {
        onsite[cm(band, band, size)] = Complex{
            20.0 + static_cast<double>(band), 0.0
        };
    }
    auto x_hopping = Matrix(size * size);
    auto y_hopping = Matrix(size * size);
    for (std::size_t band = 0; band < active; ++band) {
        x_hopping[cm(occupied + band, occupied + band, size)] =
            Complex{0.35, 0.0};
        y_hopping[cm(occupied + band, occupied + band, size)] =
            Complex{0.25, 0.0};
    }
    for (std::size_t column = 0; column < size; ++column) {
        for (std::size_t row = 0; row < size; ++row) {
            if (row == column) {
                continue;
            }
            const auto seed =
                static_cast<double>((row + 2) * (column + 3));
            x_hopping[cm(row, column, size)] += 0.004 * Complex{
                std::sin(seed), std::cos(0.7 * seed)
            };
            y_hopping[cm(row, column, size)] += 0.004 * Complex{
                std::cos(0.4 * seed), std::sin(0.9 * seed)
            };
        }
    }
    return std::make_shared<TightBindingModel>(
        std::vector<HoppingTerm>{
            {
                .lattice_vector = {0, 0},
                .matrix = std::move(onsite),
            },
            {
                .lattice_vector = {1, 0},
                .matrix = x_hopping,
            },
            {
                .lattice_vector = {-1, 0},
                .matrix = adjoint(x_hopping, size),
            },
            {
                .lattice_vector = {0, 1},
                .matrix = y_hopping,
            },
            {
                .lattice_vector = {0, -1},
                .matrix = adjoint(y_hopping, size),
            },
        }
    );
}

void expect_positive(double value, const std::string &message) {
    expect(value > 1e-12, message);
}

void test_two_affine_cut_disagreement() {
    using integration_detail::charge_error_detail::cut_disagreement;
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

void test_point_cache_uses_exact_dyadic_keys() {
    namespace error_detail = fermisimplex::integration_detail;

    const auto point = error_detail::CachedPoint({1, -2}, 50);
    const auto equivalent = error_detail::CachedPoint({2, -4}, 51);
    const auto nearby = error_detail::CachedPoint({3, -4}, 51);
    expect(point == equivalent, "dyadic key canonicalization");
    expect(!(point == nearby), "nearby dyadic keys must remain distinct");

    auto cache = error_detail::PointCache{};
    const auto &stored = cache.store_hamiltonian(
        point,
        Matrix{Complex{1.0, 0.0}}
    );
    expect(
        cache.find_hamiltonian(equivalent) == &stored,
        "canonical-equivalent dyadic points should share one record"
    );
    expect(
        cache.find_hamiltonian(nearby) == nullptr,
        "a nearby exact dyadic point should miss the cache"
    );
    cache.store_hamiltonian(nearby, Matrix{Complex{2.0, 0.0}});
    expect_eq(cache.size(), 2, "distinct exact dyadic cache records");
}

void test_point_cache_lazy_payloads_and_lifetime_accounting() {
    namespace error_detail = fermisimplex::integration_detail;

    auto profile = error_detail::ChargeProfile{};
    auto final_payload = std::size_t{0};
    {
        auto cache = error_detail::PointCache{&profile};
        const auto matrix_point = error_detail::CachedPoint({0}, 0);
        const auto values_point = error_detail::CachedPoint({1}, 1);
        const auto full_point = error_detail::CachedPoint({1}, 2);
        const auto upgrade_point = error_detail::CachedPoint({3}, 2);

        const auto &matrix = cache.store_hamiltonian(
            matrix_point,
            Matrix{
                Complex{1.0, 0.0},
                Complex{0.0, 0.0},
                Complex{0.0, 0.0},
                Complex{2.0, 0.0},
            }
        );
        auto expected_payload = matrix.capacity() * sizeof(Complex);
        expect_eq(
            cache.payload_bytes(),
            expected_payload,
            "matrix-only cache payload"
        );
        expect(
            cache.find_spectrum(matrix_point) == nullptr,
            "matrix-only record must not create a spectrum"
        );

        const auto &values = cache.store_eigenvalues(
            values_point,
            {-3.0, 0.25, 4.0}
        );
        expected_payload += values.capacity() * sizeof(double);
        expect_eq(
            cache.payload_bytes(),
            expected_payload,
            "values-only cache payload"
        );
        expect(
            cache.find_hamiltonian(values_point) == nullptr,
            "values-only record must not create a Hamiltonian"
        );
        expect(
            cache.find_spectrum(values_point) != nullptr &&
                !cache.find_spectrum(values_point)->has_vectors(),
            "values-only record must explicitly omit eigenvectors"
        );

        const auto &full = cache.store_spectrum(
            full_point,
            error_detail::CachedSpectrum{
                .eigenvalues = {-4.0, 5.0},
                .eigenvectors = Matrix{
                    Complex{1.0, 0.0},
                    Complex{0.0, 0.0},
                    Complex{0.0, 0.0},
                    Complex{1.0, 0.0},
                },
            }
        );
        expected_payload +=
            full.eigenvalues.capacity() * sizeof(double) +
            full.eigenvectors->capacity() * sizeof(Complex);
        expect_eq(
            cache.payload_bytes(),
            expected_payload,
            "full-spectrum cache payload"
        );
        expect(full.has_vectors(), "full spectrum must retain eigenvectors");

        const auto payload_before_values_request = cache.payload_bytes();
        const auto &retained_values = cache.store_eigenvalues(
            full_point,
            {91.0, 92.0}
        );
        expect_near(
            retained_values.front(),
            -4.0,
            0.0,
            "values storage must retain an existing full spectrum"
        );
        expect(
            cache.find_spectrum(full_point)->has_vectors(),
            "values storage must not discard cached eigenvectors"
        );
        expect_eq(
            cache.payload_bytes(),
            payload_before_values_request,
            "redundant values storage must not change payload"
        );

        const auto &upgrade_values = cache.store_eigenvalues(
            upgrade_point,
            {-2.0, 0.0, 3.0}
        );
        const auto old_upgrade_payload =
            upgrade_values.capacity() * sizeof(double);
        expected_payload += old_upgrade_payload;
        const auto &upgraded = cache.store_spectrum(
            upgrade_point,
            error_detail::CachedSpectrum{
                .eigenvalues = {-2.0, 0.0, 3.0},
                .eigenvectors = Matrix{
                    Complex{1.0, 0.0}, Complex{0.0, 0.0},
                    Complex{0.0, 0.0}, Complex{0.0, 0.0},
                    Complex{1.0, 0.0}, Complex{0.0, 0.0},
                    Complex{0.0, 0.0}, Complex{0.0, 0.0},
                    Complex{1.0, 0.0},
                },
            }
        );
        expected_payload -= old_upgrade_payload;
        expected_payload +=
            upgraded.eigenvalues.capacity() * sizeof(double) +
            upgraded.eigenvectors->capacity() * sizeof(Complex);
        expect(upgraded.has_vectors(), "values-to-vectors cache upgrade");
        expect_eq(
            cache.size(),
            4,
            "one record per lazy point payload"
        );
        expect_eq(
            cache.payload_bytes(),
            expected_payload,
            "upgraded cache payload accounting"
        );
        expect_eq(
            profile.error_cache_live_records,
            cache.size(),
            "profile live cache records"
        );
        expect_eq(
            profile.error_cache_peak_records,
            cache.size(),
            "profile peak cache records"
        );
        expect_eq(
            profile.error_cache_live_payload_bytes,
            cache.payload_bytes(),
            "profile live cache payload"
        );
        expect(
            profile.error_cache_peak_payload_bytes >=
                profile.error_cache_live_payload_bytes,
            "profile peak payload must cover live payload"
        );
        final_payload = cache.payload_bytes();
    }

    expect(final_payload > 0, "cache should have retained a positive payload");
    expect_eq(
        profile.error_cache_live_records,
        0,
        "cache destruction must release all live records"
    );
    expect_eq(
        profile.error_cache_live_payload_bytes,
        0,
        "cache destruction must release all live payload"
    );
    expect(
        profile.error_cache_peak_records >= 4,
        "cache destruction must preserve peak records"
    );
    expect(
        profile.error_cache_peak_payload_bytes >= final_payload,
        "cache destruction must preserve peak payload"
    );
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
    expect_near(
        result.density_cut_error, 0.0, 1e-12,
        "affine child and reported cuts agree"
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

void test_balanced_pockets_have_cut_disagreement_without_charge_bias() {
    const auto result = estimate(
        scalar_model([](double x) {
            return x - 0.5 + 0.4 * std::sin(2.0 * std::numbers::pi * x);
        }),
        2
    );
    // e(1-x)=-e(x), so both true and reported total charges are 1/2.
    // The quarter-point samples nevertheless uncover oppositely signed
    // occupation mistakes on either side of the Brillouin zone.
    expect_near(result.value, 0.5, 1e-12, "balanced-pocket charge");
    expect(
        result.density_cut_error > 2.0 * result.stopping_error + 1e-3,
        "unsigned child-cut disagreement survives charge cancellation"
    );
}

void test_fixed_root_uses_shifted_width_for_hidden_pocket() {
    const auto result = estimate(
        scalar_model([](double x) {
            return (x - 0.5) * (x - 0.5) - 0.1;
        }),
        2
    );
    expect_near(result.value, 0.0, 1e-12, "hidden-pocket root cut");
    expect(
        result.density_cut_error >= 2.0 * std::sqrt(0.1),
        "midpoint uncertainty should cover this analytic hidden pocket"
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
    expect_eq(
        recursive.error_stats.norm_eigensystems,
        0,
        "scalar midpoint defects should use the analytic spectral norm"
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

void test_curved_two_band_uses_analytic_defect_norm() {
    const auto model = std::make_shared<TestModel>(
        1,
        2,
        [](std::span<const double> point) {
            const auto x2 = point[0] * point[0];
            return diagonal_matrix({x2 - 0.1, x2 - 0.2});
        }
    );
    const auto result = estimate(model, 1);

    expect_positive(
        result.stopping_error,
        "curved two-band model should have a nonzero midpoint estimate"
    );
    expect_eq(
        result.error_stats.norm_eigensystems,
        0,
        "two-band midpoint defects should use the analytic spectral norm"
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
    expect_near(
        result.density_cut_error, 3.0, 1e-12,
        "large-q root gate reports its occupation-width cut indicator"
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
    expect_near(
        embedded.density_cut_error,
        scalar.density_cut_error,
        1e-12,
        "frozen spectators should not change the cut disagreement"
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
        embedded.error_stats.schur_evaluations > 0,
        "Schur reduction should evaluate the corrected surrogate"
    );
    expect_eq(
        embedded.error_stats.conservative_fallbacks,
        0,
        "separated spectators should not trigger a fallback"
    );
}

void test_complex_basis_preserves_the_active_band() {
    const auto scalar = estimate(
        scalar_model([](double x) { return x * x - 0.1; }),
        1
    );
    const auto model = std::make_shared<TestModel>(
        1,
        2,
        [](std::span<const double> point) {
            const auto active = point[0] * point[0] - 0.1;
            constexpr auto safe = -2.0;
            const auto diagonal = 0.5 * (safe + active);
            const auto coupling = Complex{0.0, 0.5 * (active - safe)};
            return Matrix{
                diagonal,
                std::conj(coupling),
                coupling,
                diagonal,
            };
        }
    );
    const auto embedded = estimate(model, 1);

    expect_near(
        embedded.value - 1.0,
        scalar.value,
        1e-12,
        "complex safe state should only add integer charge"
    );
    expect_near(
        embedded.stopping_error,
        scalar.stopping_error,
        1e-12,
        "complex basis should preserve the active-band estimate"
    );
    expect(embedded.error_stats.schur_reductions > 0, "complex Schur reduction");
    expect_eq(embedded.error_stats.schur_failures, 0, "complex Schur failures");
}

void test_coupled_reduction_uses_frozen_correction() {
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
    expect_eq(result.error_stats.schur_evaluations, 3, "coupled Schur evaluations");
    expect_eq(
        result.error_stats.minimum_active_dimension,
        1,
        "coupled active dimension"
    );
    expect_eq(result.error_stats.schur_failures, 0, "coupled Schur failures");
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

void test_reopened_single_band_reduces_before_exact_defects() {
    const auto model = std::make_shared<TestModel>(
        1,
        4,
        [](std::span<const double> point) {
            const auto x = point[0];
            const auto active = -0.2 + 1.6 * x * (1.0 - x);
            return diagonal_matrix({-3.0, active, 5.0, 6.0});
        }
    );
    const auto result = estimate(model, 0);

    expect_eq(
        result.error_stats.initial_active_dimension_sum,
        0,
        "the curved embedded root should begin certified"
    );
    expect_eq(
        result.error_stats.minimum_active_dimension,
        1,
        "curvature should reopen exactly one active band"
    );
    expect_eq(
        result.error_stats.schur_reductions,
        1,
        "the reopened band should be reduced at the terminal"
    );
    expect_eq(
        result.error_stats.full_eigensystems,
        0,
        "late scalar reduction should avoid midpoint full eigensystems"
    );
    expect_eq(
        result.error_stats.norm_eigensystems,
        0,
        "late scalar reduction should avoid full defect eigensystems"
    );
    expect_positive(
        result.stopping_error,
        "the reopened band should retain a nonzero charge error"
    );
}

void test_frozen_correction_avoids_pointwise_safe_singularity() {
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

    expect_eq(
        result.error_stats.schur_failures,
        0,
        "frozen correction should not factor pointwise safe blocks"
    );
    expect_eq(
        result.error_stats.conservative_fallbacks,
        0,
        "an interior pointwise singularity should not force a fallback"
    );
    expect(
        std::isfinite(result.stopping_error),
        "frozen-correction estimate"
    );
}

void test_projected_tight_binding_matches_general_schur_evaluation() {
    for (const auto active : {1U, 2U, 3U, 4U, 8U}) {
        const auto tight_binding = dense_tight_binding_model(active);
        const auto projected = estimate(tight_binding, 2, 0.07);
        const auto general = estimate(
            std::make_shared<DelegatingModel>(tight_binding), 2, 0.07
        );

        expect_near(
            projected.value,
            general.value,
            3e-12,
            "projected tight-binding charge"
        );
        expect_near(
            projected.stopping_error,
            general.stopping_error,
            3e-11,
            "projected tight-binding stopping error"
        );
        expect_eq(
            projected.error_stats.schur_reductions,
            general.error_stats.schur_reductions,
            "projected tight-binding Schur reductions"
        );
        expect_eq(
            projected.error_stats.schur_evaluations,
            general.error_stats.schur_evaluations,
            "projected tight-binding Schur evaluations"
        );
        expect_eq(
            projected.error_stats.root_simplices,
            general.error_stats.root_simplices,
            "projected tight-binding root simplices"
        );
        expect_eq(
            projected.error_stats.full_eigensystems,
            general.error_stats.full_eigensystems,
            "projected tight-binding full eigensystems"
        );
        expect_eq(
            projected.error_stats.reduced_eigensystems,
            general.error_stats.reduced_eigensystems,
            "projected tight-binding reduced eigensystems"
        );
        expect_eq(
            projected.error_stats.norm_eigensystems,
            general.error_stats.norm_eigensystems,
            "projected tight-binding norm eigensystems"
        );
        expect_eq(
            projected.error_stats.micro_simplices,
            general.error_stats.micro_simplices,
            "projected tight-binding microsimplices"
        );
        expect_eq(
            projected.error_stats.terminal_simplices,
            general.error_stats.terminal_simplices,
            "projected tight-binding terminal simplices"
        );
        expect_eq(
            projected.error_stats.initial_active_dimension_sum,
            general.error_stats.initial_active_dimension_sum,
            "projected tight-binding initial active dimension"
        );
        expect_eq(
            projected.error_stats.terminal_active_dimension_sum,
            general.error_stats.terminal_active_dimension_sum,
            "projected tight-binding terminal active dimension"
        );
        expect_eq(
            projected.error_stats.minimum_active_dimension,
            general.error_stats.minimum_active_dimension,
            "projected tight-binding minimum active dimension"
        );
        expect_eq(
            projected.stats.refinements,
            general.stats.refinements,
            "projected tight-binding adaptive refinements"
        );
        expect_eq(
            projected.stats.simplex_visits,
            general.stats.simplex_visits,
            "projected tight-binding adaptive simplex visits"
        );
        expect_eq(
            projected.error_stats.root_simplices,
            general.error_stats.root_simplices,
            "projected tight-binding root simplices"
        );
        expect_eq(
            projected.error_stats.full_eigensystems,
            general.error_stats.full_eigensystems,
            "projected tight-binding full eigensystems"
        );
        expect_eq(
            projected.error_stats.reduced_eigensystems,
            general.error_stats.reduced_eigensystems,
            "projected tight-binding reduced eigensystems"
        );
        expect_eq(
            projected.error_stats.norm_eigensystems,
            general.error_stats.norm_eigensystems,
            "projected tight-binding norm eigensystems"
        );
        expect_eq(
            projected.error_stats.micro_simplices,
            general.error_stats.micro_simplices,
            "projected tight-binding microsimplices"
        );
        expect_eq(
            projected.error_stats.terminal_simplices,
            general.error_stats.terminal_simplices,
            "projected tight-binding terminal simplices"
        );
        expect_eq(
            projected.error_stats.initial_active_dimension_sum,
            general.error_stats.initial_active_dimension_sum,
            "projected tight-binding initial active dimension"
        );
        expect_eq(
            projected.error_stats.terminal_active_dimension_sum,
            general.error_stats.terminal_active_dimension_sum,
            "projected tight-binding terminal active dimension"
        );
        expect_eq(
            projected.error_stats.minimum_active_dimension,
            general.error_stats.minimum_active_dimension,
            "projected tight-binding minimum active dimension"
        );
        expect_eq(
            projected.stats.refinements,
            general.stats.refinements,
            "projected tight-binding adaptive refinements"
        );
        expect_eq(
            projected.stats.simplex_visits,
            general.stats.simplex_visits,
            "projected tight-binding adaptive simplex visits"
        );
        expect_eq(
            projected.error_stats.hamiltonian_evaluations,
            0,
            "projected tight binding should avoid error-stage Hamiltonians"
        );
        expect(
            general.error_stats.hamiltonian_evaluations > 0,
            "the general Schur path should sample dense Hamiltonians"
        );
        expect_eq(
            projected.error_stats.schur_failures,
            0,
            "projected tight-binding Schur failures"
        );
        expect_eq(
            projected.error_stats.conservative_fallbacks,
            0,
            "projected tight-binding fallbacks"
        );
        expect_eq(
            projected.error_stats.conservative_fallbacks,
            general.error_stats.conservative_fallbacks,
            "projected tight-binding fallback equivalence"
        );
        expect_eq(
            projected.error_stats.schur_failures,
            general.error_stats.schur_failures,
            "projected tight-binding failure equivalence"
        );
        expect_eq(
            projected.error_stats.conservative_fallbacks,
            general.error_stats.conservative_fallbacks,
            "projected tight-binding fallback equivalence"
        );
        expect_eq(
            projected.error_stats.schur_failures,
            general.error_stats.schur_failures,
            "projected tight-binding failure equivalence"
        );
    }
}

void test_projected_polynomial_matches_dense_schur_directly() {
    namespace error_detail = fermisimplex::integration_detail;

    constexpr auto mu = 0.07;
    const auto anchor = std::array{0.137, 0.283};
    const auto points = std::array{
        std::array{0.0, 0.0},
        std::array{0.173, 0.419},
        std::array{0.5, 0.25},
        std::array{0.913, 0.071},
    };

    for (const auto active : {1U, 2U, 3U, 4U, 8U}) {
        const auto tight_binding = dense_tight_binding_model(active);
        auto eigensystem = SpectralMesh(tight_binding, kTol, 0).spectrum(
            anchor
        );
        for (auto &eigenvalue : eigensystem.eigenvalues) {
            eigenvalue -= mu;
        }
        const auto eigensystem_view = error_detail::SchurEigensystemView{
            .eigenvalues = eigensystem.eigenvalues,
            .eigenvectors = eigensystem.eigenvectors,
        };
        const auto active_begin = (tight_binding->ndof() - active) / 2;
        const auto active_end = active_begin + active;
        const auto projected_layer = error_detail::make_schur_layer(
            eigensystem_view,
            active_begin,
            active_end,
            false
        );
        const auto dense_layer = error_detail::make_schur_layer(
            eigensystem_view,
            active_begin,
            active_end,
            true
        );
        const auto projected = error_detail::make_projected_hopping_model(
            *tight_binding,
            mu,
            projected_layer
        );
        expect(
            projected.has_value(),
            "projected polynomial construction for q=" +
                std::to_string(active)
        );

        for (std::size_t point_index = 0;
             point_index < points.size();
             ++point_index) {
            auto shifted = tight_binding->evaluate(points[point_index]);
            for (std::size_t band = 0;
                 band < tight_binding->ndof();
                 ++band) {
                shifted[cm(band, band, tight_binding->ndof())] -= mu;
            }
            auto stats = ChargeErrorStats{};
            const auto dense = error_detail::apply_schur_layer(
                shifted,
                dense_layer,
                stats
            );
            const auto polynomial = projected->evaluate(points[point_index]);
            expect_eq(
                polynomial.size(),
                dense.size(),
                "projected polynomial matrix size"
            );
            const auto context =
                "direct projected polynomial q=" +
                std::to_string(active) + " point=" +
                std::to_string(point_index);
            for (std::size_t index = 0; index < dense.size(); ++index) {
                expect_near(
                    polynomial[index].real(),
                    dense[index].real(),
                    2e-10,
                    context + " real"
                );
                expect_near(
                    polynomial[index].imag(),
                    dense[index].imag(),
                    2e-10,
                    context + " imaginary"
                );
            }
            expect_eq(
                stats.schur_evaluations,
                1,
                "one dense Schur evaluation per comparison"
            );
        }
    }
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
    expect_positive(
        result.density_cut_error - (upper_charge - lower_charge),
        "curved child cuts disagree with the original charge cut"
    );
    const auto exact_cut_mismatch =
        std::numbers::pi * offset / 4.0 - offset * offset / 2.0;
    expect(
        result.density_cut_error >= exact_cut_mismatch,
        "2D cut indicator should cover this analytic quadratic mismatch"
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
    expect_positive(
        recursive.density_cut_error,
        "recursive cut indicator should retain a visible 3D error"
    );
    const auto exact_cut_mismatch =
        std::numbers::pi * std::pow(offset, 1.5) / 6.0 -
        std::pow(offset, 3) / 6.0;
    expect(
        recursive.density_cut_error >= exact_cut_mismatch,
        "3D cut indicator should cover this analytic quadratic mismatch"
    );
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
        test_density_cut_error_result_boundary();
        test_two_affine_cut_disagreement();
        test_point_cache_uses_exact_dyadic_keys();
        test_point_cache_lazy_payloads_and_lifetime_accounting();
        test_affine_scalar_has_zero_error();
        test_balanced_pockets_have_cut_disagreement_without_charge_bias();
        test_fixed_root_uses_shifted_width_for_hidden_pocket();
        test_fixed_scalar_uses_the_cheap_terminal_path();
        test_curved_scalar_uses_depth_and_midpoints();
        test_quadratic_2d_uses_complete_depth_one_geometry();
        test_quadratic_3d_uses_beta_geometry_and_eight_children();
        test_two_band_space_bypasses_root_gate();
        test_curved_two_band_uses_analytic_defect_norm();
        test_large_active_space_uses_tight_fallback();
        test_safe_spectators_reduce_to_the_active_band();
        test_complex_basis_preserves_the_active_band();
        test_coupled_reduction_uses_frozen_correction();
        test_new_microvertices_detect_a_visible_harmonic();
        test_certified_root_rechecks_reopened_band_curvature();
        test_reopened_single_band_reduces_before_exact_defects();
        test_frozen_correction_avoids_pointwise_safe_singularity();
        test_projected_polynomial_matches_dense_schur_directly();
        test_projected_tight_binding_matches_general_schur_evaluation();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
