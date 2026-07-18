#include "test_helpers.h"

#include <array>
#include <complex>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

namespace {

using namespace fermisimplex;
using namespace fermisimplex::test;

class FixedMatrixModel final : public HamiltonianModel {
public:
    explicit FixedMatrixModel(std::vector<Complex> matrix)
        : matrix_(std::move(matrix)) {}

    std::size_t ndim() const noexcept override { return 1; }
    std::size_t ndof() const noexcept override { return 2; }

    std::vector<Complex> evaluate(std::span<const double>) const override {
        return matrix_;
    }

private:
    std::vector<Complex> matrix_;
};

void test_tight_binding_consolidates_and_validates_hermiticity() {
    const auto model = TightBindingModel(std::vector<HoppingTerm>{
        {.lattice_vector = {1}, .matrix = {1.0}},
        {.lattice_vector = {1}, .matrix = {-0.5}},
        {.lattice_vector = {-1}, .matrix = {0.5}},
    });
    expect_near(
        model.evaluate(std::array<double, 1>{0.0}).front().real(),
        1.0,
        kTol,
        "consolidated hopping evaluation"
    );

    const auto cancelled = TightBindingModel(std::vector<HoppingTerm>{
        {.lattice_vector = {1}, .matrix = {1.0}},
        {.lattice_vector = {1}, .matrix = {-1.0}},
    });
    expect_near(
        std::abs(cancelled.evaluate(std::array<double, 1>{0.37}).front()),
        0.0,
        kTol,
        "cancelled hopping terms should not affect evaluation"
    );

    const auto roundoff_pair = TightBindingModel(std::vector<HoppingTerm>{
        {.lattice_vector = {1}, .matrix = {1e12}},
        {.lattice_vector = {-1}, .matrix = {1e12 + 1e-3}},
    });
    const auto roundoff_value =
        roundoff_pair.evaluate(std::array<double, 1>{0.25}).front();
    expect_near(
        roundoff_value.imag(),
        0.0,
        kTol,
        "accepted opposite hoppings should be canonicalized exactly"
    );
    expect_runtime_error(
        [&] {
            (void)roundoff_pair.evaluate(std::array<double, 1>{
                std::numeric_limits<double>::infinity(),
            });
        },
        "coordinates must be finite",
        "tight-binding evaluation should reject non-finite coordinates"
    );

    expect_runtime_error(
        [] {
            (void)TightBindingModel(std::vector<HoppingTerm>{
                {.lattice_vector = {1}, .matrix = {0.5}},
            });
        },
        "opposite partner",
        "tight binding should reject a missing Hermitian partner"
    );
    expect_runtime_error(
        [] {
            (void)TightBindingModel(std::vector<HoppingTerm>{
                {.lattice_vector = {0}, .matrix = {Complex{0.0, 1.0}}},
            });
        },
        "adjoint",
        "the zero-vector hopping should itself be Hermitian"
    );
    expect_runtime_error(
        [] {
            (void)TightBindingModel(std::vector<HoppingTerm>{
                {.lattice_vector = {1}, .matrix = {0.5}},
                {.lattice_vector = {-1}, .matrix = {0.25}},
            });
        },
        "adjoint",
        "opposite hopping matrices should be adjoints"
    );
    expect_runtime_error(
        [] {
            (void)TightBindingModel(std::vector<HoppingTerm>{
                {
                    .lattice_vector = {0},
                    .matrix = {std::numeric_limits<double>::quiet_NaN()},
                },
            });
        },
        "entries must be finite",
        "tight binding should reject non-finite entries"
    );
    expect_runtime_error(
        [] {
            (void)TightBindingModel(std::vector<HoppingTerm>{
                {
                    .lattice_vector = {std::numeric_limits<std::int64_t>::min()},
                    .matrix = {0.0},
                },
            });
        },
        "must be negatable",
        "tight binding should reject lattice vectors that cannot be negated"
    );
}

void test_mesh_converts_user_curvature_to_simplex_error() {
    constexpr auto winding = 3;
    auto model = winding_model(winding);
    const auto expected_curvature = winding_curvature_bound(winding);

    auto mesh = SpectralMesh(model, kTol, 4);
    const auto simplex_id = first_active_simplex(mesh.geometry());
    expect_near(
        mesh.linearization_error_bound(simplex_id, expected_curvature),
        0.5 * expected_curvature / 256.0,
        1e-12,
        "mesh linearization error"
    );

    expect_runtime_error(
        [&] { (void)model->evaluate(std::span<const double>{}); },
        "point dimension",
        "tight-binding evaluation should validate point dimension"
    );
}

void test_spectral_mesh_validates_hamiltonian_values() {
    const auto point = std::array<double, 1>{0.0};

    auto nonhermitian = SpectralMesh(std::make_shared<FixedMatrixModel>(
        std::vector<Complex>{0.0, 0.0, 1.0, 0.0}
    ));
    expect_runtime_error(
        [&] { (void)nonhermitian.spectrum(point); },
        "Hamiltonian must be Hermitian",
        "spectral evaluation should reject non-Hermitian matrices"
    );

    auto nonfinite = SpectralMesh(std::make_shared<FixedMatrixModel>(
        std::vector<Complex>{
            std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0,
        }
    ));
    expect_runtime_error(
        [&] { (void)nonfinite.spectrum(point); },
        "Hamiltonian entries must be finite",
        "spectral evaluation should reject non-finite matrices"
    );

    const auto invalid_point = std::array<double, 1>{
        std::numeric_limits<double>::quiet_NaN(),
    };
    auto finite = SpectralMesh(constant_insulator());
    expect_runtime_error(
        [&] { (void)finite.spectrum(invalid_point); },
        "point coordinates must be finite",
        "spectral evaluation should reject non-finite coordinates"
    );
}

}  // namespace

int main() {
    try {
        test_tight_binding_consolidates_and_validates_hermiticity();
        test_mesh_converts_user_curvature_to_simplex_error();
        test_spectral_mesh_validates_hamiltonian_values();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
