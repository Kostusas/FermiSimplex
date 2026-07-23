#include "integration/charge.h"
#include "test_helpers.h"

#include <fermisimplex/integration.h>
#include <fermisimplex/fermi_surface.h>

#include <adaptivesimplex/adaptive/types.h>

#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace {

using namespace fermisimplex;
using namespace fermisimplex::test;
namespace adaptive = adaptivesimplex::adaptive;

class AffineXModel final : public HamiltonianModel {
public:
    std::size_t ndim() const noexcept override { return 2; }
    std::size_t ndof() const noexcept override { return 1; }

    std::vector<Complex> evaluate(std::span<const double> point) const override {
        return {Complex{point[0], 0.0}};
    }
};

class AffineConeModel final : public HamiltonianModel {
public:
    std::size_t ndim() const noexcept override { return 2; }
    std::size_t ndof() const noexcept override { return 2; }

    std::vector<Complex> evaluate(std::span<const double> point) const override {
        const auto x = point[0] - 0.5;
        const auto z = point[1] - 0.5;
        return {
            Complex{z, 0.0},
            Complex{x, 0.0},
            Complex{x, 0.0},
            Complex{-z, 0.0},
        };
    }
};

adaptive::Options fixed_options(double target_error) {
    return adaptive::Options{
        .target_error = target_error,
        .max_refinements = 0,
        .preview_depth = 1,
        .min_refinement_batch_size = 1,
        .max_refinement_batch_size = 100,
    };
}

std::shared_ptr<TightBindingModel> aliased_pocket() {
    return std::make_shared<TightBindingModel>(
        std::vector<HoppingTerm>{
            {.lattice_vector = {0}, .matrix = {0.5}},
            {.lattice_vector = {8}, .matrix = {0.5}},
            {.lattice_vector = {-8}, .matrix = {0.5}},
        }
    );
}

void test_constant_insulator_charge_is_certified() {
    auto mesh = SpectralMesh(constant_insulator());
    const auto result = integrate_charge(mesh, 0.0, fixed_options(0.0), 0.0);

    expect_near(result.value, 1.0, kTol, "constant insulator charge");
    expect_near(result.stopping_error, 0.0, kTol, "constant insulator estimate");
    expect_near(
        result.certified_error_bound,
        0.0,
        kTol,
        "constant insulator certificate"
    );
    expect_eq(result.stats.evaluations, 5, "charge vertex evaluations");
    expect_eq(
        result.stats.simplex_visits,
        6,
        "charge must count coarse and preview simplex visits"
    );
}

void test_fermi_surface_counts_work() {
    auto mesh = SpectralMesh(constant_insulator());
    const auto result = fermi_surface(mesh, 0.0, 0.1);

    expect_eq(result.stats.evaluations, 3, "surface vertex evaluations");
    expect_eq(result.stats.simplex_visits, 2, "surface simplex visits");
    expect_eq(result.stats.refinements, 0, "gapped surface refinement count");
}

void test_default_curvature_bound_is_zero() {
    auto mesh = SpectralMesh(winding_model(3));
    const auto implicit_zero = estimate_charge_on_current_mesh(
        mesh,
        0.0,
        3.0
    );
    const auto explicit_zero = estimate_charge_on_current_mesh(
        mesh,
        0.0,
        3.0,
        1,
        0.0
    );

    expect_near(
        implicit_zero.certified_error_bound,
        explicit_zero.certified_error_bound,
        kTol,
        "omitted curvature must mean zero curvature"
    );
}

void test_curvature_prevents_false_certification_of_aliased_band() {
    auto mesh = SpectralMesh(aliased_pocket());
    const auto result = estimate_charge_on_current_mesh(
        mesh,
        0.0,
        0.0,
        1,
        256.0 * std::numbers::pi_v<double> * std::numbers::pi_v<double>
    );

    expect(
        result.certified_error_bound > 0.0,
        "global tight-binding curvature must expose unresolved aliased structure"
    );
}

void test_charge_derivative_handles_repeated_vertex_energies() {
    for (const auto root_level : {0U, 1U, 2U}) {
        auto mesh = SpectralMesh(
            std::make_shared<AffineXModel>(),
            kTol,
            root_level
        );
        const auto result = estimate_charge_on_current_mesh(
            mesh,
            0.25,
            1.0
        );
        expect_near(
            result.dcharge_dmu,
            1.0,
            1e-12,
            "affine x band should have unit charge derivative"
        );
    }
}

void test_affine_matrix_charge_does_not_converge_at_root() {
    auto mesh = SpectralMesh(std::make_shared<AffineConeModel>(), kTol, 0);
    expect_runtime_error(
        [&] {
            (void)integrate_charge(
                mesh,
                0.25,
                fixed_options(1e-3)
            );
        },
        "did not converge",
        "affine matrix charge must request refinement when its eigenvalues curve"
    );
}

void test_integration_rejects_nonfinite_mu() {
    auto mesh = SpectralMesh(constant_insulator());
    const auto options = fixed_options(1.0);
    for (const auto mu : {
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
         }) {
        expect_runtime_error(
            [&] { (void)integrate_charge(mesh, mu, options); },
            "mu must be finite",
            "charge integration should reject a non-finite chemical potential"
        );
        expect_runtime_error(
            [&] { (void)estimate_charge_on_current_mesh(mesh, mu, 1.0); },
            "mu must be finite",
            "fixed-mesh charge should reject a non-finite chemical potential"
        );
        expect_runtime_error(
            [&] { (void)integrate_density_matrix(mesh, mu, {{0}}, options); },
            "mu must be finite",
            "density integration should reject a non-finite chemical potential"
        );
    }

    expect_runtime_error(
        [&] {
            (void)estimate_charge_on_current_mesh(
                mesh,
                0.0,
                std::numeric_limits<double>::quiet_NaN()
            );
        },
        "target_error",
        "fixed-mesh charge should reject a non-finite target error"
    );
    expect_runtime_error(
        [&] { (void)estimate_charge_on_current_mesh(mesh, 0.0, 1.0, 0); },
        "preview_depth",
        "fixed-mesh charge should reject a zero preview depth"
    );
}

}  // namespace

int main() {
    try {
        test_constant_insulator_charge_is_certified();
        test_fermi_surface_counts_work();
        test_default_curvature_bound_is_zero();
        test_curvature_prevents_false_certification_of_aliased_band();
        test_charge_derivative_handles_repeated_vertex_energies();
        test_affine_matrix_charge_does_not_converge_at_root();
        test_integration_rejects_nonfinite_mu();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
