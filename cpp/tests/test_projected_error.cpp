#include "integration/charge.h"
#include "integration/projected_error.h"
#include "test_helpers.h"

#include <fermisimplex/integration.h>

#include <array>
#include <exception>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {

using namespace fermisimplex;
using namespace fermisimplex::test;
namespace core = adaptivesimplex::core;

class QuadraticModel final : public HamiltonianModel {
public:
    explicit QuadraticModel(double sign) : sign_(sign) {}

    std::size_t ndim() const noexcept override { return 1; }
    std::size_t ndof() const noexcept override { return 1; }

    std::vector<Complex> evaluate(std::span<const double> point) const override {
        ++evaluation_count_;
        return {Complex{sign_ * point[0] * point[0], 0.0}};
    }

    std::size_t evaluation_count() const noexcept { return evaluation_count_; }

private:
    double sign_ = 1.0;
    mutable std::size_t evaluation_count_ = 0;
};

class EmbeddedAvoidedCrossingModel final : public HamiltonianModel {
public:
    std::size_t ndim() const noexcept override { return 1; }
    std::size_t ndof() const noexcept override { return 8; }

    std::vector<Complex> evaluate(std::span<const double> point) const override {
        constexpr auto coupling = 0.05;
        const auto z = point[0] - 0.5;
        auto result =
            std::vector<Complex>(ndof() * ndof(), Complex{0.0, 0.0});
        constexpr auto diagonal =
            std::array<double, 8>{-9.0, -6.0, -3.0, 0.0, 0.0, 3.0, 6.0, 9.0};
        for (std::size_t index = 0; index < ndof(); ++index) {
            result[index + index * ndof()] =
                Complex{diagonal[index], 0.0};
        }
        result[3 + 3 * ndof()] = Complex{z, 0.0};
        result[4 + 4 * ndof()] = Complex{-z, 0.0};
        result[3 + 4 * ndof()] = Complex{coupling, 0.0};
        result[4 + 3 * ndof()] = Complex{coupling, 0.0};
        return result;
    }
};

void fill_simplex(SpectralMesh &mesh, core::SimplexId simplex_id) {
    auto &geometry = mesh.geometry();
    auto &eigensystems = mesh.eigensystems();
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        eigensystems.insert(vertex_id, mesh.spectrum(point));
    }
}

void test_supplied_curvature_reports_a_certified_bound() {
    const auto model = std::make_shared<QuadraticModel>(1.0);
    auto mesh = SpectralMesh(model, kTol, 0);
    const auto simplex_id = first_active_simplex(mesh.geometry());
    fill_simplex(mesh, simplex_id);
    const auto vertex_evaluations = model->evaluation_count();

    const auto contribution = integration_detail::charge_on_simplex(
        0.25,
        mesh,
        mesh.geometry(),
        simplex_id,
        2.0
    );
    expect(
        contribution.certified_error_bound > 0.0,
        "quadratic simplex should require a conservative error bound"
    );
    expect(
        model->evaluation_count() > vertex_evaluations,
        "the single charge estimator should sample the projected residual"
    );
}

void test_projected_residual_signs_match_actual_minus_linear() {
    auto convex = SpectralMesh(std::make_shared<QuadraticModel>(1.0), kTol, 0);
    const auto convex_simplex = first_active_simplex(convex.geometry());
    fill_simplex(convex, convex_simplex);
    const auto negative = estimate_projected_residual(
        convex,
        convex_simplex,
        0,
        1
    );
    expect_near(negative.positive_estimate, 0.0, kTol, "convex positive residual");
    expect_near(negative.negative_estimate, 0.25, kTol, "convex negative residual");

    const auto convex_charge = integration_detail::charge_on_simplex(
        0.25,
        convex,
        convex.geometry(),
        convex_simplex,
        0.0
    );
    expect(
        convex_charge.projected_error > 0.0,
        "a negative actual-minus-linear residual must expand possible occupation"
    );

    auto concave = SpectralMesh(std::make_shared<QuadraticModel>(-1.0), kTol, 0);
    const auto concave_simplex = first_active_simplex(concave.geometry());
    fill_simplex(concave, concave_simplex);
    const auto positive = estimate_projected_residual(
        concave,
        concave_simplex,
        0,
        1
    );
    expect_near(positive.negative_estimate, 0.0, kTol, "concave negative residual");
    expect_near(positive.positive_estimate, 0.25, kTol, "concave positive residual");

    const auto concave_charge = integration_detail::charge_on_simplex(
        -0.25,
        concave,
        concave.geometry(),
        concave_simplex,
        0.0
    );
    expect(
        concave_charge.projected_error > 0.0,
        "a positive actual-minus-linear residual must reduce guaranteed occupation"
    );
}

void multiply_cached_band_by_phase(
    SpectralMesh &mesh,
    core::SimplexId simplex_id,
    std::size_t band
) {
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    for (std::size_t vertex = 0; vertex < simplex.vertex_ids.size(); ++vertex) {
        auto eigensystem = mesh.eigensystems().get(simplex.vertex_ids[vertex]);
        const auto angle = 0.37 + 0.81 * static_cast<double>(vertex);
        const auto phase = std::polar(1.0, angle);
        for (std::size_t row = 0; row < mesh.ndof(); ++row) {
            eigensystem.eigenvectors[row + band * mesh.ndof()] *= phase;
        }
        mesh.eigensystems().insert(
            simplex.vertex_ids[vertex],
            std::move(eigensystem)
        );
    }
}

void test_interpolated_projector_tracks_a_rotating_isolated_band() {
    auto mesh = SpectralMesh(
        std::make_shared<EmbeddedAvoidedCrossingModel>(),
        kTol,
        0
    );
    const auto simplex_id = first_active_simplex(mesh.geometry());
    fill_simplex(mesh, simplex_id);

    const auto estimate = estimate_projected_residual(
        mesh,
        simplex_id,
        4,
        5
    );
    expect_eq(
        estimate.union_dimension,
        3,
        "guarded avoided-crossing union should remain a partial subspace"
    );
    constexpr auto coupling = 0.05;
    const auto endpoint_energy = std::hypot(0.5, coupling);
    expect_near(
        estimate.negative_estimate,
        endpoint_energy - coupling,
        1e-12,
        "interpolated projector should recover the midpoint upper band"
    );
    expect_near(
        estimate.positive_estimate,
        endpoint_energy - coupling,
        1e-12,
        "partial-union estimate should preserve both possible shift signs"
    );

    multiply_cached_band_by_phase(mesh, simplex_id, 4);
    const auto gauged = estimate_projected_residual(
        mesh,
        simplex_id,
        4,
        5
    );
    expect_near(
        gauged.negative_estimate,
        estimate.negative_estimate,
        1e-12,
        "projector interpolation must be invariant under vertex phases"
    );
    expect_near(
        gauged.positive_estimate,
        estimate.positive_estimate,
        1e-12,
        "projector interpolation positive shift must be gauge invariant"
    );
}

}  // namespace

int main() {
    try {
        test_supplied_curvature_reports_a_certified_bound();
        test_projected_residual_signs_match_actual_minus_linear();
        test_interpolated_projector_tracks_a_rotating_isolated_band();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
