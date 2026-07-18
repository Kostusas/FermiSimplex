#include "integration/charge.h"
#include "integration/projected_error.h"
#include "test_helpers.h"

#include <lineartetrahedron/integration.h>

#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {

using namespace lineartetrahedron;
using namespace lineartetrahedron::test;
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

}  // namespace

int main() {
    try {
        test_supplied_curvature_reports_a_certified_bound();
        test_projected_residual_signs_match_actual_minus_linear();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
