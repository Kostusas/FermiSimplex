#include "certificate/simplex_certificate.h"
#include "integration/charge.h"
#include "integration/charge_certificate_cache.h"
#include "integration/runtime.h"
#include "test_helpers.h"

#include <adaptivesimplex/adaptive/types.h>
#include <adaptivesimplex/core/root_mesh.h>

#include <exception>
#include <iostream>
#include <memory>
#include <utility>

namespace {

using namespace lineartetrahedron::test;
namespace certificate = lineartetrahedron::simplex_certificate;

void test_charge_certificate_cache_respects_mu_range() {
    auto cache = lineartetrahedron::ChargeCertificateCache{};
    auto cert = certificate::SimplexCertificate{};
    cert.status = certificate::SimplexCertificateStatus::CertifiedGapped;
    cert.mu_interval = certificate::MuInterval{.lower = -0.2, .upper = 0.3};
    cert.energy_bound = 0.0;

    cache.insert(7, cert);
    expect_eq(cache.size(), 1, "cache should store reusable certificates");
    expect(cache.find(7, 0.0, 0.0) != nullptr, "cache should hit inside the mu range");
    expect(cache.find(7, 0.3, 0.0) != nullptr, "cache should hit the upper mu endpoint");
    expect(cache.find(7, 0.31, 0.0) == nullptr, "cache should miss outside the mu range");
    expect(cache.find(8, 0.0, 0.0) == nullptr, "cache should miss different simplex ids");
    expect(
        cache.find(7, 0.0, 1.0) == nullptr,
        "cache should miss when requested energy bound is stricter than stored energy bound"
    );

    auto stricter_cert = cert;
    stricter_cert.energy_bound = 2.0;
    cache.insert(7, stricter_cert);
    expect(cache.find(7, 0.0, 1.0) != nullptr, "cache should reuse stricter energy-bound records");
    expect(cache.find(7, 0.0, 3.0) == nullptr, "cache should reject weaker energy-bound records");

    auto empty_range = cert;
    empty_range.mu_interval = certificate::MuInterval{};
    cache.insert(7, empty_range);
    expect_eq(cache.size(), 2, "cache should ignore empty mu ranges");

    cache.erase(7);
    expect_eq(cache.size(), 0, "cache erase should remove simplex certificates");
}

void test_charge_on_simplex_reuses_cached_certificate() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 4);
    const auto simplex_id = first_active_simplex(geometry);
    auto workspace = lineartetrahedron::IntegrationWorkspace(winding_model(1), kTol);
    fill_workspace_cache(geometry, simplex_id, workspace);

    auto certificate_cache = lineartetrahedron::ChargeCertificateCache{};
    (void)lineartetrahedron::charge_on_simplex(
        0.0,
        workspace,
        geometry,
        simplex_id,
        workspace.cache(),
        true,
        &certificate_cache,
        0.0
    );
    expect_eq(
        certificate_cache.size(),
        1,
        "first certified charge evaluation should cache one certificate"
    );
    const auto *cached = certificate_cache.find(simplex_id, 0.0, 0.0);
    expect(cached != nullptr, "cached certificate should contain the original mu");
    expect(
        cached->mu_interval.upper > 0.0,
        "test fixture should provide a positive reusable mu interval"
    );

    const auto inside_mu = 0.5 * cached->mu_interval.upper;
    (void)lineartetrahedron::charge_on_simplex(
        inside_mu,
        workspace,
        geometry,
        simplex_id,
        workspace.cache(),
        true,
        &certificate_cache,
        0.0
    );
    expect_eq(
        certificate_cache.size(),
        1,
        "charge evaluation inside cached mu range should reuse the certificate"
    );
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
        "inconclusive charge error should use the certified [0,2] occupation-width error"
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

    const auto strict_hessian = visible_runtime.evaluate_charge(0.0, options, true, 1.0e6);
    expect_near(
        strict_hessian.charge_error,
        visible_default.charge_error,
        1e-12,
        "Hessian input should not contribute charge certificate error"
    );
}

void test_projected_error_detects_nonlinear_visible_cut() {
    std::vector<std::int64_t> keys{1, -1};
    std::vector<Complex> matrices{Complex{0.5, 0.0}, Complex{0.5, 0.0}};
    auto runtime = lineartetrahedron::IntegrationRuntime(
        std::make_shared<lineartetrahedron::TightBindingModel>(
            1,
            1,
            std::move(keys),
            std::move(matrices)
        ),
        {0},
        {},
        {},
        {},
        kTol
    );
    const auto options = adaptivesimplex::adaptive::Options{
        .target_error = 1.0,
        .max_refinements = 0,
        .preview_depth = 1,
        .min_refinement_batch_size = 1,
        .max_refinement_batch_size = 100,
    };

    const auto result = runtime.evaluate_charge(0.0, options);
    expect(
        result.charge_error > 0.0,
        "projected residual shell should add error for nonlinear visible cuts"
    );
}

}  // namespace

int main() {
    try {
        test_charge_certificate_cache_respects_mu_range();
        test_charge_on_simplex_reuses_cached_certificate();
        test_charge_path_uses_occupation_bounds();
        test_projected_error_detects_nonlinear_visible_cut();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
