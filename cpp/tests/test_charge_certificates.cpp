#include "certificate/simplex_certificate.h"
#include "integration/charge.h"
#include "integration/charge_certificate_cache.h"
#include "integration/runtime.h"
#include "test_helpers.h"

#include <adaptivesimplex/adaptive/types.h>
#include <adaptivesimplex/core/root_mesh.h>

#include <exception>
#include <iostream>

namespace {

using namespace lineartetrahedron::test;
namespace certificate = lineartetrahedron::simplex_certificate;

void test_charge_certificate_cache_respects_mu_range() {
    auto cache = lineartetrahedron::ChargeCertificateCache{};
    auto cert = certificate::SimplexCertificate{};
    cert.status = certificate::SimplexCertificateStatus::CertifiedGapped;
    cert.mu_interval = certificate::MuInterval{.lower = -0.2, .upper = 0.3};

    cache.insert(7, cert);
    expect_eq(cache.size(), 1, "cache should store reusable certificates");
    expect(cache.find(7, 0.0) != nullptr, "cache should hit inside the mu range");
    expect(cache.find(7, 0.3) != nullptr, "cache should hit the upper mu endpoint");
    expect(cache.find(7, 0.31) == nullptr, "cache should miss outside the mu range");
    expect(cache.find(8, 0.0) == nullptr, "cache should miss different simplex ids");

    auto empty_range = cert;
    empty_range.mu_interval = certificate::MuInterval{};
    cache.insert(7, empty_range);
    expect_eq(cache.size(), 1, "cache should ignore empty mu ranges");

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
        &certificate_cache
    );
    expect_eq(
        certificate_cache.size(),
        1,
        "first certified charge evaluation should cache one certificate"
    );
    const auto *cached = certificate_cache.find(simplex_id, 0.0);
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
        &certificate_cache
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
        "charge path should use the certified [0,2] occupation-width error"
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
}

}  // namespace

int main() {
    try {
        test_charge_certificate_cache_respects_mu_range();
        test_charge_on_simplex_reuses_cached_certificate();
        test_charge_path_uses_occupation_bounds();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
