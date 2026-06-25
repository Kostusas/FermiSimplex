#include "certificate/internal.h"
#include "certificate/simplex_certificate.h"
#include "core/tight_binding.h"
#include "core/vertex_spectra.h"
#include "integration/charge.h"
#include "integration/runtime.h"

#include <adaptivesimplex/adaptive/types.h>
#include <adaptivesimplex/core/root_mesh.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;
namespace detail = lineartetrahedron::simplex_certificate::detail;

constexpr double kTol = 1e-12;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_eq(size_t actual, size_t expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void expect_near(double actual, double expected, double tol, const std::string &message) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

std::vector<Complex> diagonal_matrix(std::initializer_list<double> diagonal) {
    const auto size = diagonal.size();
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    auto index = size_t{0};
    for (const auto value : diagonal) {
        result[detail::column_major_index(index, index, size)] = Complex{value, 0.0};
        ++index;
    }
    return result;
}

std::vector<Complex> hermitian_2x2(double a, double b, double c) {
    return {
        Complex{a, 0.0},
        Complex{b, 0.0},
        Complex{b, 0.0},
        Complex{c, 0.0},
    };
}

std::vector<Complex> winding_hamiltonian(int winding, double reduced_k) {
    const auto phase = 2.0 * lineartetrahedron::kPi * static_cast<double>(winding) * reduced_k;
    const auto c = std::cos(phase);
    const auto s = std::sin(phase);
    return hermitian_2x2(c, s, -c);
}

std::pair<size_t, size_t> winding_bounds(int winding, std::initializer_list<double> points) {
    std::vector<std::vector<Complex>> plus_blocks;
    std::vector<std::vector<Complex>> minus_blocks;
    for (const auto point : points) {
        auto h = winding_hamiltonian(winding, point);
        plus_blocks.push_back(diagonal_matrix({
            std::real(h[detail::column_major_index(0, 0, 2)]),
        }));
        minus_blocks.push_back(diagonal_matrix({
            -std::real(h[detail::column_major_index(1, 1, 2)]),
        }));
    }
    const auto r_minus = detail::estimate_common_rank(minus_blocks, 1, kTol);
    const auto r_plus = detail::estimate_common_rank(plus_blocks, 1, kTol);
    return {r_minus, 2 - r_plus};
}

std::shared_ptr<lineartetrahedron::TightBindingModel> winding_model(int winding) {
    std::vector<std::int64_t> keys{
        static_cast<std::int64_t>(winding),
        static_cast<std::int64_t>(-winding),
    };
    const auto sigma_z = diagonal_matrix({1.0, -1.0});
    const auto sigma_x = hermitian_2x2(0.0, 1.0, 0.0);

    std::vector<Complex> matrices;
    matrices.reserve(2 * 4);
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        matrices.push_back(0.5 * sigma_z[index] + Complex{0.0, 0.5} * sigma_x[index]);
    }
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        matrices.push_back(0.5 * sigma_z[index] - Complex{0.0, 0.5} * sigma_x[index]);
    }
    return std::make_shared<lineartetrahedron::TightBindingModel>(
        1,
        2,
        std::move(keys),
        std::move(matrices)
    );
}

void test_explicit_common_rank_cases() {
    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({2.0, 3.0}),
                diagonal_matrix({1.5, 2.5}),
                diagonal_matrix({4.0, 1.2}),
            },
            2,
            kTol
        ),
        2,
        "all-positive blocks should certify full rank"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({2.0, -1.0}),
                diagonal_matrix({1.5, -0.5}),
            },
            2,
            kTol
        ),
        1,
        "single common positive direction should certify rank one"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({1.0, -1.0}),
                diagonal_matrix({-1.0, 1.0}),
            },
            2,
            kTol
        ),
        0,
        "antipodal positive cones should certify no common direction"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({3.0, 2.0, -1.0}),
                diagonal_matrix({2.0, 3.0, -0.5}),
                diagonal_matrix({4.0, 1.5, -2.0}),
            },
            3,
            kTol
        ),
        2,
        "sorted-prefix test should return the largest passing prefix rank"
    );

    expect_eq(
        detail::estimate_common_rank(
            {
                diagonal_matrix({1.0, 3.0, 2.0}),
                diagonal_matrix({1.0, -1.0, 2.0}),
            },
            3,
            kTol
        ),
        2,
        "sorted-prefix test should sort directions by worst vertex margin"
    );

    const auto full_rank_estimate = detail::estimate_common_rank_with_mu_radius(
        {
            diagonal_matrix({2.0, 3.0}),
            diagonal_matrix({1.5, 2.5}),
        },
        2,
        kTol
    );
    expect_eq(full_rank_estimate.rank, 2, "rank estimate should report full rank");
    expect(
        full_rank_estimate.mu_radius > 1.0 && full_rank_estimate.mu_radius < 1.6,
        "rank estimate should report a finite one-sided mu radius"
    );

    const auto zero_rank_estimate = detail::estimate_common_rank_with_mu_radius(
        {
            diagonal_matrix({1.0, -1.0}),
            diagonal_matrix({-1.0, 1.0}),
        },
        2,
        kTol
    );
    expect_eq(zero_rank_estimate.rank, 0, "zero-rank estimate should report zero rank");
    expect(
        std::isinf(zero_rank_estimate.mu_radius),
        "zero-rank estimate should impose no one-sided mu constraint"
    );
}

void test_winding_model_bounds() {
    const auto short_bounds = winding_bounds(2, {0.0, 0.0625});
    expect_eq(short_bounds.first, 1, "short winding interval lower bound");
    expect_eq(short_bounds.second, 1, "short winding interval upper bound");

    const auto antipodal_bounds = winding_bounds(2, {0.0, 0.25});
    expect_eq(antipodal_bounds.first, 0, "antipodal winding interval lower bound");
    expect_eq(antipodal_bounds.second, 2, "antipodal winding interval upper bound");
}

void test_certified_simplex_reports_mu_bounds() {
    auto default_certificate = lineartetrahedron::simplex_certificate::SimplexCertificate{};
    expect(
        default_certificate.lower_mu_bound > default_certificate.upper_mu_bound,
        "default certificate should have an empty mu range"
    );

    auto geometry = adaptivesimplex::core::root_geometry(1, 4);
    const auto active = geometry.simplices().active_simplices();
    const auto simplex_id = *active.begin();
    const auto model = winding_model(1);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(model);
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    for (const auto vertex_id : geometry.simplices().simplex(simplex_id).vertex_ids) {
        cache.insert(vertex_id, evaluator.evaluate(geometry, vertex_id));
    }

    const auto certificate = lineartetrahedron::simplex_certificate::certify_simplex_gap(
        0.0,
        geometry,
        simplex_id,
        cache,
        0.0,
        kTol,
        true
    );
    expect(
        certificate.status ==
            lineartetrahedron::simplex_certificate::SimplexCertificateStatus::CertifiedGapped,
        "short winding interval should be certified gapped"
    );
    expect(
        certificate.lower_mu_bound <= 0.0 && certificate.upper_mu_bound >= 0.0,
        "certified mu range should contain the certified chemical potential"
    );
    expect(
        certificate.lower_mu_bound < certificate.upper_mu_bound,
        "constant-gap winding certificate should have a non-empty reusable mu range"
    );
}

void test_occupation_bound_certificate_reports_mu_bounds() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 2);
    const auto active = geometry.simplices().active_simplices();
    const auto simplex_id = *active.begin();
    const auto model = winding_model(2);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(model);
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    for (const auto vertex_id : geometry.simplices().simplex(simplex_id).vertex_ids) {
        cache.insert(vertex_id, evaluator.evaluate(geometry, vertex_id));
    }

    const auto certificate = lineartetrahedron::simplex_certificate::certify_simplex_gap(
        0.0,
        geometry,
        simplex_id,
        cache,
        0.0,
        kTol,
        true
    );
    expect(
        certificate.status ==
            lineartetrahedron::simplex_certificate::SimplexCertificateStatus::Inconclusive,
        "winding interval should be inconclusive but occupation-bounded"
    );
    expect(certificate.has_occupation_bounds, "inconclusive certificate should report bounds");
    expect_eq(certificate.lower_occupation_bound, 0, "inconclusive lower occupation bound");
    expect_eq(certificate.upper_occupation_bound, 2, "inconclusive upper occupation bound");
    expect(
        certificate.lower_mu_bound <= 0.0 && certificate.upper_mu_bound >= 0.0,
        "occupation-bound mu range should contain the certified chemical potential"
    );
    expect(
        certificate.lower_mu_bound < certificate.upper_mu_bound,
        "occupation-bound certificate should report a non-empty mu range"
    );
}

void test_charge_certificate_cache_respects_mu_range() {
    auto cache = lineartetrahedron::ChargeCertificateCache{};
    auto certificate = lineartetrahedron::simplex_certificate::SimplexCertificate{};
    certificate.status =
        lineartetrahedron::simplex_certificate::SimplexCertificateStatus::CertifiedGapped;
    certificate.lower_mu_bound = -0.2;
    certificate.upper_mu_bound = 0.3;

    cache.insert(7, certificate);
    expect_eq(cache.size(), 1, "cache should store reusable certificates");
    expect(cache.find(7, 0.0) != nullptr, "cache should hit inside the mu range");
    expect(cache.find(7, 0.3) != nullptr, "cache should hit the upper mu endpoint");
    expect(cache.find(7, 0.31) == nullptr, "cache should miss outside the mu range");
    expect(cache.find(8, 0.0) == nullptr, "cache should miss different simplex ids");

    auto empty_range = certificate;
    empty_range.lower_mu_bound = 1.0;
    empty_range.upper_mu_bound = -1.0;
    cache.insert(7, empty_range);
    expect_eq(cache.size(), 1, "cache should ignore empty mu ranges");

    cache.erase(7);
    expect_eq(cache.size(), 0, "cache erase should remove simplex certificates");
}

void test_charge_on_simplex_reuses_cached_certificate() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 4);
    const auto active = geometry.simplices().active_simplices();
    const auto simplex_id = *active.begin();
    auto workspace = lineartetrahedron::IntegrationWorkspace(winding_model(1), kTol);
    auto &cache = workspace.cache();
    for (const auto vertex_id : geometry.simplices().simplex(simplex_id).vertex_ids) {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        cache.insert(
            vertex_id,
            workspace.evaluate_vertex(std::span<const double>(point.data(), point.size()))
        );
    }

    auto certificate_cache = lineartetrahedron::ChargeCertificateCache{};
    (void)lineartetrahedron::charge_on_simplex(
        0.0,
        workspace,
        geometry,
        simplex_id,
        cache,
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
        cached->upper_mu_bound > 0.0,
        "test fixture should provide a positive reusable mu interval"
    );

    const auto inside_mu = 0.5 * cached->upper_mu_bound;
    (void)lineartetrahedron::charge_on_simplex(
        inside_mu,
        workspace,
        geometry,
        simplex_id,
        cache,
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
        test_explicit_common_rank_cases();
        test_winding_model_bounds();
        test_certified_simplex_reports_mu_bounds();
        test_occupation_bound_certificate_reports_mu_bounds();
        test_charge_certificate_cache_respects_mu_range();
        test_charge_on_simplex_reuses_cached_certificate();
        test_charge_path_uses_occupation_bounds();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
