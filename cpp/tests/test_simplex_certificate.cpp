#include "certificate/bounds/occupation_bounds.h"
#include "certificate/simplex_certificate.h"
#include "core/vertex_spectra.h"
#include "test_helpers.h"

#include <adaptivesimplex/core/root_mesh.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace lineartetrahedron::test;
namespace certificate = lineartetrahedron::simplex_certificate;
namespace detail = lineartetrahedron::simplex_certificate::detail;

double simplex_diameter_squared(
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto max_squared = 0.0;
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        const auto left_point =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[left]).to_point();
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            const auto right_point =
                geometry.vertices().dyadic_vertex(simplex.vertex_ids[right]).to_point();
            auto squared = 0.0;
            for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
                const auto delta = left_point[axis] - right_point[axis];
                squared += delta * delta;
            }
            max_squared = std::max(max_squared, squared);
        }
    }
    return max_squared;
}

std::vector<lineartetrahedron::VertexSpectra> simplex_spectra(
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra> &cache
) {
    std::vector<lineartetrahedron::VertexSpectra> result;
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    result.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        result.push_back(cache.get(vertex_id));
    }
    return result;
}

certificate::SimplexCertificate certify_direct(
    const std::vector<lineartetrahedron::VertexSpectra> &spectra,
    double mu = 0.0,
    double margin = 0.0,
    double tol = certificate::kDefaultTolerance,
    bool estimate_occupation_bounds = false
) {
    std::vector<std::span<const double>> eigenvalues;
    std::vector<std::span<const Complex>> eigenvectors;
    eigenvalues.reserve(spectra.size());
    eigenvectors.reserve(spectra.size());
    for (const auto &entry : spectra) {
        eigenvalues.push_back(
            std::span<const double>(entry.eigenvalues.data(), entry.eigenvalues.size())
        );
        eigenvectors.push_back(
            std::span<const Complex>(entry.eigenvectors.data(), entry.eigenvectors.size())
        );
    }
    return certificate::certify_simplex(
        std::span<const std::span<const double>>(eigenvalues.data(), eigenvalues.size()),
        std::span<const std::span<const Complex>>(eigenvectors.data(), eigenvectors.size()),
        mu,
        margin,
        tol,
        estimate_occupation_bounds
    );
}

lineartetrahedron::VertexSpectra diagonal_spectra(std::initializer_list<double> eigenvalues) {
    const auto values = std::vector<double>(eigenvalues);
    std::vector<Complex> eigenvectors(values.size() * values.size(), Complex{0.0, 0.0});
    for (size_t index = 0; index < values.size(); ++index) {
        eigenvectors[detail::column_major_index(index, index, values.size())] = Complex{1.0, 0.0};
    }
    return lineartetrahedron::VertexSpectra{
        .eigenvalues = values,
        .eigenvectors = eigenvectors,
    };
}

void test_certified_simplex_reports_mu_bounds() {
    auto default_certificate = certificate::SimplexCertificate{};
    expect(
        !certificate::has_mu_interval(default_certificate),
        "default certificate should have an empty mu range"
    );

    auto geometry = adaptivesimplex::core::root_geometry(1, 4);
    const auto simplex_id = first_active_simplex(geometry);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(winding_model(1));
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    fill_vertex_cache(geometry, simplex_id, evaluator, cache);

    const auto spectra = simplex_spectra(geometry, simplex_id, cache);
    const auto default_result = certify_direct(spectra);
    const auto result = certificate::certify_mesh_simplex(
        geometry,
        simplex_id,
        cache,
        0.0,
        0.0,
        0.0,
        kTol,
        true
    );
    expect(
        default_result.status == certificate::SimplexCertificateStatus::CertifiedGapped,
        "default certificate arguments should certify the short winding interval"
    );
    expect(
        result.status == certificate::SimplexCertificateStatus::CertifiedGapped,
        "short winding interval should be certified gapped"
    );
    expect_eq(result.occupation_bounds.lower, 1, "certified lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 1, "certified upper occupation bound");
    expect_near(result.energy_bound, 0.0, 1e-15, "zero Hessian bound should report zero energy bound");
    expect(certificate::reusable_at(result, 0.0), "certified mu range should contain mu");
    expect(
        result.mu_interval.lower < result.mu_interval.upper,
        "constant-gap winding certificate should have a non-empty reusable mu range"
    );

    const auto direct_result = certify_direct(
        spectra,
        0.0,
        0.0,
        kTol,
        true
    );
    expect(
        direct_result.status == result.status,
        "direct and mesh certification should report the same status"
    );
    expect_eq(
        direct_result.occupation_bounds.lower,
        result.occupation_bounds.lower,
        "direct and mesh lower occupation bounds"
    );
    expect_eq(
        direct_result.occupation_bounds.upper,
        result.occupation_bounds.upper,
        "direct and mesh upper occupation bounds"
    );
}

void test_mesh_hessian_bound_matches_direct_margin() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 4);
    const auto simplex_id = first_active_simplex(geometry);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(winding_model(1));
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    fill_vertex_cache(geometry, simplex_id, evaluator, cache);

    const auto hessian_bound = 0.5;
    const auto margin = 0.5 * hessian_bound * simplex_diameter_squared(geometry, simplex_id);
    const auto spectra = simplex_spectra(geometry, simplex_id, cache);
    const auto direct = certify_direct(spectra, 0.0, margin, kTol, true);
    const auto mesh = certificate::certify_mesh_simplex(
        geometry,
        simplex_id,
        cache,
        0.0,
        hessian_bound,
        0.0,
        kTol,
        true
    );

    expect(direct.status == mesh.status, "mesh Hessian bound should match direct energy margin");
    expect_eq(direct.occupation_bounds.lower, mesh.occupation_bounds.lower, "Hessian lower bound");
    expect_eq(direct.occupation_bounds.upper, mesh.occupation_bounds.upper, "Hessian upper bound");
    expect_near(direct.energy_bound, margin, 1e-15, "direct margin should be reported");
    expect_near(mesh.energy_bound, margin, 1e-15, "mesh Hessian energy bound");
}

void test_large_mesh_hessian_bound_blocks_certification() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 4);
    const auto simplex_id = first_active_simplex(geometry);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(winding_model(1));
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    fill_vertex_cache(geometry, simplex_id, evaluator, cache);

    const auto result = certificate::certify_mesh_simplex(
        geometry,
        simplex_id,
        cache,
        0.0,
        1.0e6,
        0.0,
        kTol,
        true
    );
    expect(
        result.status != certificate::SimplexCertificateStatus::CertifiedGapped,
        "large Hessian bound should block a small-gap mesh certificate"
    );
}

void test_occupation_bound_certificate_reports_mu_bounds() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 2);
    const auto simplex_id = first_active_simplex(geometry);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(winding_model(2));
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    fill_vertex_cache(geometry, simplex_id, evaluator, cache);

    const auto result = certificate::certify_mesh_simplex(
        geometry,
        simplex_id,
        cache,
        0.0,
        0.0,
        0.0,
        kTol,
        true
    );
    expect(
        result.status == certificate::SimplexCertificateStatus::Inconclusive,
        "winding interval should be inconclusive but occupation-bounded"
    );
    expect_eq(result.occupation_bounds.lower, 0, "inconclusive lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "inconclusive upper occupation bound");
    expect(certificate::reusable_at(result, 0.0), "occupation-bound mu range should contain mu");
    expect(
        result.mu_interval.lower < result.mu_interval.upper,
        "occupation-bound certificate should report a non-empty mu range"
    );
}

void test_visible_gapless_reports_conservative_bounds() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 1);
    const auto simplex_id = first_active_simplex(geometry);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(winding_model(1));
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    fill_vertex_cache(geometry, simplex_id, evaluator, cache);

    const auto result = certificate::certify_mesh_simplex(
        geometry,
        simplex_id,
        cache,
        1.0,
        0.0,
        0.0,
        kTol,
        true
    );
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "vertex eigenvalue at mu should be visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 0, "visible gapless lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "visible gapless upper occupation bound");
}

void test_visible_gapless_estimates_occupation_bounds() {
    const auto spectra = std::vector<lineartetrahedron::VertexSpectra>{
        diagonal_spectra({-2.0, -1.0, 1.0}),
        diagonal_spectra({-2.0, 0.0, 1.0}),
        diagonal_spectra({-2.0, -0.5, 1.0}),
    };

    const auto result = certify_direct(spectra, 0.0, 0.0, kTol, true);
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "vertex eigenvalue at mu should stay visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 1, "visible estimated lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "visible estimated upper occupation bound");
}

void test_visible_occupation_mismatch_estimates_occupation_bounds() {
    const auto spectra = std::vector<lineartetrahedron::VertexSpectra>{
        diagonal_spectra({-2.0, -1.0, 1.0}),
        diagonal_spectra({-2.0, 0.5, 1.0}),
        diagonal_spectra({-2.0, -0.5, 1.0}),
    };

    const auto result = certify_direct(spectra, 0.0, 0.0, kTol, true);
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "vertex occupation mismatch should stay visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 1, "mismatch estimated lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "mismatch estimated upper occupation bound");
}

void test_visible_gapless_without_estimator_reports_conservative_bounds() {
    const auto spectra = std::vector<lineartetrahedron::VertexSpectra>{
        diagonal_spectra({-2.0, -1.0, 1.0}),
        diagonal_spectra({-2.0, 0.0, 1.0}),
        diagonal_spectra({-2.0, -0.5, 1.0}),
    };

    const auto result = certify_direct(spectra, 0.0, 0.0, kTol, false);
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "disabled estimator should not change visible status"
    );
    expect_eq(result.occupation_bounds.lower, 0, "unestimated visible lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 3, "unestimated visible upper occupation bound");
}

void test_inconclusive_without_estimator_reports_conservative_bounds() {
    auto geometry = adaptivesimplex::core::root_geometry(1, 2);
    const auto simplex_id = first_active_simplex(geometry);
    auto evaluator = lineartetrahedron::VertexSpectraEvaluator(winding_model(2));
    auto cache = adaptivesimplex::core::VertexCache<lineartetrahedron::VertexSpectra>{};
    fill_vertex_cache(geometry, simplex_id, evaluator, cache);

    const auto result = certificate::certify_mesh_simplex(
        geometry,
        simplex_id,
        cache,
        0.0,
        0.0,
        0.0,
        kTol,
        false
    );
    expect(
        result.status == certificate::SimplexCertificateStatus::Inconclusive,
        "antipodal interval should be inconclusive without estimator"
    );
    expect_eq(result.occupation_bounds.lower, 0, "unestimated lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "unestimated upper occupation bound");
}

void test_zero_width_mu_bounds_are_valid_but_not_reusable_nearby() {
    const auto estimate = detail::estimate_ordered_subset_rank_with_mu_radius(
        {
            {
                Complex{1.0, 0.0},
                Complex{1.0, 0.0},
                Complex{1.0, 0.0},
                Complex{2.0, 0.0},
            },
        },
        2,
        kTol
    );
    expect_eq(estimate.rank, 2, "positive block should certify rank two");
    expect_near(
        estimate.mu_radius,
        0.0,
        1e-14,
        "non-diagonally-dominant positive block should have zero Gershgorin mu radius"
    );

    auto result = certificate::SimplexCertificate{};
    result.status = certificate::SimplexCertificateStatus::Inconclusive;
    result.occupation_bounds = certificate::OccupationBounds{.lower = 2, .upper = 2};
    result.mu_interval = certificate::MuInterval{.lower = 0.0, .upper = 0.0};
    expect(certificate::has_mu_interval(result), "zero-width mu interval is still an interval");
    expect(certificate::reusable_at(result, 0.0), "zero-width interval should hit its endpoint");
    expect(!certificate::reusable_at(result, 1e-8), "zero-width interval should miss nearby mu");
}

void test_empty_simplex_spectra_throws() {
    auto threw = false;
    try {
        (void)certificate::certify_simplex(
            std::span<const std::span<const double>>{},
            std::span<const std::span<const Complex>>{}
        );
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect(threw, "empty simplex spectra should throw");
}

}  // namespace

int main() {
    try {
        test_certified_simplex_reports_mu_bounds();
        test_mesh_hessian_bound_matches_direct_margin();
        test_large_mesh_hessian_bound_blocks_certification();
        test_occupation_bound_certificate_reports_mu_bounds();
        test_visible_gapless_reports_conservative_bounds();
        test_visible_gapless_estimates_occupation_bounds();
        test_visible_occupation_mismatch_estimates_occupation_bounds();
        test_visible_gapless_without_estimator_reports_conservative_bounds();
        test_inconclusive_without_estimator_reports_conservative_bounds();
        test_zero_width_mu_bounds_are_valid_but_not_reusable_nearby();
        test_empty_simplex_spectra_throws();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
