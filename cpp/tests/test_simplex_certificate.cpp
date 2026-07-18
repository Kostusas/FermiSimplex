#include "certification/mesh_certificate.h"
#include "test_helpers.h"

#include <fermisimplex/certification.h>

#include <adaptivesimplex/core/root_mesh.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace fermisimplex::test;
namespace certificate = fermisimplex::certification;

std::vector<fermisimplex::Eigensystem> simplex_spectra(
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<fermisimplex::Eigensystem> &cache
) {
    std::vector<fermisimplex::Eigensystem> result;
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    result.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        result.push_back(cache.get(vertex_id));
    }
    return result;
}

void test_certified_simplex_reports_mu_bounds() {
    auto default_certificate = certificate::SimplexCertificate{};
    expect(
        !default_certificate.mu_interval.has_value(),
        "default certificate should have an empty mu range"
    );

    auto mesh = fermisimplex::SpectralMesh(winding_model(1), kTol, 4);
    const auto &geometry = mesh.geometry();
    const auto simplex_id = first_active_simplex(geometry);
    auto &cache = mesh.eigensystems();
    fill_vertex_cache(geometry, simplex_id, mesh, cache);

    const auto spectra = simplex_spectra(geometry, simplex_id, cache);
    const auto default_result = certify_direct(spectra);
    const auto result = certificate::certify_mesh_simplex(
        mesh,
        simplex_id,
        0.0,
        0.0,
        kTol
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
    expect(
        certificate::occupation_bounds_valid_at(result, 0.0),
        "certified occupation-bound range should contain mu"
    );
    expect(
        result.mu_interval.has_value() &&
            result.mu_interval->lower < result.mu_interval->upper,
        "constant-gap winding certificate should have a non-empty reusable mu range"
    );

    const auto direct_result = certify_direct(
        spectra,
        0.0,
        0.0,
        kTol
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

void test_mesh_linearization_error_bound_matches_direct_error() {
    auto mesh = fermisimplex::SpectralMesh(winding_model(1), kTol, 4);
    const auto &geometry = mesh.geometry();
    const auto simplex_id = first_active_simplex(geometry);
    auto &cache = mesh.eigensystems();
    fill_vertex_cache(geometry, simplex_id, mesh, cache);

    const auto linearization_error_bound = 0.25;
    const auto spectra = simplex_spectra(geometry, simplex_id, cache);
    const auto direct = certify_direct(spectra, 0.0, linearization_error_bound, kTol);
    const auto mesh_certificate = certificate::certify_mesh_simplex(
        mesh,
        simplex_id,
        0.0,
        linearization_error_bound,
        kTol
    );

    expect(
        direct.status == mesh_certificate.status,
        "mesh and direct linearization errors should match"
    );
    expect_eq(
        direct.occupation_bounds.lower,
        mesh_certificate.occupation_bounds.lower,
        "direct lower bound"
    );
    expect_eq(
        direct.occupation_bounds.upper,
        mesh_certificate.occupation_bounds.upper,
        "direct upper bound"
    );
}

void test_large_linearization_error_bound_blocks_certification() {
    auto mesh = fermisimplex::SpectralMesh(winding_model(1), kTol, 4);
    const auto &geometry = mesh.geometry();
    const auto simplex_id = first_active_simplex(geometry);
    auto &cache = mesh.eigensystems();
    fill_vertex_cache(geometry, simplex_id, mesh, cache);

    const auto result = certificate::certify_mesh_simplex(
        mesh,
        simplex_id,
        0.0,
        1.0e6,
        kTol
    );
    expect(
        result.status != certificate::SimplexCertificateStatus::CertifiedGapped,
        "large linearization error should block a small-gap mesh certificate"
    );
}

void test_occupation_bound_certificate_reports_mu_bounds() {
    auto mesh = fermisimplex::SpectralMesh(winding_model(2), kTol, 2);
    const auto &geometry = mesh.geometry();
    const auto simplex_id = first_active_simplex(geometry);
    auto &cache = mesh.eigensystems();
    fill_vertex_cache(geometry, simplex_id, mesh, cache);

    const auto result = certificate::certify_mesh_simplex(
        mesh,
        simplex_id,
        0.0,
        0.0,
        kTol
    );
    expect(
        result.status == certificate::SimplexCertificateStatus::Inconclusive,
        "winding interval should be inconclusive but occupation-bounded"
    );
    expect_eq(result.occupation_bounds.lower, 0, "inconclusive lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "inconclusive upper occupation bound");
    expect(
        certificate::occupation_bounds_valid_at(result, 0.0),
        "occupation-bound mu range should contain mu"
    );
    expect(
        result.mu_interval.has_value() &&
            result.mu_interval->lower < result.mu_interval->upper,
        "occupation-bound certificate should report a non-empty mu range"
    );
}

void test_visible_gapless_reports_conservative_bounds() {
    auto mesh = fermisimplex::SpectralMesh(winding_model(1), kTol, 1);
    const auto &geometry = mesh.geometry();
    const auto simplex_id = first_active_simplex(geometry);
    auto &cache = mesh.eigensystems();
    fill_vertex_cache(geometry, simplex_id, mesh, cache);

    const auto result = certificate::certify_mesh_simplex(
        mesh,
        simplex_id,
        1.0,
        0.0,
        kTol
    );
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "vertex eigenvalue at mu should be visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 0, "visible gapless lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "visible gapless upper occupation bound");
}

void test_visible_gapless_estimates_occupation_bounds() {
    const auto spectra = std::vector<fermisimplex::Eigensystem>{
        diagonal_spectra({-2.0, -1.0, 1.0}),
        diagonal_spectra({-2.0, 0.0, 1.0}),
        diagonal_spectra({-2.0, -0.5, 1.0}),
    };

    const auto result = certify_direct(spectra, 0.0, 0.0, kTol);
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "vertex eigenvalue at mu should stay visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 1, "visible estimated lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "visible estimated upper occupation bound");
}

void test_visible_occupation_mismatch_estimates_occupation_bounds() {
    const auto spectra = std::vector<fermisimplex::Eigensystem>{
        diagonal_spectra({-2.0, -1.0, 1.0}),
        diagonal_spectra({-2.0, 0.5, 1.0}),
        diagonal_spectra({-2.0, -0.5, 1.0}),
    };

    const auto result = certify_direct(spectra, 0.0, 0.0, kTol);
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "vertex occupation mismatch should stay visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 1, "mismatch estimated lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "mismatch estimated upper occupation bound");
}

void test_identical_visible_gapless_spectra_have_useful_bounds() {
    const auto spectra = std::vector<fermisimplex::Eigensystem>{
        diagonal_spectra({-2.0, 0.0, 1.0}),
        diagonal_spectra({-2.0, 0.0, 1.0}),
        diagonal_spectra({-2.0, 0.0, 1.0}),
    };

    const auto result = certify_direct(spectra);
    expect(
        result.status == certificate::SimplexCertificateStatus::VisibleGapless,
        "an anchor eigenvalue at mu should be visibly gapless"
    );
    expect_eq(result.occupation_bounds.lower, 1, "identical visible lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 2, "identical visible upper occupation bound");
}

void test_asymmetric_mu_interval_uses_the_correct_radii() {
    const auto result = certify_direct({diagonal_spectra({-2.0, 5.0})});
    expect(
        result.status == certificate::SimplexCertificateStatus::CertifiedGapped,
        "asymmetric diagonal spectrum should certify"
    );
    expect(result.mu_interval.has_value(), "certified spectrum should report a mu interval");
    expect_near(result.mu_interval->lower, -2.0, 2e-10, "occupied radius sets lower mu");
    expect_near(result.mu_interval->upper, 5.0, 2e-10, "unoccupied radius sets upper mu");
}

void test_exact_occupation_bounds_promote_an_inconclusive_gap_proof() {
    constexpr auto sine = 0.005;
    const auto cosine = std::sqrt(1.0 - sine * sine);
    const auto spectra = std::vector<fermisimplex::Eigensystem>{
        diagonal_spectra({-2.0, 2.0}),
        {
            .eigenvalues = {-1.0, 10000.0},
            .eigenvectors = {
                Complex{cosine, 0.0},
                Complex{sine, 0.0},
                Complex{-sine, 0.0},
                Complex{cosine, 0.0},
            },
        },
    };

    const auto result = certify_direct(spectra);
    expect(
        result.status == certificate::SimplexCertificateStatus::CertifiedGapped,
        "matching rigorous occupation bounds should certify the gap"
    );
    expect_eq(result.occupation_bounds.lower, 1, "promoted lower occupation bound");
    expect_eq(result.occupation_bounds.upper, 1, "promoted upper occupation bound");
}

}  // namespace

int main() {
    try {
        test_certified_simplex_reports_mu_bounds();
        test_mesh_linearization_error_bound_matches_direct_error();
        test_large_linearization_error_bound_blocks_certification();
        test_occupation_bound_certificate_reports_mu_bounds();
        test_visible_gapless_reports_conservative_bounds();
        test_visible_gapless_estimates_occupation_bounds();
        test_visible_occupation_mismatch_estimates_occupation_bounds();
        test_identical_visible_gapless_spectra_have_useful_bounds();
        test_asymmetric_mu_interval_uses_the_correct_radii();
        test_exact_occupation_bounds_promote_an_inconclusive_gap_proof();
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
