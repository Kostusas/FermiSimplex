#pragma once

#include <fermisimplex/spectral_mesh.h>

#include <adaptivesimplex/adaptive/types.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fermisimplex {

struct IntegrationStats {
    // Eigensystems newly added to the shared cache by this operation.
    std::int64_t evaluations = 0;
    // Total simplex-rule calls, including optional preview contributions.
    std::int64_t simplex_visits = 0;
    std::int64_t refinements = 0;
    std::size_t cached_vertices = 0;
    std::int64_t active_simplices = 0;
    std::int64_t active_vertices = 0;
    bool target_reached = false;
};

struct ChargeErrorStats {
    std::int64_t root_simplices = 0;
    std::int64_t hamiltonian_evaluations = 0;
    std::int64_t full_eigensystems = 0;
    std::int64_t reduced_eigensystems = 0;
    std::int64_t norm_eigensystems = 0;
    std::int64_t schur_evaluations = 0;
    std::int64_t schur_reductions = 0;
    std::int64_t micro_simplices = 0;
    std::int64_t terminal_simplices = 0;
    // Sampled occupation-range fallbacks, not rigorous error certificates.
    std::int64_t conservative_fallbacks = 0;
    std::int64_t schur_failures = 0;
    std::int64_t initial_active_dimension_sum = 0;
    std::int64_t terminal_active_dimension_sum = 0;
    // Zero means no successful Schur reduction was recorded.
    std::size_t minimum_active_dimension = 0;
};

struct ChargeResult {
    double value = 0.0;
    // Recursive sampled estimate of the linear-tetrahedron charge error.
    double stopping_error = 0.0;
    double dcharge_dmu = 0.0;
    std::int64_t visible_gapless_simplices = 0;
    std::int64_t inconclusive_simplices = 0;
    ChargeErrorStats error_stats;
    IntegrationStats stats;
};

struct CurrentMeshChargeResult {
    double value = 0.0;
    double dcharge_dmu = 0.0;
};

struct DensityComponent {
    std::size_t lattice_vector_index = 0;
    std::size_t row = 0;
    std::size_t column = 0;
};

struct DensityComponentsResult {
    // Values follow the order of the requested components.
    std::vector<std::complex<double>> values;
    // Adaptive quadrature estimate over the requested components.
    double stopping_error = 0.0;
    IntegrationStats stats;
};

struct DensityMatrixResult {
    // Row-major [lattice_vector][row][column] storage.
    std::vector<std::complex<double>> matrices;
    // Adaptive quadrature estimate, not a density-matrix certificate.
    double stopping_error = 0.0;
    std::size_t lattice_vector_count = 0;
    std::size_t ndof = 0;
    IntegrationStats stats;
};

ChargeResult integrate_charge(
    SpectralMesh &mesh,
    double mu,
    const adaptivesimplex::adaptive::Options &options,
    std::uint32_t error_depth = 2
);

// Evaluates missing eigensystems at existing vertices and applies the
// linear-simplex charge rule. Performs no certification or refinement.
CurrentMeshChargeResult estimate_charge_on_current_mesh(
    SpectralMesh &mesh,
    double mu
);

// Components index entries in lattice_vectors as (vector, row, column).
DensityComponentsResult integrate_density_components(
    SpectralMesh &mesh,
    double mu,
    std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components,
    const adaptivesimplex::adaptive::Options &options
);

// With preview_depth=0, integrates the current mesh without refinement.
DensityMatrixResult integrate_density_matrix(
    SpectralMesh &mesh,
    double mu,
    std::vector<LatticeVector> lattice_vectors,
    const adaptivesimplex::adaptive::Options &options
);

}  // namespace fermisimplex
