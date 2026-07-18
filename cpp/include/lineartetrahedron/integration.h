#pragma once

#include <lineartetrahedron/spectral_mesh.h>

#include <adaptivesimplex/adaptive/types.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

struct IntegrationStats {
    // Eigensystems newly added to the shared cache by this operation.
    std::int64_t evaluations = 0;
    std::int64_t refinements = 0;
    std::size_t cached_vertices = 0;
    std::int64_t active_simplices = 0;
    std::int64_t active_vertices = 0;
    bool target_reached = false;
};

struct ChargeResult {
    double value = 0.0;
    // Projected-residual and adaptive-preview estimate used for refinement.
    double stopping_error = 0.0;
    // Rigorous occupation-width bound derived from the supplied curvature.
    double certified_error_bound = 0.0;
    double dcharge_dmu = 0.0;
    std::int64_t visible_gapless_simplices = 0;
    std::int64_t inconclusive_simplices = 0;
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
    double curvature_bound = 0.0
);

ChargeResult estimate_charge_on_current_mesh(
    SpectralMesh &mesh,
    double mu,
    double target_error,
    std::uint32_t preview_depth = 1,
    double curvature_bound = 0.0
);

DensityMatrixResult integrate_density_matrix(
    SpectralMesh &mesh,
    double mu,
    std::vector<LatticeVector> lattice_vectors,
    const adaptivesimplex::adaptive::Options &options
);

}  // namespace lineartetrahedron
