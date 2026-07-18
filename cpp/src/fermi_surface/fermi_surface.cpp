#include <lineartetrahedron/fermi_surface.h>

#include <lineartetrahedron/spectral_mesh.h>

#include "fermi_surface/adaptive_refinement.h"

#include <cmath>
#include <stdexcept>

namespace lineartetrahedron {

FermiSurfaceResult fermi_surface(
    SpectralMesh &mesh,
    double mu,
    double min_feature_size,
    std::int64_t max_evaluations,
    double curvature_bound
) {
    if (!std::isfinite(mu)) {
        throw std::runtime_error("fermi_surface: mu must be finite");
    }
    if (!(min_feature_size > 0.0) || !std::isfinite(min_feature_size)) {
        throw std::runtime_error("fermi_surface: min_feature_size must be positive");
    }
    if (max_evaluations < -1) {
        throw std::runtime_error("fermi_surface: max_evaluations must be -1 or non-negative");
    }
    if (!std::isfinite(curvature_bound) || curvature_bound < 0.0) {
        throw std::runtime_error(
            "fermi_surface: curvature_bound must be finite and non-negative"
        );
    }

    return fermi_surface_detail::run_fermi_surface(
        mesh,
        mu,
        min_feature_size,
        max_evaluations,
        curvature_bound
    );
}

}  // namespace lineartetrahedron
