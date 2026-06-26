#include "fermi_surface/fermi_surface.h"

#include "fermi_surface/adaptive_refinement.h"
#include "fermi_surface/vertex_evaluation.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace lineartetrahedron {

void reset_fermi_surface_stats() {
    fermi_surface_detail::fermi_surface_stats_ = FermiSurfaceStats{};
}

FermiSurfaceStats fermi_surface_stats() {
    return fermi_surface_detail::fermi_surface_stats_;
}

FermiSurfaceResult fermi_surface(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_states
) {
    using namespace fermi_surface_detail;

    if (!model) {
        throw std::runtime_error("fermi_surface: model must not be null");
    }
    if (model->ndim() < 1) {
        throw std::runtime_error("fermi_surface: dimension must be positive");
    }
    if (!(min_feature_size > 0.0) || !std::isfinite(min_feature_size)) {
        throw std::runtime_error("fermi_surface: min_feature_size must be positive");
    }
    if (margin < 0.0 || !std::isfinite(margin)) {
        throw std::runtime_error("fermi_surface: margin must be finite and non-negative");
    }

    return run_fermi_surface(
        std::move(model),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_states
    );
}

}  // namespace lineartetrahedron
