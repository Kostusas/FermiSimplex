#include "fermi_surface/fermi_surface.h"

#include "internal.h"

#include <adaptivesimplex/core/root_mesh.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace lineartetrahedron {

namespace core = adaptivesimplex::core;

namespace fermi_surface_detail {

thread_local FermiSurfaceStats fermi_surface_stats_;

core::Geometry make_fermi_geometry(size_t ndim) {
    return core::root_geometry(ndim, ndim == 1 ? 2U : 1U);
}

std::vector<core::SimplexId> active_simplices(const core::Geometry &geometry) {
    const auto active = geometry.simplices().active_simplices();
    return std::vector<core::SimplexId>(active.begin(), active.end());
}

}  // namespace fermi_surface_detail

void reset_fermi_surface_stats() {
    fermi_surface_detail::fermi_surface_stats_ = FermiSurfaceStats{};
}

FermiSurfaceStats fermi_surface_stats() {
    return fermi_surface_detail::fermi_surface_stats_;
}

FermiSurfaceResult fermi_surface(
    std::shared_ptr<TightBindingModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_nearest_vertex_states
) {
    return fermi_surface_from_model(
        std::static_pointer_cast<const HamiltonianModel>(std::move(model)),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_nearest_vertex_states
    );
}

FermiSurfaceResult fermi_surface_from_model(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_nearest_vertex_states
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
        return_nearest_vertex_states
    );
}

}  // namespace lineartetrahedron
