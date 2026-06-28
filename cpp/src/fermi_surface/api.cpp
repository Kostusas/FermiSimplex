#include "fermi_surface/fermi_surface.h"

#include "fermi_surface/adaptive_refinement.h"
#include "fermi_surface/vertex_evaluation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lineartetrahedron {
namespace {

double simplex_diameter(
    const fermi_surface_detail::core::Geometry &geometry,
    fermi_surface_detail::core::SimplexId simplex_id
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
    return std::sqrt(max_squared);
}

fermi_surface_detail::EnergyBoundFunction scalar_energy_bound_function(
    double hessian_bound,
    double anharmonicity_bound
) {
    return [hessian_bound, anharmonicity_bound](
               const fermi_surface_detail::core::Geometry &geometry,
               fermi_surface_detail::core::SimplexId simplex_id
           ) {
        const auto diameter = simplex_diameter(geometry, simplex_id);
        return 0.5 * hessian_bound * diameter * diameter +
               0.5 * anharmonicity_bound * diameter * diameter * diameter;
    };
}

}  // namespace

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
    double hessian_bound,
    double anharmonicity_bound,
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
    if (hessian_bound < 0.0 || !std::isfinite(hessian_bound)) {
        throw std::runtime_error("fermi_surface: hessian_bound must be finite and non-negative");
    }
    if (anharmonicity_bound < 0.0 || !std::isfinite(anharmonicity_bound)) {
        throw std::runtime_error(
            "fermi_surface: anharmonicity_bound must be finite and non-negative"
        );
    }

    return run_fermi_surface(
        std::move(model),
        mu,
        min_feature_size,
        max_diagonalizations,
        scalar_energy_bound_function(hessian_bound, anharmonicity_bound),
        tol,
        return_states
    );
}

}  // namespace lineartetrahedron
