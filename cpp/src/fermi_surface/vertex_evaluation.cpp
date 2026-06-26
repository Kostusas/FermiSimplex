#include "fermi_surface/vertex_evaluation.h"

#include <adaptivesimplex/core/root_mesh.h>

namespace lineartetrahedron::fermi_surface_detail {

thread_local FermiSurfaceStats fermi_surface_stats_;

core::Geometry make_fermi_geometry(size_t ndim) {
    return core::root_geometry(ndim, ndim == 1 ? 2U : 1U);
}

std::vector<core::SimplexId> active_simplices(const core::Geometry &geometry) {
    const auto active = geometry.simplices().active_simplices();
    return std::vector<core::SimplexId>(active.begin(), active.end());
}

}  // namespace lineartetrahedron::fermi_surface_detail
