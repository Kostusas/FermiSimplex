#pragma once

#include <lineartetrahedron/fermi_surface.h>

#include <cstdint>

namespace lineartetrahedron::fermi_surface_detail {

FermiSurfaceResult run_fermi_surface(
    SpectralMesh &mesh,
    double mu,
    double min_feature_size,
    std::int64_t max_evaluations,
    double curvature_bound
);

}  // namespace lineartetrahedron::fermi_surface_detail
