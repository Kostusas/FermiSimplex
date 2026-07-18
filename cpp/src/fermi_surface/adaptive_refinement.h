#pragma once

#include <fermisimplex/fermi_surface.h>

#include <cstdint>

namespace fermisimplex::fermi_surface_detail {

FermiSurfaceResult run_fermi_surface(
    SpectralMesh &mesh,
    double mu,
    double min_feature_size,
    std::int64_t max_evaluations,
    double curvature_bound
);

}  // namespace fermisimplex::fermi_surface_detail
