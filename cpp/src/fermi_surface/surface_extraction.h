#pragma once

#include <lineartetrahedron/fermi_surface.h>
#include <lineartetrahedron/spectral_mesh.h>

#include <adaptivesimplex/core/types.h>

#include <span>

namespace lineartetrahedron::fermi_surface_detail {

namespace core = adaptivesimplex::core;

void extract_terminal_surface(
    const SpectralMesh &mesh,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    FermiSurfaceResult &result
);

}  // namespace lineartetrahedron::fermi_surface_detail
