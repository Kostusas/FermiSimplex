#pragma once

#include <fermisimplex/fermi_surface.h>
#include <fermisimplex/spectral_mesh.h>

#include <adaptivesimplex/core/types.h>

#include <span>

namespace fermisimplex::fermi_surface_detail {

namespace core = adaptivesimplex::core;

void extract_terminal_surface(
    const SpectralMesh &mesh,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    FermiSurfaceResult &result
);

}  // namespace fermisimplex::fermi_surface_detail
