#pragma once

#include "fermi_surface/fermi_surface.h"
#include "fermi_surface/vertex_evaluation.h"

#include <adaptivesimplex/core/types.h>

#include <span>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {

void extract_terminal_surface(
    const HamiltonianModel &model,
    const core::Geometry &geometry,
    const SpectraCache &cache,
    std::span<const core::SimplexId> simplex_ids,
    double mu,
    double tol,
    bool return_states,
    FermiSurfaceResult &result
);

}  // namespace lineartetrahedron::fermi_surface_detail
