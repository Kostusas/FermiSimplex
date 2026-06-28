#pragma once

#include "fermi_surface/fermi_surface.h"
#include "fermi_surface/simplex_classification.h"

#include <cstdint>
#include <memory>

namespace lineartetrahedron::fermi_surface_detail {

FermiSurfaceResult run_fermi_surface(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    EnergyBoundFunction energy_bound,
    double tol,
    bool return_states
);

}  // namespace lineartetrahedron::fermi_surface_detail
