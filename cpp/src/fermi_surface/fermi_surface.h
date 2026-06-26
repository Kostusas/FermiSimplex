#pragma once

#include "core/tight_binding.h"
#include "core/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace lineartetrahedron {

struct FermiSurfaceStats {
    std::uint64_t evaluated_vertices = 0;
};

void reset_fermi_surface_stats();

FermiSurfaceStats fermi_surface_stats();

FermiSurfaceResult fermi_surface(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_states = false
);

std::vector<std::int64_t> product_simplex_triangulation_cells(
    size_t negative_count,
    size_t positive_count
);

}  // namespace lineartetrahedron
