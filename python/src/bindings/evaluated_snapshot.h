#pragma once

#include <fermisimplex/spectral_mesh.h>
#include <nanobind/nanobind.h>

namespace fermisimplex::bindings {
nanobind::dict evaluated_snapshot_arrays(const SpectralMesh &mesh, bool include_eigenvectors);
}  // namespace fermisimplex::bindings
