#pragma once

#include <nanobind/nanobind.h>

namespace fermisimplex::bindings {

namespace nb = nanobind;

void bind_certification(nb::module_ &m);
void bind_integration_types(nb::module_ &m);
void bind_spectral_mesh(nb::module_ &m);
void bind_fermi_surface(nb::module_ &m);

}  // namespace fermisimplex::bindings
