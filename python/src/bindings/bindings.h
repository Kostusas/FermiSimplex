#pragma once

#include <nanobind/nanobind.h>

namespace lineartetrahedron::bindings {

namespace nb = nanobind;

void bind_tight_binding(nb::module_ &m);
void bind_certification(nb::module_ &m);
void bind_integration_types(nb::module_ &m);
void bind_spectral_mesh(nb::module_ &m);
void bind_fermi_surface(nb::module_ &m);

}  // namespace lineartetrahedron::bindings
