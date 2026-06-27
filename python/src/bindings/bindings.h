#pragma once

#include <nanobind/nanobind.h>

namespace lineartetrahedron::bindings {

namespace nb = nanobind;

void bind_tight_binding(nb::module_ &m);
void bind_certificate(nb::module_ &m);
void bind_integration(nb::module_ &m);
void bind_fermi_surface(nb::module_ &m);

}  // namespace lineartetrahedron::bindings
