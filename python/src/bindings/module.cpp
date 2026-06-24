#include "bindings.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_native, m) {
    m.doc() = "Native runtime for lineartetrahedron";

    lineartetrahedron::bindings::bind_tight_binding(m);
    lineartetrahedron::bindings::bind_integration(m);
    lineartetrahedron::bindings::bind_fermi_surface(m);
}
