#include "bindings.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_native, m) {
    m.doc() = "Native numerical core for lineartetrahedron";

    lineartetrahedron::bindings::bind_tight_binding(m);
    lineartetrahedron::bindings::bind_certification(m);
    lineartetrahedron::bindings::bind_integration_types(m);
    lineartetrahedron::bindings::bind_fermi_surface(m);
    lineartetrahedron::bindings::bind_spectral_mesh(m);
}
