#include "bindings.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_native, m) {
    m.doc() = "Native numerical core for FermiSimplex";

    fermisimplex::bindings::bind_certification(m);
    fermisimplex::bindings::bind_integration_types(m);
    fermisimplex::bindings::bind_fermi_surface(m);
    fermisimplex::bindings::bind_spectral_mesh(m);
}
