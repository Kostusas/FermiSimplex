#pragma once

#include "bindings.h"

#include "core/tight_binding.h"

#include <cstddef>
#include <memory>

namespace lineartetrahedron::bindings {

std::shared_ptr<const HamiltonianModel> make_python_hamiltonian_model(
    nb::object callable,
    size_t ndim,
    size_t ndof
);

}  // namespace lineartetrahedron::bindings
