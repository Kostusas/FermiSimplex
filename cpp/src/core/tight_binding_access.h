#pragma once

#include <fermisimplex/hamiltonian.h>

#include <span>

namespace fermisimplex::core_detail {

// Internal access to the immutable representation of the native model. Keep
// this out of the released Hamiltonian API: only implementation code that can
// exploit TightBindingModel exactly should see its hopping terms.
struct TightBindingModelAccess {
    static std::span<const HoppingTerm> hoppings(
        const TightBindingModel &model
    ) noexcept {
        return model.hoppings_;
    }
};

}  // namespace fermisimplex::core_detail
