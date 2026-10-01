#pragma once

#include <fermisimplex/hamiltonian.h>

namespace fermisimplex::core_detail {

// Constant-spectrum information established once on the immutable native
// model. A callable is never assumed constant from finitely many samples.
struct TightBindingModelAccess {
    static std::optional<double> constant_spectrum_roundoff(
        const HamiltonianModel &model
    ) noexcept {
        const auto *tb = dynamic_cast<const TightBindingModel *>(&model);
        return tb ? tb->constant_spectrum_roundoff_ : std::nullopt;
    }
};

}  // namespace fermisimplex::core_detail
