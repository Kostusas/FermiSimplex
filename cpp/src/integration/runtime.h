#pragma once

#include "integration/density.h"
#include "integration/workspace.h"

#include <adaptivesimplex/adaptive/types.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace lineartetrahedron {

class IntegrationRuntime {
public:
    IntegrationRuntime(
        std::shared_ptr<TightBindingModel> model,
        std::vector<std::int64_t> keys,
        std::vector<std::int64_t> component_rows,
        std::vector<std::int64_t> component_cols,
        std::vector<std::int64_t> component_key_indices,
        double tol = 1e-14
    );

    size_t ndim() const noexcept { return workspace_.ndim(); }
    size_t ndof() const noexcept { return workspace_.ndof(); }
    size_t density_component_count() const noexcept { return density_.size(); }
    size_t n_cached_nodes() const noexcept { return workspace_.n_cached_nodes(); }
    std::int64_t n_active_simplices() const noexcept { return workspace_.n_active_simplices(); }
    std::int64_t n_active_vertices() const { return workspace_.n_active_vertices(); }

    ChargeIntegrateResult integrate_charge(
        double mu,
        const adaptivesimplex::adaptive::Options &options
    );
    ChargeIntegrateResult evaluate_charge(
        double mu,
        const adaptivesimplex::adaptive::Options &options
    );
    DensityIntegrateResult integrate_density(
        double mu,
        const adaptivesimplex::adaptive::Options &options
    );

private:
    IntegrationWorkspace workspace_;
    DensityComponents density_;
};

}  // namespace lineartetrahedron
