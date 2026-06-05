#pragma once

#include "lineartetrahedron/integration_state.h"
#include "lineartetrahedron/types.h"

#include <adaptivesimplex/adaptive/types.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace lineartetrahedron {

class IntegrationRuntime {
public:
    IntegrationRuntime(
        std::shared_ptr<TightBindingModel> model,
        KeyArray keys,
        ComponentIndexArray component_rows,
        ComponentIndexArray component_cols,
        ComponentIndexArray component_key_indices,
        double tol = 1e-14
    );

    size_t ndim() const noexcept { return state_.ndim(); }
    size_t ndof() const noexcept { return state_.ndof(); }
    size_t density_component_count() const noexcept { return state_.density_component_count(); }
    size_t n_cached_nodes() const noexcept { return state_.n_cached_nodes(); }
    std::int64_t n_active_simplices() const noexcept { return state_.n_active_simplices(); }
    std::int64_t n_active_vertices() const { return state_.n_active_vertices(); }

    ChargeIntegrateResult integrate_charge(
        double mu,
        double charge_atol,
        std::int64_t max_refinements
    );
    DensityIntegrateResult integrate_density(
        double mu,
        double density_atol,
        std::int64_t max_refinements
    );

private:
    adaptivesimplex::adaptive::Options options(double target, std::int64_t max_refinements) const;

    IntegrationState state_;
};

}  // namespace lineartetrahedron
