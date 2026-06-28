#pragma once

#include "bindings.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>

namespace lineartetrahedron::bindings {

namespace core = adaptivesimplex::core;

class EnergyBoundModel {
public:
    EnergyBoundModel(nb::object hessian_bound, double anharmonicity_bound);

    double on_simplex(const core::Geometry &geometry, core::SimplexId simplex_id) const;

private:
    nb::object hessian_bound_;
    double scalar_hessian_bound_ = 0.0;
    double anharmonicity_bound_ = 0.0;
    bool hessian_bound_is_callable_ = false;
};

}  // namespace lineartetrahedron::bindings
