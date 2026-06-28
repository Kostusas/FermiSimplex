#include "energy_bound.h"

#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {
namespace {

void require_finite_nonnegative(double value, const char *name) {
    if (value < 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be finite and non-negative");
    }
}

double simplex_diameter(const core::Geometry &geometry, const core::Simplex &simplex) {
    auto max_squared = 0.0;
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        const auto left_point =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[left]).to_point();
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            const auto right_point =
                geometry.vertices().dyadic_vertex(simplex.vertex_ids[right]).to_point();
            auto squared = 0.0;
            for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
                const auto delta = left_point[axis] - right_point[axis];
                squared += delta * delta;
            }
            max_squared = std::max(max_squared, squared);
        }
    }
    return std::sqrt(max_squared);
}

std::vector<double> simplex_center(const core::Geometry &geometry, const core::Simplex &simplex) {
    std::vector<double> center(geometry.ndim(), 0.0);
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
            center[axis] += point[axis];
        }
    }
    const auto scale = 1.0 / static_cast<double>(simplex.vertex_ids.size());
    for (auto &value : center) {
        value *= scale;
    }
    return center;
}

}  // namespace

EnergyBoundModel::EnergyBoundModel(nb::object hessian_bound, double anharmonicity_bound)
    : hessian_bound_(std::move(hessian_bound)),
      anharmonicity_bound_(anharmonicity_bound),
      hessian_bound_is_callable_(PyCallable_Check(hessian_bound_.ptr()) != 0) {
    require_finite_nonnegative(anharmonicity_bound_, "anharmonicity_bound");
    if (!hessian_bound_is_callable_) {
        scalar_hessian_bound_ = nb::cast<double>(hessian_bound_);
        require_finite_nonnegative(scalar_hessian_bound_, "hessian_bound");
    }
}

double EnergyBoundModel::on_simplex(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) const {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto diameter = simplex_diameter(geometry, simplex);
    auto hessian_bound = scalar_hessian_bound_;

    if (hessian_bound_is_callable_) {
        const auto center = simplex_center(geometry, simplex);
        nb::gil_scoped_acquire gil;
        nb::tuple args = nb::steal<nb::tuple>(
            PyTuple_New(static_cast<Py_ssize_t>(center.size()))
        );
        if (!args.ptr()) {
            nb::raise_python_error();
        }
        for (size_t axis = 0; axis < center.size(); ++axis) {
            PyObject *value = PyFloat_FromDouble(center[axis]);
            if (!value) {
                nb::raise_python_error();
            }
            PyTuple_SET_ITEM(args.ptr(), static_cast<Py_ssize_t>(axis), value);
        }
        hessian_bound = nb::cast<double>(hessian_bound_(*args));
        require_finite_nonnegative(hessian_bound, "hessian_bound callable return value");
    }

    return 0.5 * hessian_bound * diameter * diameter +
           0.5 * anharmonicity_bound_ * diameter * diameter * diameter;
}

}  // namespace lineartetrahedron::bindings
