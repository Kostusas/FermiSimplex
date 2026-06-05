#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace lineartetrahedron {

constexpr double kPi = 3.141592653589793238462643383279502884;

using PointArray = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using ComponentIndexArray =
    nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>, nb::c_contig>;
using KeyArray = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<2>, nb::c_contig>;
using HoppingMatrixArray =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<3>, nb::c_contig>;

template <typename T>
nb::ndarray<nb::numpy, T> make_array(
    std::vector<T> &&data,
    std::initializer_list<size_t> shape
) {
    T *raw = new T[data.size()];
    std::move(data.begin(), data.end(), raw);
    nb::capsule owner(raw, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return nb::ndarray<nb::numpy, T>(raw, shape, owner);
}

struct ChargeValue {
    double charge = 0.0;
    double derivative = 0.0;

    ChargeValue &operator+=(const ChargeValue &other) noexcept {
        charge += other.charge;
        derivative += other.derivative;
        return *this;
    }

    ChargeValue &operator-=(const ChargeValue &other) noexcept {
        charge -= other.charge;
        derivative -= other.derivative;
        return *this;
    }
};

struct ChargeIntegrateResult {
    double charge = 0.0;
    double charge_error = 0.0;
    double dcharge_dmu = 0.0;
    std::int64_t work = 0;
    std::int64_t refinements = 0;
    std::int64_t n_active_simplices = 0;
    std::int64_t n_active_vertices = 0;
    bool converged = false;
    bool error_estimate_available = true;
};

struct DensityIntegrateResult {
    std::vector<std::complex<double>> estimate;
    std::vector<double> error_vector;
    double error_scalar = 0.0;
    std::int64_t work = 0;
    std::int64_t refinements = 0;
    std::int64_t n_active_simplices = 0;
    std::int64_t n_active_vertices = 0;
    bool converged = false;
    bool error_estimate_available = true;
};

}  // namespace lineartetrahedron
