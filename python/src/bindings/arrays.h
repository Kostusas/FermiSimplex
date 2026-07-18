#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {

namespace nb = nanobind;

using PointArray = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using LatticeVectorArray =
    nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<2>, nb::c_contig>;
using HoppingMatrixArray =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<3>, nb::c_contig>;
using CallbackMatrixArray =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<2>, nb::c_contig>;
using EigenvalueArray = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;
using EigenvectorArray =
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

inline std::vector<std::vector<std::int64_t>> copy_lattice_vectors(
    LatticeVectorArray vectors
) {
    std::vector<std::vector<std::int64_t>> result(vectors.shape(0));
    for (std::size_t index = 0; index < vectors.shape(0); ++index) {
        result[index].assign(
            vectors.data() + index * vectors.shape(1),
            vectors.data() + (index + 1) * vectors.shape(1)
        );
    }
    return result;
}

}  // namespace lineartetrahedron::bindings
