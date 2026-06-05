#pragma once

#include "lineartetrahedron/types.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

class TightBindingModel {
public:
    TightBindingModel(KeyArray keys, HoppingMatrixArray matrices);

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }

    nb::ndarray<nb::numpy, std::complex<double>> evaluate_point(PointArray point) const;

    std::vector<std::complex<double>> evaluate_point_raw(const double *point) const;

private:
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t nterms_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<std::complex<double>> matrices_;
};

}  // namespace lineartetrahedron
