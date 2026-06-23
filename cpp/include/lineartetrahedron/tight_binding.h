#pragma once

#include "lineartetrahedron/types.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

class HamiltonianModel {
public:
    virtual ~HamiltonianModel() = default;

    virtual size_t ndim() const noexcept = 0;
    virtual size_t ndof() const noexcept = 0;
    virtual std::vector<std::complex<double>> evaluate_reduced_point_raw(
        const double *point
    ) const = 0;
};

class TightBindingModel : public HamiltonianModel {
public:
    TightBindingModel(KeyArray keys, HoppingMatrixArray matrices);

    size_t ndim() const noexcept override { return ndim_; }
    size_t ndof() const noexcept override { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }

    nb::ndarray<nb::numpy, std::complex<double>> evaluate_point(PointArray point) const;

    std::vector<std::complex<double>> evaluate_reduced_point_raw(
        const double *point
    ) const override;
    std::vector<std::complex<double>> evaluate_point_raw(const double *point) const;

private:
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t nterms_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<std::complex<double>> matrices_;
};

}  // namespace lineartetrahedron
