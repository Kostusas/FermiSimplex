#pragma once

#include "core/types.h"

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
    TightBindingModel(
        size_t ndim,
        size_t ndof,
        std::vector<std::int64_t> keys,
        std::vector<std::complex<double>> matrices
    );

    size_t ndim() const noexcept override { return ndim_; }
    size_t ndof() const noexcept override { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }

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
