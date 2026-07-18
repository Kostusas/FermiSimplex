#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron {

using LatticeVector = std::vector<std::int64_t>;

class HamiltonianModel {
public:
    virtual ~HamiltonianModel() = default;

    virtual std::size_t ndim() const noexcept = 0;
    virtual std::size_t ndof() const noexcept = 0;
    // Returns a finite ndof x ndof Hermitian matrix in column-major order.
    virtual std::vector<std::complex<double>> evaluate(
        std::span<const double> reduced_point
    ) const = 0;
};

struct HoppingTerm {
    LatticeVector lattice_vector;
    // Square matrix in column-major order. All terms must have one shape.
    std::vector<std::complex<double>> matrix;
};

class TightBindingModel final : public HamiltonianModel {
public:
    // Represents H(k) = sum_R H_R exp(-2 pi i k.R).
    explicit TightBindingModel(std::vector<HoppingTerm> hoppings);

    std::size_t ndim() const noexcept override { return ndim_; }
    std::size_t ndof() const noexcept override { return ndof_; }

    std::vector<std::complex<double>> evaluate(
        std::span<const double> reduced_point
    ) const override;

private:
    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
    std::vector<HoppingTerm> hoppings_;
};

}  // namespace lineartetrahedron
