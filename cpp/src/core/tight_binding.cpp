#include <fermisimplex/hamiltonian.h>

#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace fermisimplex {
namespace {

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;

bool is_finite(std::complex<double> value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

LatticeVector opposite(const LatticeVector &lattice_vector) {
    auto result = lattice_vector;
    for (auto &component : result) {
        if (component == std::numeric_limits<std::int64_t>::min()) {
            throw std::runtime_error(
                "TightBindingModel: lattice-vector components must be negatable"
            );
        }
        component = -component;
    }
    return result;
}

double hermiticity_roundoff(
    const Matrix &matrix,
    const Matrix &partner,
    std::size_t size
) {
    auto scale = 1.0;
    for (const auto *values : {&matrix, &partner}) {
        for (const auto value : *values) {
            scale = std::max({scale, std::abs(value.real()), std::abs(value.imag())});
        }
    }
    return 128.0 * std::numeric_limits<double>::epsilon() *
           static_cast<double>(std::max<std::size_t>(size, 1)) * scale;
}

void validate_partner(
    const Matrix &matrix,
    const Matrix &partner,
    std::size_t size
) {
    const auto roundoff = hermiticity_roundoff(matrix, partner, size);
    for (std::size_t column = 0; column < size; ++column) {
        for (std::size_t row = 0; row < size; ++row) {
            const auto value = matrix[column * size + row];
            const auto adjoint_partner = std::conj(partner[row * size + column]);
            if (std::abs(value - adjoint_partner) > roundoff) {
                throw std::runtime_error(
                    "TightBindingModel: H[-R] must equal the adjoint of H[R]"
                );
            }
        }
    }
}

void canonicalize_partner(Matrix &matrix, Matrix &partner, std::size_t size) {
    auto canonical = Matrix(size * size);
    for (std::size_t column = 0; column < size; ++column) {
        for (std::size_t row = 0; row < size; ++row) {
            canonical[column * size + row] = 0.5 * (
                matrix[column * size + row] +
                std::conj(partner[row * size + column])
            );
        }
    }

    matrix = canonical;
    if (&matrix == &partner) {
        return;
    }
    for (std::size_t column = 0; column < size; ++column) {
        for (std::size_t row = 0; row < size; ++row) {
            partner[column * size + row] =
                std::conj(canonical[row * size + column]);
        }
    }
}

}  // namespace

TightBindingModel::TightBindingModel(std::vector<HoppingTerm> hoppings) {
    if (hoppings.empty()) {
        throw std::runtime_error("TightBindingModel: hoppings must not be empty");
    }

    ndim_ = hoppings.front().lattice_vector.size();
    ndof_ = static_cast<std::size_t>(
        std::sqrt(static_cast<long double>(hoppings.front().matrix.size()))
    );
    if (ndim_ == 0) {
        throw std::runtime_error("TightBindingModel: dimension must be positive");
    }
    if (ndof_ == 0 || ndof_ * ndof_ != hoppings.front().matrix.size()) {
        throw std::runtime_error("TightBindingModel: matrices must be non-empty and square");
    }

    auto consolidated = std::map<LatticeVector, Matrix>{};
    for (const auto &term : hoppings) {
        if (term.lattice_vector.size() != ndim_) {
            throw std::runtime_error(
                "TightBindingModel: lattice vectors must have one dimension"
            );
        }
        if (term.matrix.size() != ndof_ * ndof_) {
            throw std::runtime_error(
                "TightBindingModel: hopping matrices must have one square shape"
            );
        }
        (void)opposite(term.lattice_vector);

        auto entry = consolidated.try_emplace(
            term.lattice_vector,
            ndof_ * ndof_,
            std::complex<double>{0.0, 0.0}
        ).first;
        for (std::size_t index = 0; index < term.matrix.size(); ++index) {
            if (!is_finite(term.matrix[index])) {
                throw std::runtime_error(
                    "TightBindingModel: hopping-matrix entries must be finite"
                );
            }
            entry->second[index] += term.matrix[index];
            if (!is_finite(entry->second[index])) {
                throw std::runtime_error(
                    "TightBindingModel: consolidated hopping entries must be finite"
                );
            }
        }
    }

    for (auto term = consolidated.begin(); term != consolidated.end();) {
        const auto is_zero = std::all_of(
            term->second.begin(),
            term->second.end(),
            [](std::complex<double> value) { return value == 0.0; }
        );
        if (is_zero) {
            term = consolidated.erase(term);
        } else {
            ++term;
        }
    }

    for (auto &[lattice_vector, matrix] : consolidated) {
        const auto opposite_vector = opposite(lattice_vector);
        const auto partner = consolidated.find(opposite_vector);
        if (partner == consolidated.end()) {
            throw std::runtime_error(
                "TightBindingModel: every lattice vector must have an opposite partner"
            );
        }
        if (opposite_vector < lattice_vector) {
            continue;
        }
        validate_partner(matrix, partner->second, ndof_);
        canonicalize_partner(matrix, partner->second, ndof_);
    }

    hoppings_.reserve(consolidated.size());
    for (auto &[lattice_vector, matrix] : consolidated) {
        hoppings_.push_back(HoppingTerm{
            .lattice_vector = lattice_vector,
            .matrix = std::move(matrix),
        });
    }
    packed_hoppings_.reserve(hoppings_.size() * ndof_ * ndof_);
    for (const auto &term : hoppings_) {
        packed_hoppings_.insert(
            packed_hoppings_.end(),
            term.matrix.begin(),
            term.matrix.end()
        );
    }
}

std::vector<std::complex<double>> TightBindingModel::evaluate(
    std::span<const double> reduced_point
) const {
    auto phases = Matrix(hoppings_.size());
    for (std::size_t term_index = 0;
         term_index < hoppings_.size();
         ++term_index) {
        const auto &term = hoppings_[term_index];
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            phase_arg +=
                2.0 * std::numbers::pi_v<double> * reduced_point[axis] *
                static_cast<double>(term.lattice_vector[axis]);
        }
        phases[term_index] = std::exp(Complex{0.0, -phase_arg});
    }
    const auto matrix_size = ndof_ * ndof_;
    auto h = Matrix(matrix_size);
    linalg::matrix_multiply(
        'N',
        'N',
        matrix_size,
        1,
        hoppings_.size(),
        Complex{1.0, 0.0},
        packed_hoppings_.data(),
        matrix_size,
        phases.data(),
        hoppings_.size(),
        Complex{0.0, 0.0},
        h.data(),
        matrix_size
    );
    return h;
}

}  // namespace fermisimplex
