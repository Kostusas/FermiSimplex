#include "integration/charge_error/projected_schur.h"

#include "core/tight_binding_access.h"
#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace fermisimplex::integration_detail {
namespace {

using Complex = std::complex<double>;
using Matrix = SchurMatrix;

std::size_t matrix_index(
    std::size_t row,
    std::size_t column,
    std::size_t rows
) {
    return row + column * rows;
}

bool finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void make_hermitian(Matrix &matrix, std::size_t size) {
    for (std::size_t column = 0; column < size; ++column) {
        matrix[matrix_index(column, column, size)] =
            Complex{matrix[matrix_index(column, column, size)].real(), 0.0};
        for (std::size_t row = column + 1; row < size; ++row) {
            const auto lower = matrix[matrix_index(row, column, size)];
            const auto upper = matrix[matrix_index(column, row, size)];
            const auto value = 0.5 * (lower + std::conj(upper));
            matrix[matrix_index(row, column, size)] = value;
            matrix[matrix_index(column, row, size)] = std::conj(value);
        }
    }
}

Matrix multiply(
    char left_operation,
    char right_operation,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner_dimension,
    std::span<const Complex> left,
    std::size_t left_leading_dimension,
    std::span<const Complex> right,
    std::size_t right_leading_dimension
) {
    auto result = Matrix(rows * columns, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        left_operation,
        right_operation,
        rows,
        columns,
        inner_dimension,
        Complex{1.0, 0.0},
        left.data(),
        left_leading_dimension,
        right.data(),
        right_leading_dimension,
        Complex{0.0, 0.0},
        result.data(),
        rows
    );
    return result;
}

Matrix selected_columns(
    SchurEigensystemView eigensystem,
    std::span<const std::size_t> columns
) {
    const auto size = eigensystem.eigenvalues.size();
    auto result = Matrix(size * columns.size());
    for (std::size_t output = 0; output < columns.size(); ++output) {
        for (std::size_t row = 0; row < size; ++row) {
            result[matrix_index(row, output, size)] =
                eigensystem.eigenvectors[
                    matrix_index(row, columns[output], size)
                ];
        }
    }
    return result;
}

Complex conjugate_dot(
    std::span<const Complex> left,
    std::span<const Complex> right
) {
    auto result = Complex{0.0, 0.0};
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += std::conj(left[index]) * right[index];
    }
    return result;
}

bool checked_add(
    std::int64_t left,
    std::int64_t right,
    std::int64_t &result
) {
    if (
        (right > 0 &&
         left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 &&
         left < std::numeric_limits<std::int64_t>::min() - right)
    ) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_subtract(
    std::int64_t left,
    std::int64_t right,
    std::int64_t &result
) {
    if (
        (right > 0 &&
         left < std::numeric_limits<std::int64_t>::min() + right) ||
        (right < 0 &&
         left > std::numeric_limits<std::int64_t>::max() + right)
    ) {
        return false;
    }
    result = left - right;
    return true;
}

std::optional<LatticeVector> combine_lattice_vectors(
    std::span<const std::int64_t> first,
    std::span<const std::int64_t> second,
    std::span<const std::int64_t> subtract
) {
    auto result = LatticeVector(first.size());
    for (std::size_t axis = 0; axis < first.size(); ++axis) {
        auto sum = std::int64_t{0};
        if (
            !checked_add(first[axis], second[axis], sum) ||
            !checked_subtract(sum, subtract[axis], result[axis])
        ) {
            return std::nullopt;
        }
    }
    return result;
}

}  // namespace

SchurLayer make_schur_layer(
    SchurEigensystemView anchor,
    std::size_t active_begin,
    std::size_t active_end,
    bool materialize_resolvent
) {
    const auto size = anchor.eigenvalues.size();
    if (
        active_begin >= active_end || active_end > size ||
        anchor.eigenvectors.size() != size * size
    ) {
        throw SchurFailure{};
    }

    auto active_columns = std::vector<std::size_t>{};
    auto safe_columns = std::vector<std::size_t>{};
    auto inverse_safe_eigenvalues = std::vector<double>{};
    const auto active = active_end - active_begin;
    active_columns.reserve(active);
    safe_columns.reserve(size - active);
    inverse_safe_eigenvalues.reserve(size - active);
    for (std::size_t band = 0; band < size; ++band) {
        if (active_begin <= band && band < active_end) {
            active_columns.push_back(band);
        } else {
            const auto value = anchor.eigenvalues[band];
            if (!std::isfinite(value) || value == 0.0) {
                throw SchurFailure{};
            }
            safe_columns.push_back(band);
            inverse_safe_eigenvalues.push_back(1.0 / value);
        }
    }
    auto active_basis = selected_columns(anchor, active_columns);
    const auto safe_basis = selected_columns(anchor, safe_columns);
    auto safe_resolvent = Matrix{};
    if (materialize_resolvent) {
        auto scaled_safe_basis = safe_basis;
        for (std::size_t column = 0; column < safe_columns.size(); ++column) {
            for (std::size_t row = 0; row < size; ++row) {
                scaled_safe_basis[matrix_index(row, column, size)] *=
                    inverse_safe_eigenvalues[column];
            }
        }
        safe_resolvent = multiply(
            'N', 'C', size, size, safe_columns.size(),
            scaled_safe_basis, size, safe_basis, size
        );
        make_hermitian(safe_resolvent, size);
    }
    return SchurLayer{
        .parent_dimension = size,
        .active_dimension = active_columns.size(),
        .active_basis = std::move(active_basis),
        .safe_resolvent = std::move(safe_resolvent),
        .safe_basis = materialize_resolvent
            ? Matrix{}
            : std::move(safe_basis),
        .inverse_safe_eigenvalues = materialize_resolvent
            ? std::vector<double>{}
            : std::move(inverse_safe_eigenvalues),
        .x_buffer = Matrix(size * active),
        .y_buffer = Matrix(size * active),
    };
}

SchurMatrix apply_schur_layer(
    std::span<const Complex> matrix,
    const SchurLayer &layer,
    ChargeErrorStats &stats
) {
    const auto size = layer.parent_dimension;
    const auto active = layer.active_dimension;
    if (
        active == 0 || matrix.size() != size * size ||
        layer.active_basis.size() != size * active ||
        layer.safe_resolvent.size() != size * size ||
        layer.x_buffer.size() != size * active ||
        layer.y_buffer.size() != size * active
    ) {
        throw SchurFailure{};
    }

    auto &x = layer.x_buffer;
    auto &y = layer.y_buffer;
    if (active == 1) {
        linalg::matrix_multiply(
            'N', 'N', size, 1, size, Complex{1.0, 0.0},
            matrix.data(), size, layer.active_basis.data(), size,
            Complex{0.0, 0.0}, x.data(), size
        );
        auto value = conjugate_dot(layer.active_basis, x);
        linalg::matrix_multiply(
            'N', 'N', size, 1, size, Complex{1.0, 0.0},
            layer.safe_resolvent.data(), size, x.data(), size,
            Complex{0.0, 0.0}, y.data(), size
        );
        const auto frozen_shift = conjugate_dot(x, y);
        linalg::matrix_multiply(
            'N', 'N', size, 1, size, Complex{1.0, 0.0},
            matrix.data(), size, y.data(), size,
            Complex{0.0, 0.0}, x.data(), size
        );
        value += conjugate_dot(y, x) -
            Complex{2.0, 0.0} * frozen_shift;
        ++stats.schur_evaluations;
        if (!finite(value)) {
            throw SchurFailure{};
        }
        return Matrix{Complex{value.real(), 0.0}};
    }
    auto result = Matrix(active * active);
    auto frozen_shift = Matrix(active * active);

    // X = H U, A = U^H X.
    linalg::matrix_multiply(
        'N', 'N', size, active, size, Complex{1.0, 0.0},
        matrix.data(), size, layer.active_basis.data(), size,
        Complex{0.0, 0.0}, x.data(), size
    );
    linalg::matrix_multiply(
        'C', 'N', active, active, size, Complex{1.0, 0.0},
        layer.active_basis.data(), size, x.data(), size,
        Complex{0.0, 0.0}, result.data(), active
    );

    // Y = R X and F = X^H Y, where R = Us D0^-1 Us^H.
    linalg::matrix_multiply(
        'N', 'N', size, active, size, Complex{1.0, 0.0},
        layer.safe_resolvent.data(), size, x.data(), size,
        Complex{0.0, 0.0}, y.data(), size
    );
    linalg::matrix_multiply(
        'C', 'N', active, active, size, Complex{1.0, 0.0},
        x.data(), size, y.data(), size,
        Complex{0.0, 0.0}, frozen_shift.data(), active
    );

    // Reuse X for H Y and accumulate Y^H H Y into A.
    linalg::matrix_multiply(
        'N', 'N', size, active, size, Complex{1.0, 0.0},
        matrix.data(), size, y.data(), size,
        Complex{0.0, 0.0}, x.data(), size
    );
    linalg::matrix_multiply(
        'C', 'N', active, active, size, Complex{1.0, 0.0},
        y.data(), size, x.data(), size,
        Complex{1.0, 0.0}, result.data(), active
    );

    ++stats.schur_evaluations;
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] -= Complex{2.0, 0.0} * frozen_shift[index];
        if (!finite(result[index])) {
            throw SchurFailure{};
        }
    }
    make_hermitian(result, active);
    return result;
}

SchurMatrix ProjectedHoppingModel::evaluate(
    std::span<const double> point
) const {
    if (point.size() != ndim) {
        throw SchurFailure{};
    }
    const auto matrix_size = dimension * dimension;
    auto result = Matrix(matrix_size, Complex{0.0, 0.0});
    for (std::size_t term = 0; term < lattice_vectors.size(); ++term) {
        auto phase_argument = 0.0;
        for (std::size_t axis = 0; axis < ndim; ++axis) {
            phase_argument +=
                2.0 * std::numbers::pi_v<double> * point[axis] *
                static_cast<double>(lattice_vectors[term][axis]);
        }
        const auto phase = std::exp(Complex{0.0, -phase_argument});
        const auto offset = term * matrix_size;
        for (std::size_t index = 0; index < matrix_size; ++index) {
            result[index] += phase * coefficients[offset + index];
        }
    }
    for (const auto value : result) {
        if (!finite(value)) {
            throw SchurFailure{};
        }
    }
    make_hermitian(result, dimension);
    return result;
}

std::optional<ProjectedHoppingModel> make_projected_hopping_model(
    const TightBindingModel &model,
    double mu,
    const SchurLayer &layer
) {
    if (
        layer.active_dimension == 0 ||
        layer.parent_dimension != model.ndof()
    ) {
        return std::nullopt;
    }

    struct ShiftedHopping {
        const LatticeVector *lattice_vector = nullptr;
        std::span<const Complex> matrix;
    };
    const auto zero_vector = LatticeVector(model.ndim(), 0);
    const auto hoppings = core_detail::TightBindingModelAccess::hoppings(model);
    auto shifted_onsite = Matrix{};
    auto terms = std::vector<ShiftedHopping>{};
    terms.reserve(hoppings.size() + 1);
    auto has_onsite = false;
    for (const auto &term : hoppings) {
        if (term.lattice_vector == zero_vector) {
            has_onsite = true;
            if (mu != 0.0) {
                shifted_onsite = term.matrix;
                for (std::size_t index = 0; index < model.ndof(); ++index) {
                    shifted_onsite[
                        matrix_index(index, index, model.ndof())
                    ] -= mu;
                }
                terms.push_back(ShiftedHopping{
                    .lattice_vector = &term.lattice_vector,
                    .matrix = shifted_onsite,
                });
                continue;
            }
        }
        terms.push_back(ShiftedHopping{
            .lattice_vector = &term.lattice_vector,
            .matrix = term.matrix,
        });
    }
    if (!has_onsite && mu != 0.0) {
        shifted_onsite = Matrix(model.ndof() * model.ndof());
        for (std::size_t index = 0; index < model.ndof(); ++index) {
            shifted_onsite[
                matrix_index(index, index, model.ndof())
            ] = -mu;
        }
        terms.push_back(ShiftedHopping{
            .lattice_vector = &zero_vector,
            .matrix = shifted_onsite,
        });
    }

    const auto size = model.ndof();
    const auto active = layer.active_dimension;
    const auto safe = layer.inverse_safe_eigenvalues.size();
    if (
        layer.active_basis.size() != size * active || safe == 0 ||
        layer.safe_basis.size() != size * safe
    ) {
        return std::nullopt;
    }
    const auto term_count = terms.size();
    const auto block_columns = active * term_count;
    auto projected_columns = Matrix(size * block_columns);
    for (std::size_t term = 0; term < term_count; ++term) {
        linalg::matrix_multiply(
            'N',
            'N',
            size,
            active,
            size,
            Complex{1.0, 0.0},
            terms[term].matrix.data(),
            size,
            layer.active_basis.data(),
            size,
            Complex{0.0, 0.0},
            projected_columns.data() + term * size * active,
            size
        );
    }
    auto safe_coordinates = Matrix(safe * block_columns);
    linalg::matrix_multiply(
        'C',
        'N',
        safe,
        block_columns,
        size,
        Complex{1.0, 0.0},
        layer.safe_basis.data(),
        size,
        projected_columns.data(),
        size,
        Complex{0.0, 0.0},
        safe_coordinates.data(),
        safe
    );
    for (std::size_t column = 0; column < block_columns; ++column) {
        for (std::size_t index = 0; index < safe; ++index) {
            safe_coordinates[matrix_index(index, column, safe)] *=
                layer.inverse_safe_eigenvalues[index];
        }
    }
    auto resolved_columns = Matrix(size * block_columns);
    linalg::matrix_multiply(
        'N',
        'N',
        size,
        block_columns,
        safe,
        Complex{1.0, 0.0},
        layer.safe_basis.data(),
        size,
        safe_coordinates.data(),
        safe,
        Complex{0.0, 0.0},
        resolved_columns.data(),
        size
    );

    const auto coefficient_size = active * active;
    auto coefficient_indices = std::map<LatticeVector, std::size_t>{};
    auto coefficient_lattice_vectors = std::vector<LatticeVector>{};
    auto coefficients = Matrix{};
    const auto coefficient_index = [&](const LatticeVector &lattice_vector) {
        const auto next = coefficient_lattice_vectors.size();
        const auto [entry, inserted] = coefficient_indices.try_emplace(
            lattice_vector, next
        );
        if (inserted) {
            coefficient_lattice_vectors.push_back(lattice_vector);
            coefficients.resize(
                (next + 1) * coefficient_size,
                Complex{0.0, 0.0}
            );
        }
        return entry->second;
    };
    auto linear_coefficients = Matrix(active * block_columns);
    linalg::matrix_multiply(
        'C',
        'N',
        active,
        block_columns,
        size,
        Complex{1.0, 0.0},
        layer.active_basis.data(),
        size,
        projected_columns.data(),
        size,
        Complex{0.0, 0.0},
        linear_coefficients.data(),
        active
    );
    for (std::size_t term = 0; term < term_count; ++term) {
        const auto target = coefficient_index(
            *terms[term].lattice_vector
        ) * coefficient_size;
        for (std::size_t column = 0; column < active; ++column) {
            for (std::size_t row = 0; row < active; ++row) {
                coefficients[target + matrix_index(
                    row, column, active
                )] +=
                    linear_coefficients[matrix_index(
                        row,
                        term * active + column,
                        active
                    )];
            }
        }
    }
    auto quadratic_coefficients = Matrix(
        block_columns * block_columns
    );
    linalg::matrix_multiply(
        'C',
        'N',
        block_columns,
        block_columns,
        size,
        Complex{1.0, 0.0},
        projected_columns.data(),
        size,
        resolved_columns.data(),
        size,
        Complex{0.0, 0.0},
        quadratic_coefficients.data(),
        block_columns
    );
    for (std::size_t left = 0; left < term_count; ++left) {
        for (std::size_t right = 0; right < term_count; ++right) {
            const auto lattice_vector = combine_lattice_vectors(
                *terms[right].lattice_vector,
                zero_vector,
                *terms[left].lattice_vector
            );
            if (!lattice_vector.has_value()) {
                return std::nullopt;
            }
            const auto target = coefficient_index(
                *lattice_vector
            ) * coefficient_size;
            for (std::size_t column = 0; column < active; ++column) {
                for (std::size_t row = 0; row < active; ++row) {
                    coefficients[target + matrix_index(
                        row, column, active
                    )] -=
                        Complex{2.0, 0.0} * quadratic_coefficients[
                            matrix_index(
                                left * active + row,
                                right * active + column,
                                block_columns
                            )
                        ];
                }
            }
        }
    }
    auto processed_centers = std::vector<bool>(term_count, false);
    for (std::size_t center = 0; center < term_count; ++center) {
        if (processed_centers[center]) {
            continue;
        }
        const auto opposite_vector = combine_lattice_vectors(
            zero_vector,
            zero_vector,
            *terms[center].lattice_vector
        );
        if (!opposite_vector.has_value()) {
            return std::nullopt;
        }
        const auto partner = std::find_if(
            terms.begin(),
            terms.end(),
            [&](const ShiftedHopping &term) {
                return *term.lattice_vector == *opposite_vector;
            }
        );
        if (partner == terms.end()) {
            return std::nullopt;
        }
        const auto partner_index = static_cast<std::size_t>(
            std::distance(terms.begin(), partner)
        );
        auto applied = Matrix(size * block_columns);
        linalg::matrix_multiply(
            'N',
            'N',
            size,
            block_columns,
            size,
            Complex{1.0, 0.0},
            terms[center].matrix.data(),
            size,
            resolved_columns.data(),
            size,
            Complex{0.0, 0.0},
            applied.data(),
            size
        );
        auto cubic_coefficients = Matrix(
            block_columns * block_columns
        );
        linalg::matrix_multiply(
            'C',
            'N',
            block_columns,
            block_columns,
            size,
            Complex{1.0, 0.0},
            resolved_columns.data(),
            size,
            applied.data(),
            size,
            Complex{0.0, 0.0},
            cubic_coefficients.data(),
            block_columns
        );
        const auto accumulate_center = [&](std::size_t center_term,
                                           bool adjoint) {
            for (std::size_t right = 0; right < term_count; ++right) {
                for (std::size_t left = 0; left < term_count; ++left) {
                    const auto lattice_vector = combine_lattice_vectors(
                        *terms[center_term].lattice_vector,
                        *terms[right].lattice_vector,
                        *terms[left].lattice_vector
                    );
                    if (!lattice_vector.has_value()) {
                        return false;
                    }
                    const auto target = coefficient_index(
                        *lattice_vector
                    ) * coefficient_size;
                    for (std::size_t column = 0;
                         column < active;
                         ++column) {
                        for (std::size_t row = 0;
                             row < active;
                             ++row) {
                            const auto value = adjoint
                                ? std::conj(cubic_coefficients[
                                    matrix_index(
                                        right * active + column,
                                        left * active + row,
                                        block_columns
                                    )
                                ])
                                : cubic_coefficients[matrix_index(
                                    left * active + row,
                                    right * active + column,
                                    block_columns
                                )];
                            coefficients[target + matrix_index(
                                row, column, active
                            )] += value;
                        }
                    }
                }
            }
            return true;
        };
        if (!accumulate_center(center, false)) {
            return std::nullopt;
        }
        processed_centers[center] = true;
        if (partner_index != center) {
            if (!accumulate_center(partner_index, true)) {
                return std::nullopt;
            }
            processed_centers[partner_index] = true;
        }
    }

    auto result = ProjectedHoppingModel{
        .ndim = model.ndim(),
        .dimension = active,
    };
    result.lattice_vectors.reserve(coefficient_lattice_vectors.size());
    result.coefficients.reserve(coefficients.size());
    for (std::size_t coefficient = 0;
         coefficient < coefficient_lattice_vectors.size();
         ++coefficient) {
        const auto begin = coefficients.begin() +
            static_cast<std::ptrdiff_t>(coefficient * coefficient_size);
        const auto end = begin +
            static_cast<std::ptrdiff_t>(coefficient_size);
        if (std::all_of(begin, end, [](Complex value) {
                return value == Complex{0.0, 0.0};
            })) {
            continue;
        }
        result.lattice_vectors.push_back(
            coefficient_lattice_vectors[coefficient]
        );
        result.coefficients.insert(result.coefficients.end(), begin, end);
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
