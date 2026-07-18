#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace fermisimplex::linalg {

using Complex = std::complex<double>;

void matrix_multiply(
    char left_operation,
    char right_operation,
    size_t rows,
    size_t columns,
    size_t inner_dimension,
    Complex alpha,
    const Complex *left,
    size_t left_leading_dimension,
    const Complex *right,
    size_t right_leading_dimension,
    Complex beta,
    Complex *result,
    size_t result_leading_dimension
);

void hermitian_rank_2k_update(
    char operation,
    size_t size,
    size_t inner_dimension,
    Complex alpha,
    const Complex *left,
    size_t left_leading_dimension,
    const Complex *right,
    size_t right_leading_dimension,
    double beta,
    Complex *result,
    size_t result_leading_dimension
);

void hermitian_matrix_multiply(
    char side,
    size_t rows,
    size_t columns,
    Complex alpha,
    const Complex *hermitian,
    size_t hermitian_leading_dimension,
    const Complex *other,
    size_t other_leading_dimension,
    Complex beta,
    Complex *result,
    size_t result_leading_dimension
);

int cholesky_factor_lower(Complex *matrix, size_t size);

void diagonalize_hermitian_in_place(
    std::vector<Complex> &matrix,
    std::vector<double> &eigenvalues,
    size_t size,
    bool compute_vectors,
    const char *context
);

}  // namespace fermisimplex::linalg
