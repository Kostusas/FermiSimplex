#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace lineartetrahedron::simplex_rules {

std::pair<double, double> simplex_fraction_and_derivative(
    const double *energies,
    double mu,
    size_t dimension,
    double tol
);

std::vector<double> occupied_linear_moment_fractions(
    const double *sorted_energies,
    size_t ndim,
    double mu,
    double tol
);

}  // namespace lineartetrahedron::simplex_rules
