#pragma once

#include "occupation/residual_bounds.h"
#include <fermisimplex/occupation.h>

namespace fermisimplex::occupation_detail {

// Fit Y=A-B†X-X†B+X†DX and enclose S=Y-F†D^-1 F, F=B-DX.
// Safe indices contain the occupied sector followed by the empty sector.
double schur_allowance(const Polynomial &full, const std::vector<std::size_t> &safe,
    const std::vector<std::size_t> &active, const std::vector<double> &reference_safe,
    const std::vector<Matrix> &solution, std::size_t negative, double eta, double gap,
    ChargeErrorStats &stats, Polynomial &fitted, ResidualBounds &residual);

}  // namespace fermisimplex::occupation_detail
