#pragma once

#include "occupation/polynomial.h"
#include <fermisimplex/occupation.h>

namespace fermisimplex::occupation_detail {

struct Sectors {
    std::size_t negative = 0;
    std::size_t positive = 0;
    double gap = 0;
};

Sectors sign_sectors(const Polynomial &polynomial, double allowance,
                     ChargeErrorStats &stats);

struct Model {
    Polynomial polynomial;
    std::size_t safe_occupation = 0;
    double epsilon = 0;
    double eta = 0;
    double delta = 0;
};

Model build_model(const SpectralMesh &mesh,
                  adaptivesimplex::core::SimplexId simplex_id, double mu,
                  ChargeErrorStats &stats, std::optional<double> remainder);

}  // namespace fermisimplex::occupation_detail
