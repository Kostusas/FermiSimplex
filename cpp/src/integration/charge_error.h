#pragma once

#include <fermisimplex/certification.h>
#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include <adaptivesimplex/core/geometry.h>

#include <cstdint>
#include <memory>

namespace fermisimplex::certification {
class PreparedSimplexCertificate;
}

namespace fermisimplex::integration_detail {

struct ChargeProfile;

class ChargeErrorEstimator {
public:
    ChargeErrorEstimator(
        SpectralMesh &mesh,
        double mu,
        std::uint32_t depth,
        ChargeErrorStats &stats,
        ChargeProfile *profile = nullptr
    );
    ~ChargeErrorEstimator();

    ChargeErrorEstimator(const ChargeErrorEstimator &) = delete;
    ChargeErrorEstimator &operator=(const ChargeErrorEstimator &) = delete;

    double estimate(
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::SimplexId simplex_id,
        double linear_charge,
        certification::SimplexCertificate root_certificate,
        const certification::PreparedSimplexCertificate &root_preparation
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fermisimplex::integration_detail
