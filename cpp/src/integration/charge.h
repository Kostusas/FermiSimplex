#pragma once

#include "certificate/simplex_certificate.h"
#include "core/types.h"
#include "core/vertex_spectra.h"
#include "integration/workspace.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace lineartetrahedron {

class ChargeCertificateCache {
public:
    const simplex_certificate::SimplexCertificate *find(
        adaptivesimplex::core::SimplexId simplex_id,
        double mu
    ) const;
    void insert(
        adaptivesimplex::core::SimplexId simplex_id,
        simplex_certificate::SimplexCertificate certificate
    );
    void erase(adaptivesimplex::core::SimplexId simplex_id);
    void clear();
    size_t size() const noexcept;

private:
    std::unordered_map<
        adaptivesimplex::core::SimplexId,
        std::vector<simplex_certificate::SimplexCertificate>
    > records_;
};

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::SimplexId simplex_id,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    bool certify = true,
    ChargeCertificateCache *certificate_cache = nullptr
);

}  // namespace lineartetrahedron
