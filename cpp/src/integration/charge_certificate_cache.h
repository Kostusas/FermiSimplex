#pragma once

#include "certificate/simplex_certificate.h"

#include <adaptivesimplex/core/types.h>

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

}  // namespace lineartetrahedron
