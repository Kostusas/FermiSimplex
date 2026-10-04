#pragma once

#include <fermisimplex/spectral_mesh.h>
#include <optional>
#include <unordered_map>

namespace fermisimplex::occupation_detail {

// Scalar proof data only: the anchor basis already lives in the vertex cache.
// Geometry ids and the Hamiltonian stay fixed for the lifetime of a mesh.
struct Certificate {
    std::size_t negative = 0;
    std::size_t positive = 0;
    double mu = 0;
    double negative_margin = 0;
    double positive_margin = 0;
    double eta = 0;
    std::optional<double> explicit_remainder;
};

struct CertificateCache {
    std::unordered_map<adaptivesimplex::core::SimplexId, Certificate> entries;

    static CertificateCache &get(const SpectralMesh &mesh) {
        return *mesh.certificates_;
    }
};

}  // namespace fermisimplex::occupation_detail
