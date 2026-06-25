#include "integration/charge.h"

#include "certificate/simplex_certificate.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

bool has_reusable_mu_range(
    const simplex_certificate::SimplexCertificate &certificate
) {
    if (certificate.lower_mu_bound > certificate.upper_mu_bound) {
        return false;
    }
    if (certificate.status == simplex_certificate::SimplexCertificateStatus::CertifiedGapped) {
        return true;
    }
    return certificate.status == simplex_certificate::SimplexCertificateStatus::Inconclusive &&
           certificate.has_occupation_bounds;
}

double occupied_fraction_derivative(
    std::span<const double> energies,
    double mu
) {
    const size_t dimension = energies.size() - 1;
    double result = 0.0;
    for (size_t i = 0; i < energies.size(); ++i) {
        const double distance = mu - energies[i];
        if (distance <= 0.0) {
            continue;
        }
        double denominator = 1.0;
        for (size_t j = 0; j < energies.size(); ++j) {
            if (j != i) {
                denominator *= energies[j] - energies[i];
            }
        }
        result += std::pow(distance, static_cast<double>(dimension - 1)) / denominator;
    }
    return static_cast<double>(dimension) * result;
}

bool strictly_ordered(std::span<const double> energies, double tol) {
    for (size_t index = 1; index < energies.size(); ++index) {
        if (energies[index] - energies[index - 1] <= tol) {
            return false;
        }
    }
    return true;
}

std::vector<double> sorted_band_energies(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t band
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<double> energies;
    energies.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        energies.push_back(cache.get(vertex_id).eigenvalues[band]);
    }
    std::stable_sort(energies.begin(), energies.end());
    return energies;
}

}  // namespace

const simplex_certificate::SimplexCertificate *ChargeCertificateCache::find(
    core::SimplexId simplex_id,
    double mu
) const {
    const auto found = records_.find(simplex_id);
    if (found == records_.end()) {
        return nullptr;
    }
    const auto &certificates = found->second;
    for (auto iter = certificates.rbegin(); iter != certificates.rend(); ++iter) {
        if (iter->lower_mu_bound <= mu && mu <= iter->upper_mu_bound) {
            return &(*iter);
        }
    }
    return nullptr;
}

void ChargeCertificateCache::insert(
    core::SimplexId simplex_id,
    simplex_certificate::SimplexCertificate certificate
) {
    if (!has_reusable_mu_range(certificate)) {
        return;
    }
    records_[simplex_id].push_back(std::move(certificate));
}

void ChargeCertificateCache::erase(core::SimplexId simplex_id) {
    records_.erase(simplex_id);
}

void ChargeCertificateCache::clear() {
    records_.clear();
}

size_t ChargeCertificateCache::size() const noexcept {
    auto count = size_t{0};
    for (const auto &[unused_simplex_id, certificates] : records_) {
        (void)unused_simplex_id;
        count += certificates.size();
    }
    return count;
}

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    bool certify,
    ChargeCertificateCache *certificate_cache
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    ChargeValue result;
    if (certify) {
        auto certificate = simplex_certificate::SimplexCertificate{};
        const auto *cached_certificate =
            certificate_cache == nullptr ? nullptr : certificate_cache->find(simplex_id, mu);
        if (cached_certificate != nullptr) {
            certificate = *cached_certificate;
        } else {
            certificate = simplex_certificate::certify_simplex_gap(
                mu,
                geometry,
                simplex_id,
                cache,
                0.0,
                workspace.tol(),
                true
            );
            if (certificate_cache != nullptr) {
                certificate_cache->insert(simplex_id, certificate);
            }
        }
        if (certificate.status == simplex_certificate::SimplexCertificateStatus::Inconclusive) {
            if (!certificate.has_occupation_bounds) {
                throw std::runtime_error("charge certificate: inconclusive simplex has no occupation bounds");
            }
            const auto unresolved_occupation_bound =
                certificate.upper_occupation_bound - certificate.lower_occupation_bound;
            result.certificate_error =
                static_cast<double>(unresolved_occupation_bound) * simplex.volume;
        }
    }

    for (size_t band = 0; band < workspace.ndof(); ++band) {
        const auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{.level = mu, .level_tolerance = workspace.tol()}
        );

        if (moments.kind == cut::SimplexCutKind::on_level) {
            result.charge += 0.5 * simplex.volume;
            continue;
        }

        result.charge += moments.volume;
        if (moments.kind != cut::SimplexCutKind::partial) {
            continue;
        }

        const auto energies = sorted_band_energies(geometry, simplex_id, cache, band);
        const auto energy_span = std::span<const double>(energies.data(), energies.size());
        if (strictly_ordered(energy_span, workspace.tol())) {
            result.derivative +=
                simplex.volume * occupied_fraction_derivative(energy_span, mu);
        }
    }
    return result;
}

}  // namespace lineartetrahedron
