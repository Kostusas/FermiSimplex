#include "integration/charge_certificate_cache.h"

#include <utility>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;
namespace cert = simplex_certificate;

namespace {

bool is_cacheable(const cert::SimplexCertificate &certificate) {
    if (!cert::has_mu_interval(certificate)) {
        return false;
    }
    if (certificate.status == cert::SimplexCertificateStatus::CertifiedGapped) {
        return true;
    }
    return certificate.status == cert::SimplexCertificateStatus::Inconclusive;
}

}  // namespace

const cert::SimplexCertificate *ChargeCertificateCache::find(
    core::SimplexId simplex_id,
    double mu,
    double energy_bound
) const {
    const auto found = records_.find(simplex_id);
    if (found == records_.end()) {
        return nullptr;
    }
    const auto &records = found->second;
    for (auto iter = records.rbegin(); iter != records.rend(); ++iter) {
        if (
            iter->energy_bound >= energy_bound &&
            cert::reusable_at(*iter, mu)
        ) {
            return &(*iter);
        }
    }
    return nullptr;
}

void ChargeCertificateCache::insert(
    core::SimplexId simplex_id,
    cert::SimplexCertificate certificate
) {
    if (!is_cacheable(certificate)) {
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
    for (const auto &[unused_simplex_id, records] : records_) {
        (void)unused_simplex_id;
        count += records.size();
    }
    return count;
}

}  // namespace lineartetrahedron
