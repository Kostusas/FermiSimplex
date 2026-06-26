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
    double mu
) const {
    const auto found = records_.find(simplex_id);
    if (found == records_.end()) {
        return nullptr;
    }
    const auto &certificates = found->second;
    for (auto iter = certificates.rbegin(); iter != certificates.rend(); ++iter) {
        if (cert::reusable_at(*iter, mu)) {
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
    for (const auto &[unused_simplex_id, certificates] : records_) {
        (void)unused_simplex_id;
        count += certificates.size();
    }
    return count;
}

}  // namespace lineartetrahedron
