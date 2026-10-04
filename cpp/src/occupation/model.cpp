#include "occupation/model.h"

#include "occupation/certificate_cache.h"
#include "occupation/probes.h"
#include "occupation/schur.h"

#include <limits>
#include <numeric>
#include <stdexcept>

namespace fermisimplex::occupation_detail {
namespace {

std::vector<std::size_t> indices(std::size_t first, std::size_t last) {
    std::vector<std::size_t> result(last - first);
    std::iota(result.begin(), result.end(), first);
    return result;
}

struct Interpolant {
    Polynomial polynomial;
    double remainder;
};

Interpolant interpolate(const SpectralMesh &mesh,
                        adaptivesimplex::core::SimplexId simplex_id, double mu,
                        ChargeErrorStats &stats, std::optional<double> bound,
                        bool sample_remainder = true) {
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    const auto v = simplex.vertex_ids.size(), n = mesh.ndof();
    Polynomial polynomial{v, n};
    std::vector<Weights> points;
    for (std::size_t i = 0; i < v; ++i) {
        const auto vertex = simplex.vertex_ids[i];
        points.push_back(mesh.geometry().vertices().dyadic_vertex(vertex).to_point());
    }
    const auto evaluate = [&](const Weights &point) {
        auto matrix = mesh.hamiltonian(point);
        ++stats.hamiltonian_evaluations;
        for (std::size_t i = 0; i < n; ++i) matrix[i + i * n] -= mu;
        return matrix;
    };
    const auto at_weights = [&](const Weights &weights) {
        Weights point(mesh.ndim());
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t axis = 0; axis < point.size(); ++axis)
                point[axis] += weights[i] * points[i][axis];
        return evaluate(point);
    };
    for (std::size_t i = 0; i < v; ++i) polynomial.at(i, i) = evaluate(points[i]);
    for (std::size_t i = 0; i < v; ++i)
        for (std::size_t j = i + 1; j < v; ++j) {
            Weights weights(v);
            weights[i] = weights[j] = .5;
            auto midpoint = at_weights(weights);
            for (std::size_t k = 0; k < n * n; ++k)
                midpoint[k] = 2. * midpoint[k] -
                    .5 * (polynomial.at(i, i)[k] + polynomial.at(j, j)[k]);
            polynomial.at(i, j) = std::move(midpoint);
        }
    double defect = 0;
    if (sample_remainder && !bound) {
        std::vector<double> row_sums(n);
        for (const auto &weights : probe_weights(v)) {
            auto residual = at_weights(weights);
            polynomial.add_blossom_to(residual, weights, weights, -1.);
            defect = std::max(defect, hermitian_norm_bound(residual, row_sums));
        }
    }
    if (!sample_remainder) return {std::move(polynomial), 0};
    double scale = std::max(1., std::abs(mu));
    for (const auto &control : polynomial.controls) scale = std::max(scale, norm(control));
    const auto roundoff = 64 * std::numeric_limits<double>::epsilon() * scale;
    return {std::move(polynomial),
            bound.value_or(probe_remainder_factor(mesh.ndim()) * defect) + roundoff};
}

std::vector<std::size_t> safe_indices(std::size_t n, const Certificate &proof) {
    auto safe = indices(0, proof.negative);
    const auto positive = indices(n - proof.positive, n);
    safe.insert(safe.end(), positive.begin(), positive.end());
    return safe;
}

Model reduce_model(const Polynomial &full, const Eigensystem &anchor,
                   const Certificate &proof, double mu, double eta, double gap,
                   ChargeErrorStats &stats) {
    const auto v = full.vertices;
    const auto q = full.size - proof.negative - proof.positive;
    const auto active = indices(proof.negative, full.size - proof.positive);
    const auto safe = safe_indices(anchor.eigenvalues.size(), proof);
    const auto s = safe.size();
    std::vector<double> d0(s);
    for (std::size_t i = 0; i < s; ++i)
        d0[i] = anchor.eigenvalues[safe[i]] - mu;
    std::vector<Matrix> solution;
    for (std::size_t i = 0; i < v; ++i) {
        auto solved = block(full.at(i, i), full.size, safe, active);
        auto d = block(full.at(i,i), full.size, safe, safe);
        if (!linalg::solve_linear_system_in_place(d, solved, s, q, "vertex safe solve"))
            throw std::runtime_error("certified safe block solve failed");
        solution.push_back(std::move(solved));
    }
    Polynomial reduced{v, q};
    ResidualMatrices residual{v, q};
    const auto epsilon = schur_allowance(full, safe, active, d0,
        solution, proof.negative, eta, gap, stats, reduced, residual);
    ++stats.schur_reductions;
    stats.schur_evaluations += v * (v + 1) / 2;
    if (stats.minimum_active_dimension == 0 || q < stats.minimum_active_dimension)
        stats.minimum_active_dimension = q;
    return {std::move(reduced), proof.negative, epsilon, eta, gap, std::move(residual)};
}

}  // namespace

Model build_model(const SpectralMesh &mesh,
                  adaptivesimplex::core::SimplexId simplex_id, double mu,
                  ChargeErrorStats &stats, std::optional<double> remainder) {
    auto &cache = CertificateCache::get(mesh).entries;
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    const auto &anchor = mesh.eigensystems().get(simplex.vertex_ids.front());
    const auto n = mesh.ndof(), v = simplex.vertex_ids.size();
    if (const auto found = cache.find(simplex_id); found != cache.end() &&
        found->second.explicit_remainder == remainder) {
        const auto &proof = found->second;
        const auto shift = mu - proof.mu;
        // Cover the changed diagonal-subtraction scale without resampling H.
        // Keep the original reference: repeated queries must not erode margins.
        const auto roundoff = 64 * std::numeric_limits<double>::epsilon() *
                              (1 + std::sqrt(static_cast<double>(n))) * std::abs(shift);
        const auto gap = std::min(proof.negative_margin + shift,
                                  proof.positive_margin - shift) - roundoff;
        if (std::isfinite(gap) && gap > 0) {
            ++stats.certificate_reuses;
            const auto q = n - proof.negative - proof.positive;
            const auto eta = proof.eta + roundoff;
            stats.initial_active_dimension_sum += q;
            if (q == 0) return {Polynomial{v, 0}, proof.negative, eta, eta, gap};
            auto full = interpolate(mesh, simplex_id, mu, stats, remainder, false).polynomial;
            // Reuse the sign proof, then rebuild the mu-dependent safe solve
            // and its residual envelopes in the same anchor basis.
            for (auto &control : full.controls) control = rotate(control, anchor.eigenvectors, n);
            return reduce_model(full, anchor, proof, mu, eta, gap, stats);
        }
    }
    ++stats.certificate_builds;
    auto interpolation = interpolate(mesh, simplex_id, mu, stats, remainder);
    auto full = std::move(interpolation.polynomial);
    for (std::size_t i = 0; i < v; ++i)
        for (std::size_t j = i; j < v; ++j) {
            if (i == 0 && j == 0) {
                auto &control = full.at(0, 0);
                std::fill(control.begin(), control.end(), 0.);
                for (std::size_t band = 0; band < n; ++band)
                    control[band + band * n] = anchor.eigenvalues[band] - mu;
            } else {
                full.at(i, j) = rotate(full.at(i, j), anchor.eigenvectors, n);
            }
        }
    const auto eta = interpolation.remainder;
    const auto sectors = sign_sectors(full, eta, stats);
    const auto q = n - sectors.negative - sectors.positive;
    stats.initial_active_dimension_sum += q;
    cache.erase(simplex_id);
    if (q == n) return {std::move(full), 0, eta, eta, 0};
    Certificate proof{sectors.negative, sectors.positive, mu,
                      sectors.negative_margin, sectors.positive_margin,
                      eta, remainder};
    cache.emplace(simplex_id, proof);
    if (q == 0) return {Polynomial{v, 0}, sectors.negative, eta, eta, sectors.gap};
    return reduce_model(full, anchor, proof, mu, eta, sectors.gap, stats);
}

}  // namespace fermisimplex::occupation_detail
