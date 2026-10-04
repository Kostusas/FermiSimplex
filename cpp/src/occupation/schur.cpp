#include "occupation/schur.h"
#include "occupation/bernstein.h"

namespace fermisimplex::occupation_detail {

namespace {

double scaled_sector_margin(const Matrix &safe, const std::vector<double> &weights,
    std::size_t start, std::size_t count, int sign, double eta, double &scale,
    ChargeErrorStats &stats) {
    Matrix sector(count * count);
    for (std::size_t row = 0; row < count; ++row)
        for (std::size_t col = 0; col < count; ++col) {
            auto value = static_cast<double>(sign) *
                safe[start + row + (start + col) * weights.size()];
            if (row == col) value -= eta;
            sector[row + col * count] =
                weights[start + row] * value * weights[start + col];
        }
    scale = std::max(scale, norm(sector));
    double lower = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0; row < count; ++row) {
        double margin = sector[row + row * count].real();
        for (std::size_t col = 0; col < count; ++col)
            if (row != col) margin -= std::abs(sector[row + col * count]);
        lower = std::min(lower, margin);
    }
    if (lower > 0) return lower;
    std::vector<double> eigenvalues;
    linalg::diagonalize_hermitian_in_place(sector, eigenvalues, count,
        false, "scaled safe-sector margin");
    ++stats.norm_eigensystems;
    return eigenvalues.front();
}

BernsteinPolynomial fit_quadratic(const BernsteinPolynomial &source, Polynomial &fitted) {
    const auto v = source.vertices;
    BernsteinPolynomial result{v, source.rows, source.cols, {}};
    for (std::size_t i = 0; i < v; ++i) {
        Weights w(v);
        w[i] = 1;
        fitted.at(i, i) = evaluate(source, w);
    }
    for (std::size_t i = 0; i < v; ++i)
        for (std::size_t j = i; j < v; ++j) {
            if (i != j) {
                Weights w(v);
                w[i] = w[j] = .5;
                auto control = evaluate(source, w);
                for (std::size_t k = 0; k < control.size(); ++k)
                    control[k] = 2. * control[k] -
                        .5 * (fitted.at(i, i)[k] + fitted.at(j, j)[k]);
                fitted.at(i, j) = std::move(control);
            }
            Index key(v);
            ++key[i];
            ++key[j];
            result.controls.emplace(key, fitted.at(i, j));
        }
    return result;
}

// The two safe signs give separate Loewner bounds on -F†D^-1 F.
// Enclose each weighted residual Gram matrix by a constant diagonal matrix.
void bound_residual_sector(const BernsteinPolynomial &residual,
    const std::vector<double> &weights, std::size_t start, std::size_t count,
    double margin, double gap, double uncertainty, Matrix &output) {
    const auto q = residual.cols, s = residual.rows;
    Matrix sector(count * q), gram(q * q);
    std::vector<double> radius(q);
    double maximum = 0, perturbation = 0;
    for (const auto &[key, control] : residual.controls) {
        for (std::size_t row = 0; row < count; ++row) {
            const auto weight = margin > 0 ? weights[start + row] / std::sqrt(margin)
                                          : 1 / std::sqrt(gap);
            perturbation = std::max(perturbation, weight * uncertainty);
            for (std::size_t col = 0; col < q; ++col)
                sector[row + col * count] = weight * control[start + row + col * s];
        }
        maximum = std::max(maximum, norm(sector));
        // Jensen bounds the residual Gram matrix by the convex combination
        // of the control Gram matrices. Diagonal dominance encloses each one.
        linalg::matrix_multiply('C', 'N', q, q, count, 1.,
            sector.data(), count, sector.data(), count, 0., gram.data(), q);
        for (std::size_t row = 0; row < q; ++row) {
            double bound = gram[row + row * q].real();
            for (std::size_t col = 0; col < q; ++col)
                if (row != col) bound += std::abs(gram[row + col * q]);
            radius[row] = std::max(radius[row], bound);
        }
    }
    for (std::size_t band = 0; band < q; ++band)
        output[band + band * q] += radius[band];
    const auto padding = 2 * maximum * perturbation + perturbation * perturbation;
    for (std::size_t band = 0; band < q; ++band)
        output[band + band * q] += padding;
}

}  // namespace

double schur_allowance(const Polynomial &full, const std::vector<std::size_t> &safe,
    const std::vector<std::size_t> &active, const std::vector<double> &d0,
    const std::vector<Matrix> &solution,
    std::size_t negative, double eta, double gap, ChargeErrorStats &stats, Polynomial &fitted,
    ResidualBounds &residual_matrices) {
    const auto v = full.vertices, n = full.size, s = safe.size(), q = active.size();
    BernsteinPolynomial ap{v, q, q, {}}, bp{v, s, q, {}}, dp{v, s, s, {}}, xpoly{v, s, q, {}};
    std::vector<double> weights(s);
    double max_weight = 0, x = 0, gamma = std::numeric_limits<double>::infinity();
    double scale = 1, scaled_scale = 1;
    double sector_gamma[2] = {gamma, gamma};
    for (std::size_t r = 0; r < s; ++r) {
        weights[r] = 1 / std::sqrt(std::abs(d0[r]));
        max_weight = std::max(max_weight, weights[r]);
    }
    for (std::size_t i = 0; i < v; ++i) {
        Index key(v);
        key[i] = 1;
        xpoly.controls.emplace(key, solution[i]);
        x = std::max(x, norm(solution[i]));
    }
    for (std::size_t i = 0; i < v; ++i)
        for (std::size_t j = i; j < v; ++j) {
            Index key(v);
            ++key[i];
            ++key[j];
            const auto &control = full.at(i, j);
            scale = std::max(scale, norm(control));
            auto b = block(control, n, safe, active);
            bp.controls.emplace(key, std::move(b));
            ap.controls.emplace(key, block(control, n, active, active));
            auto d = block(control, n, safe, safe);
            for (const auto sign : {-1, 1}) {
                const auto start = sign == -1 ? 0 : negative;
                const auto count = sign == -1 ? negative : s - negative;
                if (count == 0) continue;
                const auto lower = scaled_sector_margin(d, weights, start, count,
                    sign, eta, scaled_scale, stats);
                gamma = std::min(gamma, lower);
                const auto sector = sign == -1 ? 0 : 1;
                sector_gamma[sector] = std::min(sector_gamma[sector], lower);
            }
            dp.controls.emplace(key, std::move(d));
        }
    // Any affine X is permitted by the exact congruence identity.
    const auto dx = product(dp, xpoly);
    const auto fpoly = subtract(elevated(bp), dx);
    const auto xb = product(adjoint(xpoly), bp);
    auto ypoly = subtract(subtract(elevated(elevated(ap)), elevated(xb)),
                          elevated(adjoint(xb)));
    const auto xdx = product(adjoint(xpoly), dx);
    for (const auto &[key, c] : xdx.controls) {
        auto &y = ypoly.controls.at(key);
        for (std::size_t k = 0; k < c.size(); ++k) y[k] += c[k];
    }
    const auto pp = fit_quadratic(ypoly, fitted);
    const auto gpoly = subtract(ypoly, elevated(elevated(pp)));
    // These scalar bounds provide a cheap first test before matrix envelopes.
    double f = 0, fw = 0, g = 0;
    for (const auto &[key, c] : fpoly.controls) {
        f = std::max(f, norm(c));
        auto weighted = c;
        for (std::size_t col = 0; col < q; ++col)
            for (std::size_t row = 0; row < s; ++row)
                weighted[row + col * s] *= weights[row];
        fw = std::max(fw, norm(weighted));
    }
    std::vector<double> rows(q);
    for (const auto &[key, c] : gpoly.controls)
        g = std::max(g, hermitian_norm_bound(c, rows));
    const auto roundoff = 256 * std::numeric_limits<double>::epsilon() * scale * (1 + x * x);
    const auto e = eta * std::sqrt(1 + x * x) + roundoff;
    auto correction = (f + e) * (f + e) / gap;
    gamma -= 256 * std::numeric_limits<double>::epsilon() * scaled_scale;
    if (gamma > 0) correction = std::min(correction,
        (fw + max_weight * e) * (fw + max_weight * e) / gamma);
    residual_matrices.remainder = g + eta * (1 + x * x) + roundoff;
    for (std::size_t sector = 0; sector < 2; ++sector) {
        const auto start = sector == 0 ? 0 : negative;
        const auto count = sector == 0 ? negative : s - negative;
        if (count == 0) continue;
        const auto margin = sector_gamma[sector] -
            256 * std::numeric_limits<double>::epsilon() * scaled_scale;
        auto &output = sector == 0 ? residual_matrices.upper : residual_matrices.lower;
        bound_residual_sector(fpoly, weights, start, count, margin, gap, e,
                              output);
    }
    return g + eta * (1 + x * x) + roundoff + correction;
}
}  // namespace fermisimplex::occupation_detail
