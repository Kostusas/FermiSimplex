#pragma once

#include "occupation/polynomial.h"
#include <map>
#include <numeric>

namespace fermisimplex::occupation_detail {

// Bernstein algebra for rectangular matrix polynomials of matching degree.
using Index = std::vector<std::size_t>;
struct BernsteinPolynomial {
    std::size_t vertices, rows, cols;
    std::map<Index, Matrix> controls;
};
inline double multinomial(const Index &a) {
    double result = 1;
    std::size_t total = 0;
    for (auto count : a) for (std::size_t i = 1; i <= count; ++i)
        result *= static_cast<double>(++total) / i;
    return result;
}
inline BernsteinPolynomial product(const BernsteinPolynomial &a, const BernsteinPolynomial &b) {
    BernsteinPolynomial out{a.vertices, a.rows, b.cols, {}};
    for (const auto &[i, x] : a.controls) for (const auto &[j, y] : b.controls) {
        Index k(i.size());
        for (std::size_t l = 0; l < k.size(); ++l) k[l] = i[l] + j[l];
        auto [it, inserted] = out.controls.try_emplace(k, a.rows * b.cols);
        const auto weight = multinomial(i) * multinomial(j) / multinomial(k);
        linalg::matrix_multiply('N', 'N', a.rows, b.cols, a.cols,
            weight, x.data(), a.rows, y.data(), b.rows, 1., it->second.data(), a.rows);
    }
    return out;
}
inline BernsteinPolynomial adjoint(const BernsteinPolynomial &a) {
    BernsteinPolynomial out{a.vertices, a.cols, a.rows, {}};
    for (const auto &[i, x] : a.controls) {
        Matrix y(x.size());
        for (std::size_t r = 0; r < a.rows; ++r)
            for (std::size_t c = 0; c < a.cols; ++c)
                y[c + r * a.cols] = std::conj(x[r + c * a.rows]);
        out.controls.emplace(i, std::move(y));
    }
    return out;
}
inline BernsteinPolynomial elevated(const BernsteinPolynomial &a) {
    BernsteinPolynomial out{a.vertices, a.rows, a.cols, {}};
    for (const auto &[i, x] : a.controls) {
        const auto degree = std::accumulate(i.begin(), i.end(), std::size_t{0});
        for (std::size_t v = 0; v < a.vertices; ++v) {
            auto j = i; ++j[v];
            auto [it, inserted] = out.controls.try_emplace(j, a.rows * a.cols);
            const auto weight = static_cast<double>(j[v]) / (degree + 1);
            for (std::size_t k = 0; k < x.size(); ++k) it->second[k] += weight * x[k];
        }
    }
    return out;
}
inline BernsteinPolynomial subtract(BernsteinPolynomial a, const BernsteinPolynomial &b) {
    for (const auto &[i, y] : b.controls) {
        auto &x = a.controls.at(i);
        for (std::size_t k = 0; k < x.size(); ++k) x[k] -= y[k];
    }
    return a;
}

inline Matrix evaluate(const BernsteinPolynomial &p, const Weights &w) {
    Matrix result(p.rows * p.cols);
    for (const auto &[a, c] : p.controls) {
        auto weight = multinomial(a);
        for (std::size_t j = 0; j < w.size(); ++j) weight *= std::pow(w[j], a[j]);
        for (std::size_t k = 0; k < c.size(); ++k) result[k] += weight * c[k];
    }
    return result;
}

}  // namespace fermisimplex::occupation_detail
