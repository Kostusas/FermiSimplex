#pragma once

#include "occupation/bernstein.h"
#include <map>
#include <memory>

namespace fermisimplex::occupation_detail {

// Rotate all controls with two matrix products. Transposing each small block
// between products turns the second basis multiplication into another batch.
inline Matrix rotate_controls(const Matrix &source, const Matrix &basis,
                              std::size_t size, Matrix &scratch) {
    Matrix result(source.size());
    scratch.resize(source.size());
    const auto columns = source.size() / size;
    const auto transpose_blocks = [&](Matrix &values) {
        for (std::size_t offset = 0; offset < values.size(); offset += size * size)
            for (std::size_t j = 0; j < size; ++j)
                for (std::size_t i = 0; i < j; ++i)
                    std::swap(values[offset + i + j * size], values[offset + j + i * size]);
    };
    linalg::matrix_multiply('C', 'N', size, columns, size, 1., basis.data(), size,
        source.data(), size, 0., result.data(), size);
    transpose_blocks(result);
    linalg::matrix_multiply('T', 'N', size, columns, size, 1., basis.data(), size,
        result.data(), size, 0., scratch.data(), size);
    transpose_blocks(scratch);
    result.swap(scratch);
    return result;
}

// Degree-six envelopes formed from residual Gram matrices and the quadratic
// model, with constant certified safe-sector metrics.
// Layout and midpoint restriction weights depend only on simplex dimension.
struct ResidualLayout {
    struct Term {
        std::size_t index;
        double weight;
    };
    struct GramTerm {
        std::size_t left, right, target;
        double weight;
    };
    std::size_t vertices;
    std::vector<Index> indices;
    std::vector<Index> residual_indices;
    std::vector<std::vector<std::vector<Term>>> splits;
    std::vector<std::vector<double>> quadratic_weights;
    std::vector<GramTerm> gram_terms;

    explicit ResidualLayout(std::size_t v) : vertices(v), splits(v * v) {
        std::map<Index, std::size_t> lookup;
        indices = bernstein_indices(v, 6);
        residual_indices = bernstein_indices(v, 3);
        for (std::size_t i = 0; i < indices.size(); ++i) lookup.emplace(indices[i], i);
        const auto &cubic = residual_indices;
        for (std::size_t i = 0; i < cubic.size(); ++i)
            for (std::size_t j = i; j < cubic.size(); ++j) {
                Index sum(v);
                for (std::size_t k = 0; k < v; ++k) sum[k] = cubic[i][k] + cubic[j][k];
                gram_terms.push_back({i, j, lookup.at(sum),
                    multinomial(cubic[i]) * multinomial(cubic[j]) / multinomial(sum)});
            }
        for (const auto &a : indices) {
            std::vector<double> weights(v * (v + 1) / 2);
            for (std::size_t j = 0; j < v; ++j)
                for (std::size_t i = 0; i <= j; ++i)
                    weights[j * (j + 1) / 2 + i] = i == j
                        ? double(a[i]) * (double(a[i]) - 1) / 30.
                        : double(a[i] * a[j]) / 15.;
            quadratic_weights.push_back(std::move(weights));
        }
        for (std::size_t replaced = 0; replaced < v; ++replaced)
            for (std::size_t other = 0; other < v; ++other) {
                if (replaced == other) continue;
                auto &table = splits[replaced * v + other];
                for (const auto &a : indices) {
                    const auto count = a[replaced];
                    std::vector<Term> terms;
                    double weight = std::ldexp(1., -static_cast<int>(count));
                    for (std::size_t k = 0; k <= count; ++k) {
                        auto b = a;
                        b[replaced] = k;
                        b[other] += count - k;
                        terms.push_back({lookup.at(b), weight});
                        if (k < count) weight *= double(count - k) / (k + 1);
                    }
                    table.push_back(std::move(terms));
                }
            }
    }

    static std::shared_ptr<const ResidualLayout> get(std::size_t vertices) {
        thread_local std::map<std::size_t, std::shared_ptr<const ResidualLayout>> layouts;
        auto &layout = layouts[vertices];
        if (!layout) layout = std::make_shared<ResidualLayout>(vertices);
        return layout;
    }
};

inline void add_gram_controls(const std::vector<Matrix> &controls, std::size_t rows,
    std::size_t size, const ResidualLayout &layout, Matrix &output) {
    Matrix pair(size * size);
    for (const auto &term : layout.gram_terms) {
        linalg::matrix_multiply('C', 'N', size, size, rows, term.weight,
            controls[term.left].data(), rows,controls[term.right].data(), rows,
            0., pair.data(), size);
        auto *target = output.data() + term.target * size * size;
        for (std::size_t col = 0; col < size; ++col)
            for (std::size_t row = 0; row < size; ++row)
                target[row + col * size] += pair[row + col * size] +
                    (term.left == term.right ? std::complex<double>{}
                                            : std::conj(pair[col + row * size]));
    }
}

// Polynomial matrix bounds: lower(k)-remainder*I <= S(k) <= upper(k)+remainder*I.
struct ResidualMatrices {
    std::shared_ptr<const ResidualLayout> layout;
    std::size_t entries;
    Matrix upper, lower;
    double remainder = 0;

    ResidualMatrices(std::size_t vertices, std::size_t size)
        : layout(ResidualLayout::get(vertices)),
          entries(size * size), upper(layout->indices.size() * entries), lower(upper) {}

    void add_model(const Polynomial &polynomial) {
        for (std::size_t c = 0; c < layout->indices.size(); ++c)
            for (std::size_t k = 0; k < entries; ++k) {
                std::complex<double> value{};
                for (std::size_t p = 0; p < polynomial.controls.size(); ++p)
                    value += layout->quadratic_weights[c][p] * polynomial.controls[p][k];
                upper[c * entries + k] += value;
                lower[c * entries + k] = value - lower[c * entries + k];
            }
    }

    ResidualMatrices bisected(std::size_t left, std::size_t right,
                              std::size_t replaced) const {
        ResidualMatrices result = *this;
        const auto other = replaced == left ? right : left;
        const auto &table = layout->splits[replaced * layout->vertices + other];
        for (std::size_t c = 0; c < table.size(); ++c) {
            if (table[c].size() == 1) continue;
            auto *upper_control = result.upper.data() + c * entries;
            auto *lower_control = result.lower.data() + c * entries;
            std::fill_n(upper_control, entries, std::complex<double>{});
            std::fill_n(lower_control, entries, std::complex<double>{});
            for (const auto &term : table[c])
                for (std::size_t k = 0; k < entries; ++k) {
                    upper_control[k] += term.weight * upper[term.index * entries + k];
                    lower_control[k] += term.weight * lower[term.index * entries + k];
                }
        }
        return result;
    }
};

}  // namespace fermisimplex::occupation_detail
