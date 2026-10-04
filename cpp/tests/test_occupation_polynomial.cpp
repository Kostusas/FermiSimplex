#include "occupation/polynomial.h"
#include "occupation/residual_matrices.h"
#include "test_helpers.h"

#include <iostream>
#include <numeric>
#include <random>

using namespace fermisimplex::occupation_detail;
using namespace fermisimplex::test;

Matrix evaluate_residual(const ResidualMatrices &p, const Weights &weights, bool occupied) {
    const auto &controls = occupied ? p.upper : p.lower;
    Matrix result(p.entries);
    for (std::size_t c = 0; c < p.layout->indices.size(); ++c) {
        double weight = 720.;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const auto count = p.layout->indices[c][i];
            for (std::size_t k = 1; k <= count; ++k) weight *= weights[i]/k;
        }
        for (std::size_t k = 0; k < result.size(); ++k) result[k] += weight*controls[c*p.entries+k];
    }
    return result;
}

int main() {
    try {
        std::mt19937 generator(4291);
        std::uniform_real_distribution<double> random(0., 1.);
        double maximum_error = 0;
        for (std::size_t dimension = 1; dimension <= 8; ++dimension)
            for (const std::size_t size : {1, 3, 12}) {
                Polynomial parent(dimension + 1, size);
                for (auto &control : parent.controls) {
                    control.resize(size * size);
                    for (std::size_t j = 0; j < size; ++j)
                        for (std::size_t i = 0; i <= j; ++i) {
                            const auto value = std::complex<double>{2 * random(generator) - 1,
                                i == j ? 0. : 2 * random(generator) - 1};
                            control[i + j * size] = value;
                            control[j + i * size] = std::conj(value);
                        }
                }
                for (std::size_t left = 0; left <= dimension; ++left)
                    for (std::size_t right = left + 1; right <= dimension; ++right)
                        for (const auto replaced : {left, right}) {
                            const auto child = parent.bisected(left, right, replaced);
                            for (int trial = 0; trial < 5; ++trial) {
                                Weights weights(dimension + 1);
                                for (auto &value : weights) value = random(generator);
                                const auto total = std::accumulate(weights.begin(), weights.end(), 0.);
                                for (auto &value : weights) value /= total;
                                auto original = weights;
                                original[replaced] = 0;
                                original[left] += .5 * weights[replaced];
                                original[right] += .5 * weights[replaced];
                                const auto reference = parent.blossom(original, original);
                                const auto value = child.blossom(weights, weights);
                                for (std::size_t k = 0; k < value.size(); ++k)
                                    maximum_error = std::max(maximum_error,
                                        std::abs(reference[k] - value[k]));
                            }
                        }
            }
        expect(maximum_error < 2e-14, "bisection preserves the matrix polynomial in 1D through 8D");
        double residual_error = 0;
        for (std::size_t dimension = 1; dimension <= 4; ++dimension) {
            ResidualMatrices parent(dimension+1, 3);
            for (auto *controls : {&parent.upper, &parent.lower})
                for (auto &entry : *controls) entry = {random(generator), random(generator)};
            for (std::size_t left = 0; left <= dimension; ++left)
                for (std::size_t right = left+1; right <= dimension; ++right)
                    for (const auto replaced : {left, right}) {
                        const auto child = parent.bisected(left, right, replaced);
                        for (std::size_t trial = 0; trial < 5; ++trial) {
                            Weights weights(dimension+1);
                            for (auto &w : weights) w = random(generator);
                            const auto total = std::accumulate(weights.begin(), weights.end(), 0.);
                            for (auto &w : weights) w /= total;
                            auto original = weights;
                            original[replaced] = 0;
                            original[left] += weights[replaced]/2;
                            original[right] += weights[replaced]/2;
                            for (bool occupied : {false, true}) {
                                const auto a = evaluate_residual(parent, original, occupied);
                                const auto b = evaluate_residual(child, weights, occupied);
                                for (std::size_t k = 0; k < a.size(); ++k)
                                    residual_error = std::max(residual_error,std::abs(a[k]-b[k]));
                            }
                        }
                    }
        }
        expect(residual_error < 2e-14, "degree-six midpoint restriction in dimensions 1 through 4");
        double rotation_error = 0;
        for (std::size_t size : {1, 2, 3, 8}) {
            Matrix basis(size*size);
            for (std::size_t j=0;j<size;++j)
                for (std::size_t i=0;i<=j;++i) {
                    basis[i+j*size] = {random(generator), i==j ? 0. : random(generator)};
                    basis[j+i*size] = std::conj(basis[i+j*size]);
                }
            std::vector<double> eigenvalues;
            fermisimplex::linalg::diagonalize_hermitian_in_place(basis,eigenvalues,size,true,"rotation test");
            ResidualMatrices matrices(3,size);
            for (auto &entry : matrices.upper) entry={random(generator),random(generator)};
            Matrix scratch;
            const auto batched=rotate_controls(matrices.upper,basis,size,scratch);
            for (std::size_t c=0;c<matrices.layout->indices.size();++c) {
                const auto first=matrices.upper.begin()+c*size*size;
                const auto reference=rotate(Matrix(first,first+size*size),basis,size);
                for (std::size_t k=0;k<size*size;++k)
                    rotation_error=std::max(rotation_error,std::abs(reference[k]-batched[c*size*size+k]));
            }
        }
        expect(rotation_error < 2e-14,"batched control rotation agrees with direct basis transformation");
        double gram_error=0;
        for (std::size_t dimension=1;dimension<=4;++dimension) {
            ResidualMatrices matrices(dimension+1,3);
            std::vector<Matrix> controls(matrices.layout->residual_indices.size(),Matrix(5*3));
            for (auto &c:controls)
                for (auto &entry:c) entry={random(generator),random(generator)};
            add_gram_controls(controls,5,3,*matrices.layout,matrices.upper);
            for (std::size_t trial=0;trial<10;++trial) {
                Weights weights(dimension+1);
                for(auto &w:weights) w=random(generator);
                const auto total=std::accumulate(weights.begin(),weights.end(),0.);
                for(auto &w:weights) w/=total;
                Matrix f(15);
                for(std::size_t c=0;c<controls.size();++c) {
                    double weight=6;
                    for(std::size_t i=0;i<weights.size();++i)
                        for(std::size_t k=1;k<=matrices.layout->residual_indices[c][i];++k)
                            weight*=weights[i]/k;
                    for(std::size_t k=0;k<f.size();++k) f[k]+=weight*controls[c][k];
                }
                const auto result=evaluate_residual(matrices,weights,true);
                for(std::size_t col=0;col<3;++col)
                    for(std::size_t row=0;row<3;++row) {
                        std::complex<double> exact{};
                        for(std::size_t k=0;k<5;++k) exact+=std::conj(f[k+row*5])*f[k+col*5];
                        gram_error=std::max(gram_error,std::abs(exact-result[row+col*3]));
                    }
            }
        }
        expect(gram_error < 2e-14,"Gram polynomial equals pointwise F adjoint F in dimensions 1 through 4");
        std::cout << "maximum Gram polynomial error=" << gram_error << '\n';
        std::cout << "maximum batched rotation error=" << rotation_error << '\n';
        std::cout << "maximum residual restriction error=" << residual_error << '\n';
        std::cout << "maximum bisection error=" << maximum_error << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
