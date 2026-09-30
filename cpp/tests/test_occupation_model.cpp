#include "occupation/model.h"
#include "occupation/enclosure.h"
#include "test_helpers.h"

#include <fermisimplex/integration.h>
#include <iostream>
#include <random>

using namespace fermisimplex;
using namespace fermisimplex::test;
using namespace fermisimplex::occupation_detail;

class CoupledModel final : public HamiltonianModel {
public:
    explicit CoupledModel(double h) : h_(h) {}
    std::size_t ndim() const noexcept override { return 1; }
    std::size_t ndof() const noexcept override { return 3; }
    Matrix evaluate(std::span<const double> point) const override {
        const auto x = h_ * point[0];
        return {-2 - .1*x, .2*x, Complex{0, -.1*x},
                .2*x, x - .4*h_, .3*x,
                Complex{0, .1*x}, .3*x, 3 + .2*x};
    }
private:
    double h_;
};

void check_occupation_queries() {
    auto mesh = SpectralMesh(std::make_shared<CoupledModel>(.2), 1e-14, 0);
    estimate_charge_on_current_mesh(mesh, 0);
    const auto id = first_active_simplex(mesh.geometry());
    for (const auto mu : {-5., 0., 5.})
        for (std::uint32_t depth = 0; depth <= 3; ++depth) {
            ChargeErrorStats charge_stats, sign_stats;
            const auto enclosure = enclose_occupation(mesh, id, mu, depth, charge_stats);
            const auto fixed = fixed_occupation(mesh, id, mu, depth, sign_stats);
            // The affine model crosses zero on the interval and has spectrum
            // strictly between -5 and 5 everywhere (also by its row bounds).
            expect(fixed == (mu != 0), "sign query agrees with the known spectrum");
            expect(fixed == enclosure.fixed_occupation(), "consumers agree on occupation");
            expect_eq(sign_stats.hamiltonian_evaluations, charge_stats.hamiltonian_evaluations,
                      "consumers use the same Hamiltonian samples");
            expect_eq(sign_stats.micro_simplices, charge_stats.micro_simplices,
                      "consumers traverse the same polynomial cells");
        }
}

void check_safe_sectors() {
    std::mt19937 generator(123);
    std::normal_distribution<double> normal;
    double worst_margin_excess = 0;
    for (std::size_t trial = 0; trial < 100; ++trial) {
        const std::size_t n = 8;
        const double allowance = .07;
        const auto shift = (static_cast<double>(trial % 5) - 2) * 2;
        const auto coupling = static_cast<double>((trial / 5) % 5) * .25;
        Polynomial polynomial{3, n};
        for (auto &control : polynomial.controls) {
            control.resize(n * n);
            for (std::size_t col = 0; col < n; ++col) {
                control[col + col * n] = (static_cast<double>(col) - 3.5) * .6 + shift;
                for (std::size_t row = 0; row < col; ++row) {
                    const Complex value = coupling * Complex{normal(generator), normal(generator)};
                    control[row + col * n] = value;
                    control[col + row * n] = std::conj(value);
                }
            }
        }
        const auto exact_margin = [&](std::size_t count, bool negative) {
            auto margin = std::numeric_limits<double>::infinity();
            if (count == 0) return margin;
            std::vector<std::size_t> sector(count);
            for (std::size_t i = 0; i < count; ++i)
                sector[i] = negative ? i : n - count + i;
            for (const auto &control : polynomial.controls) {
                auto matrix = block(control, n, sector, sector);
                if (negative) for (auto &value : matrix) value = -value;
                std::vector<double> eigenvalues;
                linalg::diagonalize_hermitian_in_place(matrix, eigenvalues,
                    count, false, "reference safe sector");
                margin = std::min(margin, eigenvalues.front() - allowance);
            }
            return margin;
        };
        const auto exact_count = [&](bool negative, std::size_t maximum) {
            std::size_t count = 0;
            while (count < maximum && exact_margin(count + 1, negative) > 0) ++count;
            return count;
        };
        const auto negative = exact_count(true, n);
        const auto positive = exact_count(false, n - negative);
        ChargeErrorStats stats;
        const auto result = sign_sectors(polynomial, allowance, stats);
        expect_eq(result.negative, negative, "largest negative prefix agrees with eigenvalues");
        expect_eq(result.positive, positive, "largest positive suffix agrees with eigenvalues");
        const auto reference = std::min(exact_margin(negative, true),
                                       exact_margin(positive, false));
        worst_margin_excess = std::max(worst_margin_excess, result.gap - reference);
        expect((negative + positive == 0 || result.gap > 0) &&
                   result.gap <= reference + 1e-12,
               "safe margin is positive and does not exceed the spectral reference");
        expect(stats.norm_eigensystems <= 2 * polynomial.controls.size(),
               "at most one margin eigensystem per control and sign");
    }
    std::cout << "safe-sector maximum margin excess=" << worst_margin_excess << '\n';
}

int main() {
    try {
        check_occupation_queries();
        check_safe_sectors();
        double previous_error = 0, previous_allowance = 0;
        for (const auto h : {.2, .1, .05, .025}) {
            auto mesh = SpectralMesh(std::make_shared<CoupledModel>(h), 1e-14, 0);
            estimate_charge_on_current_mesh(mesh, 0);
            const auto id = first_active_simplex(mesh.geometry());
            ChargeErrorStats stats;
            const auto model = build_model(mesh, id, 0, stats, 0.);
            expect_eq(model.polynomial.size, 1, "one active state");
            expect_eq(model.safe_occupation, 1, "one safe occupied state");
            expect(model.delta > 1, "indefinite safe block has a uniform gap");
            const auto &simplex = mesh.geometry().simplices().simplex(id);
            const auto a = mesh.geometry().vertices().dyadic_vertex(simplex.vertex_ids[0]).to_point()[0];
            const auto b = mesh.geometry().vertices().dyadic_vertex(simplex.vertex_ids[1]).to_point()[0];
            double error = 0;
            for (int i = 0; i <= 1000; ++i) {
                const auto t = i / 1000.;
                const auto k = a + (b - a) * t;
                const auto matrix = mesh.hamiltonian(Weights{k});
                const std::vector<std::size_t> safe{0, 2}, active{1};
                auto d = block(matrix, 3, safe, safe);
                const auto coupling = block(matrix, 3, safe, active);
                auto solution = coupling;
                expect(linalg::solve_linear_system_in_place(d, solution, 2, 1,
                    "reference Schur solve"), "reference invertible");
                auto schur = matrix[4];
                for (std::size_t j = 0; j < 2; ++j)
                    schur -= std::conj(coupling[j]) * solution[j];
                const auto polynomial = model.polynomial.blossom(Weights{1-t, t}, Weights{1-t, t})[0];
                error = std::max(error, std::abs(schur - polynomial));
            }
            expect(error <= model.epsilon, "Schur matrix error is enclosed");
            if (previous_error > 0) {
                expect(previous_error / error > 7, "actual error decreases cubically");
                expect(previous_allowance / model.epsilon > 7, "allowance decreases cubically");
            }
            std::cout << "h=" << h << " Schur error=" << error
                      << " allowance=" << model.epsilon << '\n';
            previous_error = error; previous_allowance = model.epsilon;
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
