#include "occupation/model.h"
#include "test_helpers.h"

#include <fermisimplex/integration.h>
#include <iostream>

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

int main() {
    try {
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
