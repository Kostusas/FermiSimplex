#include "fermi_surface/simplex_classification.h"
#include "occupation/enclosure.h"
#include "test_helpers.h"

#include <fermisimplex/integration.h>
#include <functional>
#include <iostream>
#include <numeric>

using namespace fermisimplex;
using namespace fermisimplex::test;
using namespace fermisimplex::occupation_detail;

class CountingModel final : public HamiltonianModel {
public:
    CountingModel(std::size_t dimension, std::size_t bands,
                  std::function<double(std::span<const double>)> energy)
        : dimension_(dimension), bands_(bands), energy_(std::move(energy)) {}
    std::size_t ndim() const noexcept override { return dimension_; }
    std::size_t ndof() const noexcept override { return bands_; }
    std::vector<Complex> evaluate(std::span<const double> point) const override {
        ++calls;
        std::vector<Complex> matrix(bands_ * bands_);
        const auto energy = energy_(point);
        for (std::size_t i = 0; i < bands_; ++i)
            matrix[i + i * bands_] = energy * (1. + double(i) / bands_);
        return matrix;
    }
    mutable std::size_t calls = 0;

private:
    std::size_t dimension_, bands_;
    std::function<double(std::span<const double>)> energy_;
};

int main() {
    try {
        for (std::size_t dimension = 1; dimension <= 5; ++dimension)
            for (const auto bands : {1, 12}) {
                auto model = std::make_shared<CountingModel>(dimension, bands,
                    [](std::span<const double> p) {
                        return std::accumulate(p.begin(), p.end(), 0.) / p.size() - .37;
                    });
                SpectralMesh mesh(model, 1e-14, 0);
                estimate_charge_on_current_mesh(mesh, 0);
                const auto active = mesh.geometry().simplices().active_simplices();
                const std::vector<core::SimplexId> ids(active.begin(), active.end());
                model->calls = 0;
                const auto refining = fermi_surface_detail::classify_frontier(mesh, ids, 0, 0, 0);
                const auto terminal = fermi_surface_detail::classify_frontier(mesh, ids, 0, 10, 0);
                expect(refining.refine == ids, "strict crossings all refine");
                expect(terminal.terminal_surface == ids, "strict crossings all reach extraction");
                expect_eq(terminal.terminal_visible, ids.size(), "crossings are visible");
                expect_eq(model->calls, 0, "cached crossing witnesses need no model samples");
            }

        // The same vertex-level tolerance can mean a small gap, an exact
        // contact, or no visible sign change around a hidden quartic pocket.
        // None supplies a strict crossing witness: each still needs the model.
        for (int scenario = 0; scenario < 3; ++scenario) {
            auto model = std::make_shared<CountingModel>(1, 1,
                [scenario](std::span<const double> p) {
                    const auto x = p[0];
                    if (scenario == 0) return .0005 + .1 * x;
                    if (scenario == 1) return x;
                    return (x - .2) * (x - .3) * (1 + x * x);
                });
            SpectralMesh mesh(model, 1e-3, 0);
            estimate_charge_on_current_mesh(mesh, 0);
            const auto id = first_active_simplex(mesh.geometry());
            expect(vertex_occupation(mesh, id, 0) != VertexOccupation::crossing,
                   "ambiguous or uniform vertices do not prove a crossing");
            model->calls = 0;
            const auto result = fermi_surface_detail::classify_frontier(mesh, {id}, 0, 2, 0);
            expect(model->calls > 0, "ambiguous cells retain polynomial probes");
            expect_eq(result.terminal_surface.size(), scenario == 0 ? 0 : 1,
                      "gap is removed but contact and hidden quartic pocket remain");
            expect_eq(result.terminal_visible, scenario == 1 ? 1 : 0,
                      "only the exact contact is visible at vertices");
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
