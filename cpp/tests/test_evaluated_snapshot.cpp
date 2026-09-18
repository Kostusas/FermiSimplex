#include "test_helpers.h"

#include <algorithm>
#include <set>

namespace fs = fermisimplex;
namespace core = adaptivesimplex::core;
using namespace fermisimplex::test;

namespace {
class UnevaluableModel final : public fs::HamiltonianModel {
public:
    std::size_t ndim() const noexcept override { return 1; }
    std::size_t ndof() const noexcept override { return 1; }
    std::vector<std::complex<double>> evaluate(std::span<const double>) const override {
        throw std::runtime_error("snapshot must not evaluate the model");
    }
};
}  // namespace

int main() {
    fs::SpectralMesh mesh(std::make_shared<UnevaluableModel>(), 1e-14, 0);
    auto &geometry = mesh.geometry();
    geometry.preview_active(0, 3);
    expect_runtime_error([&] { mesh.evaluated_snapshot(); }, "no complete evaluated covering",
                         "unevaluated mesh export");
    auto &cache = mesh.eigensystems();
    core::VertexId extra_vertex = 0;
    for (core::VertexId id = 0; id < geometry.vertices().size(); ++id) {
        const auto x = geometry.vertices().dyadic_vertex(id).to_point()[0];
        if (x == 0.0 || x == 0.125 || x == 0.5 || x == 0.75 || x == 1.0) {
            cache.insert(id, fs::Eigensystem{{x}, {{1.0, 0.0}}});
        }
        if (x == 0.125) {
            extra_vertex = id;
        }
    }
    const auto simplex_count = geometry.simplices().size();
    const auto vertex_count = geometry.vertices().size();
    const auto snapshot = mesh.evaluated_snapshot();
    const auto repeated = mesh.evaluated_snapshot(false);
    expect_eq(snapshot.vertex_ids.size(), 5, "all cached vertices exported");
    expect_eq(snapshot.simplex_ids.size(), 3, "mixed-depth complete partition");
    expect_eq(snapshot.active_vertices, 2, "active vertex count");
    expect_eq(snapshot.active_simplices, 1, "active cell count");
    expect(!repeated.eigenvectors, "optional vectors omitted");
    expect(snapshot.simplices == repeated.simplices, "repeatable connectivity");
    expect(snapshot.eigenvalues == repeated.eigenvalues, "repeatable spectra");
    expect_eq(geometry.simplices().size(), simplex_count, "unchanged tree");
    expect_eq(geometry.vertices().size(), vertex_count, "unchanged vertices");
    expect_eq(cache.size(), 5, "unchanged cache");
    expect_eq(geometry.simplices().n_active(), 1, "unchanged active partition");
    std::set<std::size_t> used(snapshot.simplices.begin(), snapshot.simplices.end());
    expect_eq(used.size(), 4, "extra cached point is not a cell vertex");
    const auto extra_row = static_cast<std::size_t>(
        std::find(snapshot.vertex_ids.begin(), snapshot.vertex_ids.end(), extra_vertex) -
        snapshot.vertex_ids.begin());
    expect(!used.contains(extra_row), "incomplete preview point retained separately");
    expect_near(snapshot.points[extra_row], 0.125, 0.0, "extra cached coordinate");
    for (std::size_t row = 0; row < snapshot.vertex_ids.size(); ++row) {
        expect_near(snapshot.eigenvalues[row], snapshot.points[row], 0.0, "cached value unchanged");
        expect(snapshot.eigenvectors->at(row) == std::complex<double>(1.0, 0.0),
               "cached vector unchanged");
    }
    const std::vector<double> expected_endpoints{0.0, 0.5, 0.75, 1.0};
    std::set<double> endpoints;
    double total_volume = 0.0;
    for (std::size_t cell = 0; cell < snapshot.simplex_ids.size(); ++cell) {
        const auto a = snapshot.points[snapshot.simplices[cell * 2]];
        const auto b = snapshot.points[snapshot.simplices[cell * 2 + 1]];
        endpoints.insert(a);
        endpoints.insert(b);
        expect_near(std::abs(b - a), snapshot.volumes[cell], 0.0, "cell volume");
        expect_eq(snapshot.active_ancestor_ids[cell], 0, "active ancestor");
        total_volume += snapshot.volumes[cell];
    }
    expect(std::vector<double>(endpoints.begin(), endpoints.end()) == expected_endpoints,
           "complete partition endpoints");
    expect_near(total_volume, 1.0, 1e-12, "exact domain volume");
    cache.clear();
    expect_eq(snapshot.vertex_ids.size(), 5, "snapshot owns its storage");
    expect_near(snapshot.eigenvalues[extra_row], 0.125, 0.0, "snapshot survives cache reset");
    return 0;
}
