#include <fermisimplex/fermisimplex.h>

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

class ConstantModel final : public fermisimplex::HamiltonianModel {
public:
    std::size_t ndim() const noexcept override { return 1; }
    std::size_t ndof() const noexcept override { return 1; }

    std::vector<std::complex<double>> evaluate(
        std::span<const double>
    ) const override {
        return {{2.0, 0.0}};
    }
};

}  // namespace

int main() {
    auto model = std::make_shared<ConstantModel>();
    fermisimplex::SpectralMesh mesh(std::move(model), 1e-14, 0);

    const std::array<double, 1> point{0.25};
    const auto eigensystem = mesh.spectrum(point);
    const auto charge = fermisimplex::estimate_charge_on_current_mesh(
        mesh, 0.0, 1.0, 1
    );
    return eigensystem.eigenvalues.size() == 1 &&
                   eigensystem.eigenvalues.front() == 2.0 &&
                   charge.value == 0.0 &&
                   charge.stopping_error == 0.0 &&
                   charge.error_stats.root_simplices > 0
               ? 0
               : 1;
}
