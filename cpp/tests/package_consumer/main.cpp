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
    return eigensystem.eigenvalues.size() == 1 &&
                   eigensystem.eigenvalues.front() == 2.0
               ? 0
               : 1;
}
