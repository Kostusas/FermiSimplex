#pragma once

#include "certificate/linalg/matrix.h"
#include "core/tight_binding.h"
#include "core/types.h"
#include "core/vertex_spectra.h"
#include "integration/workspace.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lineartetrahedron::test {

using Complex = std::complex<double>;
namespace cert_detail = simplex_certificate::detail;
namespace core = adaptivesimplex::core;

constexpr double kTol = 1e-12;

inline void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void expect_eq(size_t actual, size_t expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

inline void expect_near(double actual, double expected, double tol, const std::string &message) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

inline std::vector<Complex> diagonal_matrix(std::initializer_list<double> diagonal) {
    const auto size = diagonal.size();
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    auto index = size_t{0};
    for (const auto value : diagonal) {
        result[cert_detail::column_major_index(index, index, size)] = Complex{value, 0.0};
        ++index;
    }
    return result;
}

inline std::vector<Complex> hermitian_2x2(double a, double b, double c) {
    return {
        Complex{a, 0.0},
        Complex{b, 0.0},
        Complex{b, 0.0},
        Complex{c, 0.0},
    };
}

inline std::vector<Complex> winding_hamiltonian(int winding, double reduced_k) {
    const auto phase = 2.0 * kPi * static_cast<double>(winding) * reduced_k;
    const auto c = std::cos(phase);
    const auto s = std::sin(phase);
    return hermitian_2x2(c, s, -c);
}

inline std::shared_ptr<TightBindingModel> winding_model(int winding) {
    std::vector<std::int64_t> keys{
        static_cast<std::int64_t>(winding),
        static_cast<std::int64_t>(-winding),
    };
    const auto sigma_z = diagonal_matrix({1.0, -1.0});
    const auto sigma_x = hermitian_2x2(0.0, 1.0, 0.0);

    std::vector<Complex> matrices;
    matrices.reserve(2 * 4);
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        matrices.push_back(0.5 * sigma_z[index] + Complex{0.0, 0.5} * sigma_x[index]);
    }
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        matrices.push_back(0.5 * sigma_z[index] - Complex{0.0, 0.5} * sigma_x[index]);
    }
    return std::make_shared<TightBindingModel>(
        1,
        2,
        std::move(keys),
        std::move(matrices)
    );
}

inline core::SimplexId first_active_simplex(const core::Geometry &geometry) {
    const auto active = geometry.simplices().active_simplices();
    return *active.begin();
}

inline void fill_vertex_cache(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    VertexSpectraEvaluator &evaluator,
    core::VertexCache<VertexSpectra> &cache
) {
    for (const auto vertex_id : geometry.simplices().simplex(simplex_id).vertex_ids) {
        cache.insert(vertex_id, evaluator.evaluate(geometry, vertex_id));
    }
}

inline void fill_workspace_cache(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    IntegrationWorkspace &workspace
) {
    for (const auto vertex_id : geometry.simplices().simplex(simplex_id).vertex_ids) {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        workspace.cache().insert(
            vertex_id,
            workspace.evaluate_vertex(std::span<const double>(point.data(), point.size()))
        );
    }
}

}  // namespace lineartetrahedron::test
