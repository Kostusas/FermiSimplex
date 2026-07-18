#pragma once

#include "certification/linalg/matrix.h"

#include <lineartetrahedron/certification.h>
#include <lineartetrahedron/hamiltonian.h>
#include <lineartetrahedron/spectral_mesh.h>

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lineartetrahedron::test {

using Complex = std::complex<double>;
namespace cert_detail = certification::detail;
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

template <class Function>
void expect_runtime_error(
    Function &&function,
    const std::string &message_fragment,
    const std::string &message
) {
    try {
        std::forward<Function>(function)();
    } catch (const std::runtime_error &error) {
        expect(
            std::string(error.what()).find(message_fragment) != std::string::npos,
            message + ": unexpected error: " + error.what()
        );
        return;
    }
    throw std::runtime_error(message + ": expected a runtime error");
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

inline Eigensystem diagonal_spectra(std::initializer_list<double> eigenvalues) {
    const auto values = std::vector<double>(eigenvalues);
    std::vector<Complex> eigenvectors(values.size() * values.size(), Complex{0.0, 0.0});
    for (size_t index = 0; index < values.size(); ++index) {
        eigenvectors[cert_detail::column_major_index(index, index, values.size())] =
            Complex{1.0, 0.0};
    }
    return Eigensystem{
        .eigenvalues = values,
        .eigenvectors = eigenvectors,
    };
}

inline certification::SimplexCertificate certify_direct(
    const std::vector<Eigensystem> &spectra,
    double mu = 0.0,
    double linearization_error_bound = 0.0,
    double tolerance = certification::kDefaultTolerance
) {
    std::vector<std::span<const double>> eigenvalues;
    std::vector<std::span<const Complex>> eigenvectors;
    eigenvalues.reserve(spectra.size());
    eigenvectors.reserve(spectra.size());
    for (const auto &entry : spectra) {
        eigenvalues.emplace_back(entry.eigenvalues);
        eigenvectors.emplace_back(entry.eigenvectors);
    }
    return certification::certify_simplex(eigenvalues, eigenvectors, mu,
                                           linearization_error_bound, tolerance);
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
    const auto phase =
        2.0 * std::numbers::pi_v<double> * static_cast<double>(winding) * reduced_k;
    const auto c = std::cos(phase);
    const auto s = std::sin(phase);
    return hermitian_2x2(c, s, -c);
}

inline std::shared_ptr<TightBindingModel> winding_model(int winding) {
    const auto sigma_z = diagonal_matrix({1.0, -1.0});
    const auto sigma_x = hermitian_2x2(0.0, 1.0, 0.0);

    auto forward = HoppingTerm{.lattice_vector = {winding}};
    auto backward = HoppingTerm{.lattice_vector = {-winding}};
    forward.matrix.reserve(4);
    backward.matrix.reserve(4);
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        forward.matrix.push_back(
            0.5 * sigma_z[index] + Complex{0.0, 0.5} * sigma_x[index]
        );
    }
    for (size_t index = 0; index < sigma_z.size(); ++index) {
        backward.matrix.push_back(
            0.5 * sigma_z[index] - Complex{0.0, 0.5} * sigma_x[index]
        );
    }
    return std::make_shared<TightBindingModel>(
        std::vector<HoppingTerm>{std::move(forward), std::move(backward)}
    );
}

inline double winding_curvature_bound(int winding) {
    return 8.0 * std::numbers::pi_v<double> *
           std::numbers::pi_v<double> *
           static_cast<double>(winding * winding);
}

inline std::shared_ptr<TightBindingModel> constant_insulator() {
    return std::make_shared<TightBindingModel>(
        std::vector<HoppingTerm>{
            {.lattice_vector = {0}, .matrix = diagonal_matrix({-1.0, 1.0})},
        }
    );
}

inline core::SimplexId first_active_simplex(const core::Geometry &geometry) {
    const auto active = geometry.simplices().active_simplices();
    return *active.begin();
}

inline void fill_vertex_cache(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    SpectralMesh &mesh,
    core::VertexCache<Eigensystem> &cache
) {
    for (const auto vertex_id : geometry.simplices().simplex(simplex_id).vertex_ids) {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        cache.insert(
            vertex_id,
            mesh.spectrum(
                std::span<const double>(point.data(), point.size())
            )
        );
    }
}

}  // namespace lineartetrahedron::test
