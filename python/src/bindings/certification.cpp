#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/certification.h>

#include <nanobind/stl/optional.h>

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace fermisimplex::bindings {
namespace cert = certification;

namespace {

std::vector<std::complex<double>> column_major_vectors(
    EigenvectorArray eigenvectors
) {
    const auto vertex_count = eigenvectors.shape(0);
    const auto ndof = eigenvectors.shape(1);
    std::vector<std::complex<double>> result(vertex_count * ndof * ndof);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        for (std::size_t band = 0; band < ndof; ++band) {
            for (std::size_t row = 0; row < ndof; ++row) {
                result[(vertex * ndof + band) * ndof + row] =
                    eigenvectors.data()[(vertex * ndof + row) * ndof + band];
            }
        }
    }
    return result;
}

cert::SimplexCertificate certify(
    EigenvalueArray eigenvalues,
    EigenvectorArray eigenvectors,
    double linearization_error_bound,
    double mu,
    double tolerance
) {
    const auto vertex_count = eigenvalues.shape(0);
    const auto ndof = eigenvalues.shape(1);
    if (vertex_count < 2 || ndof == 0) {
        throw std::runtime_error("certify_simplex: spectra must be non-empty");
    }
    if (
        eigenvectors.shape(0) != vertex_count ||
        eigenvectors.shape(1) != ndof ||
        eigenvectors.shape(2) != ndof
    ) {
        throw std::runtime_error(
            "certify_simplex: eigenvectors must have shape (nvertices, ndof, ndof)"
        );
    }

    const auto vector_storage = column_major_vectors(eigenvectors);
    std::vector<std::span<const double>> value_spans;
    std::vector<std::span<const std::complex<double>>> vector_spans;
    value_spans.reserve(vertex_count);
    vector_spans.reserve(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        value_spans.emplace_back(eigenvalues.data() + vertex * ndof, ndof);
        vector_spans.emplace_back(
            vector_storage.data() + vertex * ndof * ndof,
            ndof * ndof
        );
    }

    return cert::certify_simplex(
        value_spans,
        vector_spans,
        mu,
        linearization_error_bound,
        tolerance
    );
}

}  // namespace

void bind_certification(nb::module_ &module) {
    using namespace nb::literals;

    nb::enum_<cert::SimplexCertificateStatus>(module, "CertificateStatus")
        .value(
            "CertifiedGapped",
            cert::SimplexCertificateStatus::CertifiedGapped
        )
        .value(
            "VisibleGapless",
            cert::SimplexCertificateStatus::VisibleGapless
        )
        .value("Inconclusive", cert::SimplexCertificateStatus::Inconclusive);

    nb::class_<cert::OccupationBounds>(module, "OccupationBounds")
        .def_ro("lower", &cert::OccupationBounds::lower)
        .def_ro("upper", &cert::OccupationBounds::upper);

    nb::class_<cert::MuInterval>(module, "MuInterval")
        .def_ro("lower", &cert::MuInterval::lower)
        .def_ro("upper", &cert::MuInterval::upper);

    nb::class_<cert::SimplexCertificate>(module, "SimplexCertificate")
        .def_ro("status", &cert::SimplexCertificate::status)
        .def_prop_ro(
            "occupation_bounds",
            [](const cert::SimplexCertificate &result) {
                return result.occupation_bounds;
            }
        )
        .def_prop_ro(
            "mu_interval",
            [](const cert::SimplexCertificate &result) {
                return result.mu_interval;
            }
        )
        .def_prop_ro(
            "occupation_width",
            [](const cert::SimplexCertificate &result) {
                return cert::occupation_width(result);
            }
        )
        .def(
            "occupation_bounds_valid_at",
            [](const cert::SimplexCertificate &result, double mu) {
                return cert::occupation_bounds_valid_at(result, mu);
            },
            "mu"_a
        );

    module.def(
        "certify_simplex",
        &certify,
        "eigenvalues"_a,
        "eigenvectors"_a,
        "linearization_error_bound"_a,
        "mu"_a = 0.0,
        "tolerance"_a = cert::kDefaultTolerance
    );
}

}  // namespace fermisimplex::bindings
