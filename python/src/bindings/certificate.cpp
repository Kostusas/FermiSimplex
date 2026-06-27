#include "arrays.h"
#include "bindings.h"

#include "certificate/simplex_certificate.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {

namespace certificate = simplex_certificate;

namespace {

const char *certificate_status_name(certificate::SimplexCertificateStatus status) {
    switch (status) {
    case certificate::SimplexCertificateStatus::CertifiedGapped:
        return "certified_gapped";
    case certificate::SimplexCertificateStatus::VisibleGapless:
        return "visible_gapless";
    case certificate::SimplexCertificateStatus::Inconclusive:
        return "inconclusive";
    }
    return "unknown";
}

void validate_spectra_arrays(EigenvalueArray eigenvalues, EigenvectorArray eigenvectors) {
    const size_t nvertices = eigenvalues.shape(0);
    const size_t ndof = eigenvalues.shape(1);
    if (nvertices < 2) {
        throw std::runtime_error("certify_simplex: at least two vertices are required");
    }
    if (
        eigenvectors.shape(0) != nvertices ||
        eigenvectors.shape(1) != ndof ||
        eigenvectors.shape(2) != ndof
    ) {
        throw std::runtime_error(
            "certify_simplex: eigenvectors must have shape (nvertices, ndof, ndof)"
        );
    }
}

std::vector<std::complex<double>> column_major_eigenvectors(EigenvectorArray eigenvectors) {
    const size_t nvertices = eigenvectors.shape(0);
    const size_t ndof = eigenvectors.shape(1);
    std::vector<std::complex<double>> storage(nvertices * ndof * ndof);
    for (size_t vertex = 0; vertex < nvertices; ++vertex) {
        for (size_t band = 0; band < ndof; ++band) {
            for (size_t row = 0; row < ndof; ++row) {
                storage[vertex * ndof * ndof + band * ndof + row] =
                    eigenvectors.data()[(vertex * ndof + row) * ndof + band];
            }
        }
    }
    return storage;
}

certificate::SimplexCertificate certify_simplex(
    EigenvalueArray eigenvalues,
    EigenvectorArray eigenvectors,
    double mu,
    double margin,
    double tol,
    bool estimate_occupation_bounds
) {
    validate_spectra_arrays(eigenvalues, eigenvectors);
    const size_t nvertices = eigenvalues.shape(0);
    const size_t ndof = eigenvalues.shape(1);
    const auto eigenvector_storage = column_major_eigenvectors(eigenvectors);

    std::vector<std::span<const double>> eigenvalue_spans;
    std::vector<std::span<const std::complex<double>>> eigenvector_spans;
    eigenvalue_spans.reserve(nvertices);
    eigenvector_spans.reserve(nvertices);
    for (size_t vertex = 0; vertex < nvertices; ++vertex) {
        eigenvalue_spans.push_back(
            std::span<const double>(eigenvalues.data() + vertex * ndof, ndof)
        );
        eigenvector_spans.push_back(
            std::span<const std::complex<double>>(
                eigenvector_storage.data() + vertex * ndof * ndof,
                ndof * ndof
            )
        );
    }

    return certificate::certify_simplex(
        std::span<const std::span<const double>>(
            eigenvalue_spans.data(),
            eigenvalue_spans.size()
        ),
        std::span<const std::span<const std::complex<double>>>(
            eigenvector_spans.data(),
            eigenvector_spans.size()
        ),
        mu,
        margin,
        tol,
        estimate_occupation_bounds
    );
}

}  // namespace

void bind_certificate(nb::module_ &m) {
    using namespace nb::literals;

    nb::class_<certificate::OccupationBounds>(m, "OccupationBounds")
        .def_prop_ro("lower", [](const certificate::OccupationBounds &self) { return self.lower; })
        .def_prop_ro("upper", [](const certificate::OccupationBounds &self) { return self.upper; });

    nb::class_<certificate::MuInterval>(m, "MuInterval")
        .def_prop_ro("lower", [](const certificate::MuInterval &self) { return self.lower; })
        .def_prop_ro("upper", [](const certificate::MuInterval &self) { return self.upper; });

    nb::class_<certificate::SimplexCertificate>(m, "SimplexCertificate")
        .def_prop_ro(
            "status",
            [](const certificate::SimplexCertificate &self) {
                return certificate_status_name(self.status);
            }
        )
        .def_prop_ro(
            "occupation_bounds",
            [](const certificate::SimplexCertificate &self) {
                return self.occupation_bounds;
            }
        )
        .def_prop_ro(
            "mu_interval",
            [](const certificate::SimplexCertificate &self) {
                return self.mu_interval;
            }
        )
        .def_prop_ro(
            "has_mu_interval",
            [](const certificate::SimplexCertificate &self) {
                return certificate::has_mu_interval(self);
            }
        )
        .def(
            "reusable_at",
            [](const certificate::SimplexCertificate &self, double mu) {
                return certificate::reusable_at(self, mu);
            },
            "mu"_a
        )
        .def_prop_ro(
            "occupation_width",
            [](const certificate::SimplexCertificate &self) {
                return certificate::occupation_width(self);
            }
        );

    m.def(
        "certify_simplex",
        &certify_simplex,
        "eigenvalues"_a,
        "eigenvectors"_a,
        "mu"_a = 0.0,
        "margin"_a = 0.0,
        "tol"_a = certificate::kDefaultTolerance,
        "estimate_occupation_bounds"_a = true
    );
}

}  // namespace lineartetrahedron::bindings
