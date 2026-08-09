#pragma once

#include <fermisimplex/hamiltonian.h>
#include <fermisimplex/integration.h>

#include <complex>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace fermisimplex::integration_detail {

using SchurMatrix = std::vector<std::complex<double>>;

// Signals that a Schur transform cannot be constructed or evaluated with
// finite arithmetic. The charge-error estimator converts this into its
// conservative fallback path.
struct SchurFailure {};

// Non-owning adapter for the cached eigensystem used to anchor a reduction.
// Eigenvectors are a square column-major matrix whose columns correspond to
// eigenvalues.
struct SchurEigensystemView {
    std::span<const double> eigenvalues;
    std::span<const std::complex<double>> eigenvectors;
};

// Frozen active/safe-space data shared by every evaluation below one
// reduction. Dense evaluation uses safe_resolvent; projected tight-binding
// construction uses safe_basis and inverse_safe_eigenvalues instead.
struct SchurLayer {
    std::size_t parent_dimension = 0;
    std::size_t active_dimension = 0;
    SchurMatrix active_basis;
    SchurMatrix safe_resolvent;
    SchurMatrix safe_basis;
    std::vector<double> inverse_safe_eigenvalues;
    mutable SchurMatrix x_buffer;
    mutable SchurMatrix y_buffer;
};

SchurLayer make_schur_layer(
    SchurEigensystemView anchor,
    std::size_t active_begin,
    std::size_t active_end,
    bool materialize_resolvent
);

SchurMatrix apply_schur_layer(
    std::span<const std::complex<double>> matrix,
    const SchurLayer &layer,
    ChargeErrorStats &stats
);

struct ProjectedHoppingModel {
    std::size_t ndim = 0;
    std::size_t dimension = 0;
    std::vector<LatticeVector> lattice_vectors;
    SchurMatrix coefficients;

    SchurMatrix evaluate(std::span<const double> point) const;
};

std::optional<ProjectedHoppingModel> make_projected_hopping_model(
    const TightBindingModel &model,
    double mu,
    const SchurLayer &layer
);

}  // namespace fermisimplex::integration_detail
