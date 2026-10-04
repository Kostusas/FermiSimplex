"""Occupation and density references for weak and strong active-safe coupling."""

import numpy as np
import pytest

from fermisimplex import SpectralMesh


@pytest.mark.parametrize("coupling", [0.0, 1e-4, 0.1])
def test_coupled_crossing_charge_and_density(coupling):
    rng = np.random.default_rng(7461)
    basis = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]

    def active(x):
        return x - 0.37 + 0.1 * np.sin(2 * np.pi * x)

    def off_diagonal(x):
        return coupling * np.sin(np.pi * x)

    def hamiltonian(x):
        matrix = np.diag([-3.0, active(x), 1.0, 3.0]).astype(complex)
        matrix[1, 2] = 1j * off_diagonal(x)
        matrix[2, 1] = -1j * off_diagonal(x)
        return basis @ matrix @ basis.conj().T

    # The coupled 2x2 block changes occupation at a(x)-b(x)^2=0.
    # Its derivative is positive on [0,1] for every tested coupling.
    lower, upper = 0.0, 1.0
    for _ in range(60):
        middle = (lower + upper) / 2
        if active(middle) - off_diagonal(middle) ** 2 < 0:
            lower = middle
        else:
            upper = middle
    crossing = (lower + upper) / 2

    def reference_density(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        result = np.zeros((4, 4), complex)
        for left, right in ((0.0, crossing), (crossing, 1.0)):
            points = (left + right) / 2 + (right - left) / 2 * nodes
            energies, vectors = np.linalg.eigh([hamiltonian(x) for x in points])
            projectors = (
                vectors * (energies < 0)[:, None, :]
            ) @ vectors.conj().transpose(0, 2, 1)
            result += (right - left) / 2 * np.einsum("k,kij->ij", weights, projectors)
        return result

    reference = reference_density(64)
    reference_error = np.max(np.abs(reference - reference_density(96)))
    assert reference_error < 2e-12

    mesh = SpectralMesh(hamiltonian, root_level=2)
    charge = mesh.integrate_charge(mu=0, target_error=1e-4)
    density = mesh.integrate_density_matrix(
        mu=0, lattice_vectors=[(0,)], target_error=2e-5
    )
    assert charge.stats.target_reached and density.stats.target_reached
    assert abs(charge.value - (1 + crossing)) <= charge.stopping_error + 1e-12
    assert np.max(np.abs(density.matrices[0] - reference)) <= (
        charge.density_cut_error + density.stopping_error + reference_error
    )
