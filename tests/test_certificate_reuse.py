"""Partial proofs survive mu changes without retaining reduced matrices."""

import numpy as np
import pytest

from fermisimplex import SpectralMesh


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_reused_polynomial_charge_against_exact_crossing(dimension, degree, scale):
    rng = np.random.default_rng(29)
    basis, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))

    def energy(x):
        return x - 0.37 + 0.2 * x**degree

    def matrix(x):
        levels = scale * np.array([-3, -2, energy(x), 2, 4])
        return (basis * levels) @ basis.conj().T

    def one(x):
        return matrix(x)

    def two(x, y):
        return matrix(x)

    def three(x, y, z):
        return matrix(x)

    mesh = SpectralMesh({1: one, 2: two, 3: three}[dimension], root_level=1)
    first = mesh.integrate_charge(mu=0, target_error=5, max_refinements=0)
    assert first.error_stats.certificate_builds > 0
    for relative_mu in (0.01, -0.01, 0.0):
        result = mesh.integrate_charge(
            mu=relative_mu * scale, target_error=5, max_refinements=0
        )
        assert result.error_stats.certificate_reuses > 0
        assert (
            result.error_stats.hamiltonian_evaluations
            < first.error_stats.hamiltonian_evaluations
        )
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if energy(mid) < relative_mu:
                lo = mid
            else:
                hi = mid
        # The other coordinates integrate to one; two bands are fully occupied.
        exact = 2 + (lo + hi) / 2
        assert abs(result.value - exact) <= result.stopping_error + 1e-12
        enclosures = mesh.occupation_enclosures(mu=relative_mu * scale, depth=2)
        assert sum(e.charge_lower for e in enclosures) <= exact + 1e-12
        assert sum(e.charge_upper for e in enclosures) >= exact - 1e-12
        # Every simplex with a vertex sign change must remain unresolved.
        for cell, enclosure in zip(mesh.simplices, enclosures, strict=True):
            values = energy(mesh.points[cell, 0]) - relative_mu
            if values.min() < 0 < values.max():
                assert not enclosure.fixed_occupation
