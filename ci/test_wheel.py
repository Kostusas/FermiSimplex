from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import fermisimplex
from fermisimplex import SpectralMesh


def test_installed_distribution_runs_a_native_surface_calculation():
    def hamiltonian(kx, ky):
        coordinate_sum = kx + ky
        return np.diag([coordinate_sum - 1.5, coordinate_sum - 0.5]).astype(complex)

    mesh = SpectralMesh(hamiltonian)
    surface = mesh.fermi_surface(
        mu=0.0,
        min_feature_size=0.2,
        curvature_bound=0.0,
    )

    charge = mesh.estimate_charge_on_current_mesh(mu=0.0)
    density = mesh.integrate_density_components(
        mu=0.0,
        lattice_vectors=[(0, 0)],
        components=[(0, 0, 0), (0, 1, 1)],
        target_error=1e-10,
        max_refinements=0,
    )

    assert version("FermiSimplex")
    assert surface.completed
    assert surface.coverage_certified
    assert set(surface.cell_bands.tolist()) == {0, 1}
    assert np.isfinite(charge.value)
    assert np.isfinite(charge.dcharge_dmu)
    assert density.values.shape == (2,)
    assert np.all(np.isfinite(density.values))


def test_bundled_openblas_notice_matches_platform_backend():
    package_dir = Path(fermisimplex.__file__).resolve().parent
    notice = package_dir / "licenses" / "scipy-openblas32.txt"

    if sys.platform in {"darwin", "emscripten"}:
        assert not notice.exists()
        return

    assert notice.is_file()
    assert "OpenBLAS" in notice.read_text()


def test_complex_projector_matches_exact_occupied_state():
    hamiltonian = np.array([[0.0, 1.0j], [-1.0j, 0.0]])
    mesh = SpectralMesh(lambda kx, ky: hamiltonian)
    mesh.fermi_surface(mu=0.0, min_feature_size=0.2, curvature_bound=0.0)
    density = mesh.integrate_density_components(
        mu=0.0,
        lattice_vectors=[(0, 0)],
        components=[(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],
        target_error=1e-12,
        max_refinements=0,
    )
    # H has eigenvalues +/-1, so the occupied projector is (I - H) / 2.
    expected = (np.eye(2) - hamiltonian) / 2
    np.testing.assert_allclose(density.values.reshape(2, 2), expected, atol=1e-12)
    np.testing.assert_allclose(mesh.estimate_charge_on_current_mesh(mu=0.0).value, 1.0, atol=1e-12)
