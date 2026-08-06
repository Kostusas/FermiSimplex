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
        return np.diag(
            [coordinate_sum - 1.5, coordinate_sum - 0.5]
        ).astype(complex)

    mesh = SpectralMesh(hamiltonian)
    surface = mesh.fermi_surface(
        mu=0.0,
        min_feature_size=0.2,
        curvature_bound=0.0,
    )

    charge = mesh.estimate_charge_on_current_mesh(
        mu=0.0,
        target_error=1.0,
    )

    assert version("FermiSimplex")
    assert surface.completed
    assert surface.coverage_certified
    assert set(surface.cell_bands.tolist()) == {0, 1}
    assert np.isfinite(charge.value)
    assert np.isfinite(charge.stopping_error)
    assert charge.error_stats.root_simplices > 0
    assert charge.stats.simplex_visits == charge.stats.active_simplices


def test_bundled_openblas_notice_matches_platform_backend():
    package_dir = Path(fermisimplex.__file__).resolve().parent
    notice = package_dir / "licenses" / "scipy-openblas32.txt"

    if sys.platform == "darwin":
        assert not notice.exists()
        return

    assert notice.is_file()
    assert "OpenBLAS" in notice.read_text()
