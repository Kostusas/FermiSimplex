from __future__ import annotations

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

    surface = SpectralMesh(hamiltonian).fermi_surface(
        mu=0.0,
        min_feature_size=0.2,
        curvature_bound=0.0,
    )

    assert version("FermiSimplex")
    assert surface.completed
    assert surface.coverage_certified
    assert set(surface.cell_bands.tolist()) == {0, 1}


def test_bundled_openblas_notice_is_installed():
    package_dir = Path(fermisimplex.__file__).resolve().parent
    notice = package_dir / "licenses" / "scipy-openblas32.txt"
    assert notice.is_file()
    assert "OpenBLAS" in notice.read_text()
