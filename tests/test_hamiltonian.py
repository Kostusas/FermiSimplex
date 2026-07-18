from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from lineartetrahedron import AdaptiveOptions, Hamiltonian, SpectralMesh, TightBinding

from .helpers import constant_insulator, qiwuzhang, tb_k_matrix


def test_tight_binding_evaluates_fourier_series():
    hoppings = qiwuzhang()
    model = TightBinding(hoppings)
    point = np.array([0.3, -0.7])

    assert model.ndim == 2
    assert model.ndof == 2
    assert not hasattr(model, "curvature_bound")
    assert np.allclose(model(point), tb_k_matrix(hoppings, point))


def test_tight_binding_requires_global_hermiticity():
    with pytest.raises(RuntimeError, match="opposite partner"):
        TightBinding({(1,): np.array([[0.5]], dtype=complex)})

    with pytest.raises(RuntimeError, match="adjoint"):
        TightBinding({(0,): np.array([[1.0j]], dtype=complex)})


def test_tight_binding_rejects_nonfinite_coefficients():
    with pytest.raises(RuntimeError, match="entries must be finite"):
        TightBinding({(0,): np.array([[np.nan]], dtype=complex)})


def test_callable_hamiltonian_uses_one_coordinate_vector_without_probing():
    calls: list[np.ndarray] = []

    def function(point: np.ndarray) -> np.ndarray:
        calls.append(point.copy())
        return np.diag([point[0], -point[0]])

    model = Hamiltonian(
        function,
        ndim=1,
        ndof=2,
    )

    assert calls == []
    assert np.allclose(model([0.25]), np.diag([0.25, -0.25]))
    assert len(calls) == 1
    assert calls[0].shape == (1,)


def test_callable_spectral_mesh_does_not_leak_at_interpreter_shutdown():
    script = """
import numpy as np

from lineartetrahedron import Hamiltonian, SpectralMesh

model = Hamiltonian(
    lambda point: np.array([[point[0]]], dtype=complex),
    ndim=1,
    ndof=1,
)
mesh = SpectralMesh(model)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "nanobind: leaked" not in completed.stderr


def test_hamiltonian_points_must_be_finite():
    model = Hamiltonian(
        lambda point: np.array([[point[0]]], dtype=complex),
        ndim=1,
        ndof=1,
    )

    with pytest.raises(ValueError, match="point coordinates must be finite"):
        model([np.nan])


def test_spectral_evaluation_rejects_nonhermitian_matrices():
    model = Hamiltonian(
        lambda point: np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        ndim=1,
        ndof=2,
    )
    mesh = SpectralMesh(model)

    with pytest.raises(RuntimeError, match="Hamiltonian must be Hermitian"):
        mesh.integrate_charge(
            0.0,
            AdaptiveOptions(target_error=1.0, max_refinements=0),
        )


def test_spectral_evaluation_rejects_nonfinite_matrix_entries():
    model = Hamiltonian(
        lambda point: np.array([[np.nan]], dtype=complex),
        ndim=1,
        ndof=1,
    )
    mesh = SpectralMesh(model)

    with pytest.raises(RuntimeError, match="Hamiltonian entries must be finite"):
        mesh.integrate_charge(
            0.0,
            AdaptiveOptions(target_error=1.0, max_refinements=0),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"ndim": 0, "ndof": 1}, "ndim"),
        ({"ndim": 1, "ndof": 0}, "ndof"),
    ),
)
def test_callable_hamiltonian_rejects_invalid_metadata(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Hamiltonian(lambda point: np.eye(1), **kwargs)


def test_spectral_mesh_requires_an_explicit_model():
    with pytest.raises(TypeError, match="Hamiltonian or TightBinding"):
        SpectralMesh(constant_insulator(1))


def test_spectral_mesh_exposes_its_root_resolution():
    mesh = SpectralMesh(TightBinding(constant_insulator(1)), root_level=2)

    assert mesh.root_level == 2
    assert mesh.tolerance == pytest.approx(1e-14)
    assert mesh.cached_vertices == 0
    assert mesh.active_simplices > 0
    assert mesh.active_vertices > 0


@pytest.mark.parametrize("tolerance", (-1.0, np.nan, np.inf))
def test_spectral_mesh_requires_a_finite_nonnegative_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        SpectralMesh(TightBinding(constant_insulator(1)), tolerance=tolerance)


@pytest.mark.parametrize("root_level", (-1, 31, 32))
def test_spectral_mesh_rejects_unsupported_root_levels(root_level):
    with pytest.raises(ValueError, match="root_level"):
        SpectralMesh(TightBinding(constant_insulator(1)), root_level=root_level)
