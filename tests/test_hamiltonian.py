from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import fermisimplex
from fermisimplex import SpectralMesh
from fermisimplex import _native

from .helpers import constant_insulator, qiwuzhang, tb_k_matrix


def test_tight_binding_evaluates_fourier_series():
    hoppings = qiwuzhang()
    mesh = SpectralMesh(hoppings)
    point = np.array([0.3, -0.7])

    assert mesh.ndim == 2
    assert mesh.ndof == 2
    assert np.allclose(mesh.evaluate(*point), tb_k_matrix(hoppings, point))


def test_tight_binding_requires_global_hermiticity():
    with pytest.raises(RuntimeError, match="opposite partner"):
        SpectralMesh({(1,): np.array([[0.5]], dtype=complex)})

    with pytest.raises(RuntimeError, match="adjoint"):
        SpectralMesh({(0,): np.array([[1.0j]], dtype=complex)})


def test_tight_binding_rejects_nonfinite_coefficients():
    with pytest.raises(RuntimeError, match="entries must be finite"):
        SpectralMesh({(0,): np.array([[np.nan]], dtype=complex)})


def test_callable_hamiltonian_infers_dimensions_from_function_and_result():
    calls: list[tuple[float, float]] = []

    def function(kx: float, ky: float) -> np.ndarray:
        calls.append((kx, ky))
        return np.diag([kx + ky, -kx - ky])

    mesh = SpectralMesh(function)

    assert mesh.ndim == 2
    assert mesh.ndof == 2
    assert calls == [(0.0, 0.0)]
    assert np.allclose(mesh.evaluate(0.25, 0.5), np.diag([0.75, -0.75]))
    assert calls[-1] == (0.25, 0.5)


def test_models_require_one_argument_per_coordinate():
    callable_mesh = SpectralMesh(lambda kx, ky: np.array([[kx + ky]]))
    tight_binding_mesh = SpectralMesh(constant_insulator(2))

    for mesh in (callable_mesh, tight_binding_mesh):
        with pytest.raises(ValueError, match="expected 2 coordinates"):
            mesh.evaluate([0.25, 0.5])


def test_callable_spectral_mesh_does_not_leak_at_interpreter_shutdown():
    script = """
import numpy as np

from fermisimplex import SpectralMesh

mesh = SpectralMesh(lambda k: np.array([[k]], dtype=complex))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "nanobind: leaked" not in completed.stderr


def test_hamiltonian_coordinates_must_be_finite():
    mesh = SpectralMesh(lambda k: np.array([[k]], dtype=complex))

    with pytest.raises(ValueError, match="coordinates must be finite"):
        mesh.evaluate(np.nan)


def test_spectral_evaluation_rejects_nonhermitian_matrices():
    mesh = SpectralMesh(
        lambda _k: np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    )

    with pytest.raises(RuntimeError, match="Hamiltonian must be Hermitian"):
        mesh.integrate_charge(
            mu=0.0,
            target_error=1.0,
            max_refinements=0,
        )


def test_spectral_evaluation_rejects_nonfinite_matrix_entries():
    mesh = SpectralMesh(lambda _k: np.array([[np.nan]], dtype=complex))

    with pytest.raises(RuntimeError, match="Hamiltonian entries must be finite"):
        mesh.integrate_charge(
            mu=0.0,
            target_error=1.0,
            max_refinements=0,
        )


@pytest.mark.parametrize(
    "function",
    (
        lambda: np.eye(1),
        lambda k=0.0: np.eye(1),
        lambda *k: np.eye(1),
        lambda **k: np.eye(1),
        lambda *, k: np.eye(1),
    ),
)
def test_callable_hamiltonian_requires_explicit_coordinate_parameters(function):
    with pytest.raises(TypeError, match="required positional coordinates"):
        SpectralMesh(function)


def test_callable_hamiltonian_requires_a_square_origin_matrix():
    with pytest.raises(ValueError, match="non-empty square matrix"):
        SpectralMesh(lambda _k: np.zeros((1, 2)))


def test_callable_hamiltonian_requires_a_consistent_matrix_shape():
    def function(k):
        return np.eye(2) if k == 0.0 else np.eye(1)

    mesh = SpectralMesh(function)

    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        mesh.evaluate(0.5)


def test_spectral_mesh_requires_a_callable_or_tight_binding_mapping():
    with pytest.raises(TypeError, match="callable or a tight-binding mapping"):
        SpectralMesh(object())


def test_model_and_options_wrappers_are_not_public_api():
    assert not hasattr(fermisimplex, "Hamiltonian")
    assert not hasattr(fermisimplex, "TightBinding")
    assert not hasattr(fermisimplex, "AdaptiveOptions")
    assert not hasattr(_native, "TightBindingModel")
    assert not hasattr(_native, "AdaptiveOptions")


def test_spectral_mesh_exposes_its_root_resolution():
    mesh = SpectralMesh(constant_insulator(1), root_level=2)

    assert mesh.root_level == 2
    assert mesh.tolerance == pytest.approx(1e-14)
    assert mesh.cached_vertices == 0
    assert mesh.active_simplices > 0
    assert mesh.active_vertices > 0


@pytest.mark.parametrize("tolerance", (-1.0, np.nan, np.inf))
def test_spectral_mesh_requires_a_finite_nonnegative_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        SpectralMesh(constant_insulator(1), tolerance=tolerance)


@pytest.mark.parametrize("root_level", (-1, 31, 32))
def test_spectral_mesh_rejects_unsupported_root_levels(root_level):
    with pytest.raises(ValueError, match="root_level"):
        SpectralMesh(constant_insulator(1), root_level=root_level)
