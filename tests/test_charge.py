from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import (
    AdaptiveOptions,
    Hamiltonian,
    SpectralMesh,
    TightBinding,
)

from .helpers import constant_insulator, winding_constant_gap_band


def _options(target_error: float = 1e-12) -> AdaptiveOptions:
    return AdaptiveOptions(
        target_error=target_error,
        max_refinements=0,
        preview_depth=1,
    )


def test_integrates_certified_charge_on_a_tight_binding_model():
    mesh = SpectralMesh(TightBinding(constant_insulator(2)))

    result = mesh.integrate_charge(0.0, _options(), curvature_bound=0.0)

    assert result.value == pytest.approx(1.0)
    assert result.stopping_error == pytest.approx(0.0)
    assert result.certified_error_bound == pytest.approx(0.0)
    assert result.dcharge_dmu == pytest.approx(0.0)
    assert result.stats.target_reached
    assert result.stats.refinements == 0


def test_evaluates_charge_without_refining_the_mesh():
    mesh = SpectralMesh(TightBinding(constant_insulator(1)))

    result = mesh.estimate_charge_on_current_mesh(0.0, 1e-12)

    assert result.value == pytest.approx(1.0)
    assert result.stats.refinements == 0


@pytest.mark.parametrize("target_error", (-1.0, np.inf, np.nan))
def test_current_mesh_charge_requires_a_finite_nonnegative_target(target_error):
    mesh = SpectralMesh(TightBinding(constant_insulator(1)))

    with pytest.raises(ValueError, match="target_error"):
        mesh.estimate_charge_on_current_mesh(0.0, target_error)


def test_current_mesh_charge_requires_a_positive_preview_depth():
    mesh = SpectralMesh(TightBinding(constant_insulator(1)))

    with pytest.raises(ValueError, match="preview_depth"):
        mesh.estimate_charge_on_current_mesh(0.0, 1.0, preview_depth=0)


def test_callable_hamiltonian_is_evaluated_with_coordinate_vectors():
    seen_shapes: list[tuple[int, ...]] = []

    def function(point: np.ndarray) -> np.ndarray:
        seen_shapes.append(point.shape)
        return np.diag([-1.0, 1.0]).astype(complex)

    mesh = SpectralMesh(
        Hamiltonian(
            function,
            ndim=3,
            ndof=2,
        )
    )

    result = mesh.estimate_charge_on_current_mesh(0.0, 1e-12)

    assert result.value == pytest.approx(1.0)
    assert seen_shapes
    assert set(seen_shapes) == {(3,)}


def test_none_and_zero_curvature_bounds_are_equivalent_for_charge():
    mesh = SpectralMesh(TightBinding(winding_constant_gap_band(3)))
    implicit_zero = mesh.estimate_charge_on_current_mesh(
        0.0,
        3.0,
    )
    explicit_none = mesh.estimate_charge_on_current_mesh(
        0.0,
        3.0,
        curvature_bound=None,
    )
    explicit_zero = mesh.estimate_charge_on_current_mesh(
        0.0,
        3.0,
        curvature_bound=0.0,
    )

    assert implicit_zero.value == pytest.approx(explicit_none.value)
    assert implicit_zero.value == pytest.approx(explicit_zero.value)
    assert implicit_zero.stopping_error == pytest.approx(
        explicit_none.stopping_error
    )
    assert implicit_zero.stopping_error == pytest.approx(
        explicit_zero.stopping_error
    )
    assert implicit_zero.certified_error_bound == pytest.approx(
        explicit_none.certified_error_bound
    )
    assert implicit_zero.certified_error_bound == pytest.approx(
        explicit_zero.certified_error_bound
    )


@pytest.mark.parametrize("curvature_bound", (-1.0, np.inf, np.nan))
def test_charge_rejects_invalid_curvature_bounds(curvature_bound):
    mesh = SpectralMesh(TightBinding(constant_insulator(1)))

    with pytest.raises(ValueError, match="curvature_bound"):
        mesh.estimate_charge_on_current_mesh(
            0.0,
            1.0,
            curvature_bound=curvature_bound,
        )


def test_callable_matrix_shape_is_checked_when_the_mesh_evaluates_it():
    model = Hamiltonian(
        lambda point: np.zeros((1, 2), dtype=complex),
        ndim=1,
        ndof=1,
    )
    mesh = SpectralMesh(model)

    with pytest.raises(ValueError, match="shape"):
        mesh.estimate_charge_on_current_mesh(0.0, 1e-12)
