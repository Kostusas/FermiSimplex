from __future__ import annotations

import numpy as np
import pytest

from fermisimplex import SpectralMesh

from .helpers import constant_insulator, winding_constant_gap_band


def _adaptive_arguments(target_error: float = 1e-12) -> dict[str, object]:
    return {
        "target_error": target_error,
        "max_refinements": 0,
        "preview_depth": 1,
    }


def test_integrates_certified_charge_on_a_tight_binding_model():
    mesh = SpectralMesh(constant_insulator(2))

    result = mesh.integrate_charge(
        mu=0.0,
        curvature_bound=0.0,
        **_adaptive_arguments(),
    )

    assert result.value == pytest.approx(1.0)
    assert result.stopping_error == pytest.approx(0.0)
    assert result.certified_error_bound == pytest.approx(0.0)
    assert result.dcharge_dmu == pytest.approx(0.0)
    assert result.stats.target_reached
    assert result.stats.refinements == 0


def test_evaluates_charge_without_refining_the_mesh():
    mesh = SpectralMesh(constant_insulator(1))

    result = mesh.estimate_charge_on_current_mesh(mu=0.0, target_error=1e-12)

    assert result.value == pytest.approx(1.0)
    assert result.stats.refinements == 0


@pytest.mark.parametrize("target_error", (-1.0, np.inf, np.nan))
def test_current_mesh_charge_requires_a_finite_nonnegative_target(target_error):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="target_error"):
        mesh.estimate_charge_on_current_mesh(mu=0.0, target_error=target_error)


def test_current_mesh_charge_requires_a_positive_preview_depth():
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="preview_depth"):
        mesh.estimate_charge_on_current_mesh(
            mu=0.0,
            target_error=1.0,
            preview_depth=0,
        )


def test_callable_hamiltonian_is_evaluated_with_separate_coordinates():
    seen_points: list[tuple[float, float, float]] = []

    def function(kx: float, ky: float, kz: float) -> np.ndarray:
        seen_points.append((kx, ky, kz))
        return np.diag([-1.0, 1.0]).astype(complex)

    mesh = SpectralMesh(function)

    result = mesh.estimate_charge_on_current_mesh(mu=0.0, target_error=1e-12)

    assert result.value == pytest.approx(1.0)
    assert seen_points
    assert all(len(point) == 3 for point in seen_points)


def test_none_and_zero_curvature_bounds_are_equivalent_for_charge():
    mesh = SpectralMesh(winding_constant_gap_band(3))
    implicit_zero = mesh.estimate_charge_on_current_mesh(
        mu=0.0,
        target_error=3.0,
    )
    explicit_none = mesh.estimate_charge_on_current_mesh(
        mu=0.0,
        target_error=3.0,
        curvature_bound=None,
    )
    explicit_zero = mesh.estimate_charge_on_current_mesh(
        mu=0.0,
        target_error=3.0,
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
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="curvature_bound"):
        mesh.estimate_charge_on_current_mesh(
            mu=0.0,
            target_error=1.0,
            curvature_bound=curvature_bound,
        )
