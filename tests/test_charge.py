from __future__ import annotations

import numpy as np
import pytest

from fermisimplex import ChargeErrorStats, CurrentMeshChargeResult, SpectralMesh

from .helpers import constant_insulator


def _adaptive_arguments(target_error: float = 1e-12) -> dict[str, object]:
    return {
        "target_error": target_error,
        "max_refinements": 0,
    }


def test_integrates_charge_on_a_tight_binding_model():
    mesh = SpectralMesh(constant_insulator(2))

    result = mesh.integrate_charge(
        mu=0.0,
        **_adaptive_arguments(),
    )

    assert result.value == pytest.approx(1.0)
    assert result.stopping_error == pytest.approx(0.0)
    assert result.dcharge_dmu == pytest.approx(0.0)
    assert isinstance(result.error_stats, ChargeErrorStats)
    assert result.stats.target_reached
    assert result.stats.refinements == 0
    assert result.stats.evaluations == result.stats.active_vertices
    assert result.stats.simplex_visits == result.stats.active_simplices


def test_evaluates_charge_without_refining_the_mesh():
    mesh = SpectralMesh(constant_insulator(1))

    result = mesh.estimate_charge_on_current_mesh(mu=0.0)

    assert isinstance(result, CurrentMeshChargeResult)
    assert result.value == pytest.approx(1.0)
    assert result.dcharge_dmu == pytest.approx(0.0)


def test_current_mesh_charge_does_not_refine_the_persistent_mesh():
    mesh = SpectralMesh(constant_insulator(1))
    active_simplices = mesh.active_simplices

    result = mesh.estimate_charge_on_current_mesh(mu=0.0)

    assert mesh.active_simplices == active_simplices
    assert mesh.cached_vertices == mesh.active_vertices
    assert result.value == pytest.approx(1.0)


def test_callable_hamiltonian_is_evaluated_with_separate_coordinates():
    seen_points: list[tuple[float, float, float]] = []

    def function(kx: float, ky: float, kz: float) -> np.ndarray:
        seen_points.append((kx, ky, kz))
        return np.diag([-1.0, 1.0]).astype(complex)

    mesh = SpectralMesh(function)

    result = mesh.estimate_charge_on_current_mesh(mu=0.0)

    assert result.value == pytest.approx(1.0)
    assert seen_points
    assert all(len(point) == 3 for point in seen_points)


def test_error_depth_increases_temporary_estimator_work():
    def band(k: float) -> np.ndarray:
        return np.array([[(k - 0.5) ** 2 - 0.1]], dtype=complex)

    shallow = SpectralMesh(band).integrate_charge(
        mu=0.0,
        target_error=10.0,
        max_refinements=0,
        error_depth=0,
    )
    subdivided = SpectralMesh(band).integrate_charge(
        mu=0.0,
        target_error=10.0,
        max_refinements=0,
        error_depth=1,
    )

    assert shallow.error_stats.terminal_simplices > 0
    assert (
        subdivided.error_stats.terminal_simplices
        >= shallow.error_stats.terminal_simplices
    )
    assert (
        subdivided.error_stats.hamiltonian_evaluations
        >= shallow.error_stats.hamiltonian_evaluations
    )


def test_charge_error_stats_are_nonnegative():
    result = SpectralMesh(constant_insulator(1)).integrate_charge(
        mu=0.0,
        target_error=10.0,
        max_refinements=0,
    )

    assert result.error_stats.root_simplices >= 0
    assert result.error_stats.hamiltonian_evaluations >= 0
    assert result.error_stats.micro_simplices >= 0
    assert result.error_stats.conservative_fallbacks >= 0


@pytest.mark.parametrize("error_depth", (-1, -2))
def test_charge_rejects_negative_error_depth(error_depth):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="error_depth"):
        mesh.integrate_charge(
            mu=0.0,
            target_error=10.0,
            max_refinements=0,
            error_depth=error_depth,
        )


def test_charge_error_depth_must_be_an_integer():
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(TypeError, match="error_depth"):
        mesh.integrate_charge(
            mu=0.0,
            target_error=10.0,
            max_refinements=0,
            error_depth=1.5,
        )


def test_charge_error_depth_defaults_to_two():
    def band(k: float) -> np.ndarray:
        return np.array([[(k - 0.5) ** 2 - 0.1]], dtype=complex)

    default = SpectralMesh(band).integrate_charge(
        mu=0.0,
        target_error=10.0,
        max_refinements=0,
    )
    explicit = SpectralMesh(band).integrate_charge(
        mu=0.0,
        target_error=10.0,
        max_refinements=0,
        error_depth=2,
    )

    assert default.stopping_error == pytest.approx(explicit.stopping_error)
    assert default.error_stats.micro_simplices == explicit.error_stats.micro_simplices
