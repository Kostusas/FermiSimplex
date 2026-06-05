from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChargeIntegrationInfo:
    charge: float
    charge_error: float
    dcharge_dmu: float
    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_active_simplices: int
    n_active_vertices: int
    refinements: int
    converged: bool
    error_estimate_available: bool


@dataclass(frozen=True)
class DensityIntegrationInfo:
    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_active_simplices: int
    n_active_vertices: int
    subdivisions: int
    error_estimate_available: bool


@dataclass(frozen=True)
class FixedFillingInfo:
    mu: float
    charge: float
    charge_error: float
    dcharge_dmu: float
    charge_evaluations: int
    charge_integration_calls: int
    density_integration_calls: int
    charge_n_kernel_evals: int
    density_n_kernel_evals: int
    n_kernel_evals: int
    unique_evals: int
    charge_n_evaluator_evals: int
    density_n_evaluator_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_active_simplices: int
    n_active_vertices: int
    subdivisions: int
    charge_integral_atol: float
    density_atol: float
    density_rtol: float
    error_estimate_available: bool
