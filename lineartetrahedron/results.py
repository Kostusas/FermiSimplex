from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DensityIntegrationInfo:
    n_evaluator_evals: int
    n_cached_nodes: int
    n_active_simplices: int
    n_active_vertices: int
    subdivisions: int
