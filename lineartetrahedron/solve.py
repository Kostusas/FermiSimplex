from __future__ import annotations

import numpy as np

from .backend import _tb_type, build_runtime
from .results import DensityIntegrationInfo


def full_density_components(
    keys: list[tuple[int, ...]],
    *,
    size: int,
) -> list[tuple[int, int, tuple[int, ...]]]:
    grid = np.arange(int(size), dtype=np.int64)
    rows, cols = np.meshgrid(grid, grid, indexing="ij")
    components: list[tuple[int, int, tuple[int, ...]]] = []
    for key in keys:
        normalized_key = tuple(int(part) for part in key)
        components.extend(
            (int(row), int(col), normalized_key)
            for row, col in zip(rows.reshape(-1), cols.reshape(-1), strict=True)
        )
    return components


class PreparedDensityComponents:
    def __init__(
        self,
        *,
        size: int,
        keys: list[tuple[int, ...]],
        rows: np.ndarray,
        cols: np.ndarray,
        key_indices: np.ndarray,
    ) -> None:
        self.size = int(size)
        self.keys = tuple(keys)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.key_indices = np.asarray(key_indices, dtype=np.int64)

    @property
    def value_count(self) -> int:
        return int(self.rows.size)

    def values_to_tb(self, values: np.ndarray) -> _tb_type:
        values = np.asarray(values)
        rho = {
            key: np.zeros((self.size, self.size), dtype=complex) for key in self.keys
        }
        for index, value in enumerate(values):
            key = self.keys[int(self.key_indices[index])]
            rho[key][int(self.rows[index]), int(self.cols[index])] = value
        return rho

    def values_and_errors_to_tb(
        self,
        values: np.ndarray,
        errors: np.ndarray,
    ) -> tuple[_tb_type, _tb_type]:
        rho = self.values_to_tb(values)
        errors = np.asarray(errors)
        error_dtype = errors.dtype if errors.size else float
        rho_error = {
            key: np.zeros((self.size, self.size), dtype=error_dtype)
            for key in self.keys
        }
        for index, error in enumerate(errors):
            key = self.keys[int(self.key_indices[index])]
            rho_error[key][int(self.rows[index]), int(self.cols[index])] = error
        return rho, rho_error


def prepare_density_components(
    h: _tb_type,
    keys: list[tuple[int, ...]],
    density_components,
) -> PreparedDensityComponents:
    if density_components is None:
        raise ValueError("density_components must be provided")

    size = int(next(iter(h.values())).shape[0])
    normalized_keys = [tuple(int(part) for part in key) for key in keys]
    if not normalized_keys:
        raise ValueError("keys must be non-empty")
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("keys must be unique")
    ndim = len(normalized_keys[0])
    if any(len(key) != ndim for key in normalized_keys):
        raise ValueError("All keys must have the same dimension")
    key_index_by_key = {key: index for index, key in enumerate(normalized_keys)}

    rows: list[int] = []
    cols: list[int] = []
    key_indices: list[int] = []
    seen = set()
    for component in density_components:
        if len(component) != 3:
            raise ValueError("density components must be (row, col, key) triples")
        row, col, key = component
        row = int(row)
        col = int(col)
        key = tuple(int(part) for part in key)
        if len(key) != ndim:
            raise ValueError("density component key dimension does not match keys")
        if key not in key_index_by_key:
            raise ValueError(f"density component key {key!r} is not in keys")
        if row < 0 or col < 0 or row >= size or col >= size:
            raise ValueError("density component row/column out of bounds")
        dedupe_key = (row, col, key)
        if dedupe_key in seen:
            raise ValueError(f"duplicate density component {dedupe_key!r}")
        seen.add(dedupe_key)
        rows.append(row)
        cols.append(col)
        key_indices.append(key_index_by_key[key])

    return PreparedDensityComponents(
        size=size,
        keys=normalized_keys,
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        key_indices=np.asarray(key_indices, dtype=np.int64),
    )


def _density_info(result, runtime) -> DensityIntegrationInfo:
    return DensityIntegrationInfo(
        n_evaluator_evals=int(result.work),
        n_cached_nodes=int(runtime.n_cached_nodes),
        n_active_simplices=int(result.n_active_simplices),
        n_active_vertices=int(result.n_active_vertices),
        subdivisions=int(result.refinements),
    )


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_components,
    density_atol: float,
    max_subdivisions: int | None = None,
):
    prepared = prepare_density_components(h, keys, density_components)
    runtime = build_runtime(
        h,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    result = runtime.integrate_density(
        float(mu),
        float(density_atol),
        -1 if max_subdivisions is None else int(max_subdivisions),
    )
    rho, error = prepared.values_and_errors_to_tb(
        result.estimate_array(),
        result.error_vector_array(),
    )
    return rho, error, _density_info(result, runtime)
