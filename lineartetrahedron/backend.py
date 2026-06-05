from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ._native import IntegrationRuntime, TightBindingModel

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    IntegrationRuntime = None
    TightBindingModel = None
    NATIVE_AVAILABLE = False

_tb_type = dict[tuple[int, ...], np.ndarray]
_GEOM_TOL = 1e-14


def _is_sparse_like(matrix: Any) -> bool:
    return hasattr(matrix, "toarray") and hasattr(matrix, "tocoo")


def _require_native_extension() -> None:
    if not NATIVE_AVAILABLE or TightBindingModel is None or IntegrationRuntime is None:
        raise RuntimeError(
            "Zero-temperature integration requires the compiled "
            "lineartetrahedron._native extension"
        )


def _validate_dense_tb(tb: _tb_type) -> None:
    if not tb:
        raise ValueError("Tight-binding Hamiltonian cannot be empty")
    ndim = len(next(iter(tb)))
    if ndim < 1 or ndim > 3:
        raise ValueError("LinearTetrahedron supports dimensions 1, 2, and 3")
    ndof: int | None = None
    for key, matrix in tb.items():
        if len(tuple(key)) != ndim:
            raise ValueError("All tight-binding keys must have the same dimension")
        if _is_sparse_like(matrix):
            raise TypeError("LinearTetrahedron supports dense complex128 matrices only")
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("Tight-binding values must be square matrices")
        if ndof is None:
            ndof = int(array.shape[0])
        elif array.shape != (ndof, ndof):
            raise ValueError("All tight-binding matrices must have the same shape")


def tb_to_tight_binding_model(tb: _tb_type):
    _require_native_extension()
    _validate_dense_tb(tb)
    keys = np.asarray([tuple(key) for key in tb.keys()], dtype=np.int64, order="C")
    matrices = np.asarray([np.asarray(value) for value in tb.values()], dtype=np.complex128, order="C")
    return TightBindingModel(keys, matrices)


def build_runtime(
    tb: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    component_rows: np.ndarray,
    component_cols: np.ndarray,
    component_key_indices: np.ndarray,
):
    _require_native_extension()
    model = tb_to_tight_binding_model(tb)
    return IntegrationRuntime(
        model,
        np.ascontiguousarray(np.asarray(keys, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(component_rows, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(component_cols, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(component_key_indices, dtype=np.int64)),
        float(_GEOM_TOL),
    )
