from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class EvaluatedSnapshot:
    """All retained full spectra and the finest complete evaluated partition.

    ``simplices`` indexes rows of ``points``, ``eigenvalues`` and (when requested)
    ``eigenvectors`` directly. IDs are stable within the source mesh's lifetime.
    ``vertex_is_active`` marks current active-mesh membership; its complement
    identifies preview-only cache rows. ``simplex_is_active`` distinguishes
    retained active cells from preview descendants. Extra cached points can be
    absent from the selected connectivity when deeper previews are incomplete.

    The exact coordinate of row i is ``dyadic_numerators[i] / 2**dyadic_levels[i]``.
    Use rational arithmetic on these keys for exact midpoint/centroid reuse.
    Arrays are independent of the live mesh and read-only. Charge-error
    temporary spectra are excluded. Eigenvectors use (point, orbital, band) axes.
    """

    ndim: int
    ndof: int
    active_vertices: int
    active_simplices: int
    vertex_ids: np.ndarray
    points: np.ndarray
    dyadic_numerators: np.ndarray
    dyadic_levels: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None
    vertex_is_active: np.ndarray
    simplex_ids: np.ndarray
    active_ancestor_ids: np.ndarray
    simplices: np.ndarray
    volumes: np.ndarray

    @classmethod
    def _from_native(cls, data: dict) -> EvaluatedSnapshot:
        data["vertex_is_active"] = np.asarray(data["vertex_is_active"], dtype=bool)
        for value in data.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        return cls(**data)

    @property
    def cached_vertices(self) -> int:
        return len(self.vertex_ids)

    @property
    def preview_vertices(self) -> int:
        """Cached vertices outside the active mesh."""
        return self.cached_vertices - self.active_vertices

    @property
    def partition_vertices(self) -> int:
        return int(np.unique(self.simplices).size)

    @property
    def partition_simplices(self) -> int:
        return len(self.simplex_ids)

    @property
    def simplex_is_active(self) -> np.ndarray:
        result = self.simplex_ids == self.active_ancestor_ids
        result.flags.writeable = False
        return result
