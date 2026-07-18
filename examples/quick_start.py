from __future__ import annotations

import numpy as np

from lineartetrahedron import Hamiltonian, SpectralMesh


CURVATURE_BOUND = (2.0 * np.pi) ** 2


def hamiltonian(point: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * np.asarray(point)
    return np.array([[np.cos(phase).sum()]], dtype=complex)


def main() -> None:
    mesh = SpectralMesh(Hamiltonian(hamiltonian, ndim=3, ndof=1))
    surface = mesh.fermi_surface(
        mu=0.17,
        min_feature_size=0.07,
        curvature_bound=CURVATURE_BOUND,
    )
    print(
        {
            "completed": surface.completed,
            "points": surface.points.shape,
            "cells": surface.cells.shape,
            "evaluations": surface.stats.evaluations,
        }
    )


if __name__ == "__main__":
    main()
