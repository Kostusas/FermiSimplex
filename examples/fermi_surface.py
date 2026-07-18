from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from fermisimplex import Hamiltonian, SpectralMesh


CURVATURE_BOUND = 2.08 * (2.0 * np.pi) ** 2


def hamiltonian(kx: float, ky: float) -> np.ndarray:
    x = 2.0 * np.pi * kx
    y = 2.0 * np.pi * ky
    scalar = -0.12 + 0.42 * np.cos(x) - 0.34 * np.cos(y) + 0.16 * np.cos(x + y)
    dx = 0.22 * np.sin(x) + 0.10 * np.sin(x - y)
    dz = 0.18 * np.cos(y) - 0.08 * np.cos(2.0 * x + y)
    return np.array([[scalar + dz, dx], [dx, scalar - dz]], dtype=complex)


def main() -> None:
    model = Hamiltonian(hamiltonian)
    surface = SpectralMesh(model).fermi_surface(
        0.0,
        min_feature_size=0.01,
        # User-supplied triangle-inequality bound for the Fourier terms above.
        curvature_bound=CURVATURE_BOUND,
    )

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    if surface.cells.size:
        ax.add_collection(
            LineCollection(
                surface.points[surface.cells],
                colors="#b91c1c",
                linewidths=0.8,
                alpha=0.95,
            )
        )
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="kx", ylabel="ky")
    ax.set_aspect("equal")
    ax.set_title(
        f"Fermi surface ({surface.stats.evaluations} vertex diagonalizations)"
    )
    plt.show()


if __name__ == "__main__":
    main()
