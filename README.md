# LinearTetrahedron

Adaptive Fermi-surface extraction for dense tight-binding and callable
Hamiltonians. The public Python API works in reduced Brillouin-zone coordinates
`[0, 1]^d`:

```python
surface = fermi_surface(H, mu=0.0, min_feature_size=0.01)
```

`surface.points` contains reduced-coordinate surface points, and
`surface.cells` indexes line segments into those points for 2D plots.

## Install

Recommended: use Pixi to provide the compiler, CMake/Ninja, BLAS/LAPACK, and
Python dependencies, then install the Git branch with pip:

```bash
pixi init
pixi add "python>=3.11,<3.14" numpy cmake ninja nanobind scikit-build-core \
  libblas liblapack pip cxx-compiler matplotlib
pixi run python -m pip install --no-build-isolation \
  "git+https://gitlab.kwant-project.org/qt/lineartetrahedron.git@inertia-marking"
```

The build fetches a pinned AdaptiveSimplex source automatically, so installation
needs internet access. On macOS, direct pip can often work because CMake finds
Apple Accelerate automatically; on Linux, Pixi is the safer route.

## Plot A Fermi Surface

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from lineartetrahedron import fermi_surface


def H(kx, ky):
    x = 2.0 * np.pi * kx
    y = 2.0 * np.pi * ky
    scalar = -0.12 + 0.42 * np.cos(x) - 0.34 * np.cos(y) + 0.16 * np.cos(x + y)
    dx = 0.22 * np.sin(x) + 0.10 * np.sin(x - y)
    dz = 0.18 * np.cos(y) - 0.08 * np.cos(2.0 * x + y)
    return np.array([[scalar + dz, dx], [dx, scalar - dz]], dtype=complex)


surface = fermi_surface(H, mu=0.0, min_feature_size=0.01)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
if surface.cells.size:
    ax.add_collection(
        LineCollection(surface.points[surface.cells], colors="#b91c1c", linewidths=0.8)
    )
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal")
ax.set_xlabel("kx")
ax.set_ylabel("ky")
ax.set_title(f"Fermi surface, {surface.stats.evaluated_vertices} diagonalizations")
plt.show()
```

## API

```python
from lineartetrahedron import fermi_surface

surface = fermi_surface(
    hamiltonian,
    mu=0.0,
    min_feature_size=0.01,
    max_diagonalizations=None,
)
```

`hamiltonian` may be either:

- a tight-binding dictionary `{tuple[int, ...]: np.ndarray}`;
- a callable with explicit scalar arguments, such as `H(kx, ky)`.

`max_diagonalizations` is optional and caps unique mesh-vertex diagonalizations.
If the cap is hit before the adaptive run finishes, `surface.converged` is
`False`.

## Development

From a checkout:

```bash
pixi run test
```
