# LinearTetrahedron

LinearTetrahedron is the zero-temperature linear tetrahedron backend used by
MeanFi. It owns the physics-specific C++/Python boundary:

- dense tight-binding Fourier evaluation,
- LAPACK diagonalization at adaptive simplex vertices,
- zero-temperature occupation and density contribution rules,
- Python packaging and result conversion.

The adaptive mesh mechanics come from the refactored C++20 AdaptiveSimplex
library. If an external AdaptiveSimplex CMake package is available, the build
uses it. Otherwise CMake fetches the pinned AdaptiveSimplex source during the
package build, so installation needs internet access.

## Installation

The recommended install path is Pixi plus pip. Pixi provides Python, CMake,
Ninja, BLAS/LAPACK, and the compiler toolchain; pip builds the package from the
git repository.

```bash
pixi init
pixi add python numpy cmake ninja nanobind scikit-build-core libblas liblapack
pixi run python -m pip install --no-build-isolation \
  "git+https://github.com/YOUR_ORG/lineartetrahedron.git"
```

For plotting the examples, also install matplotlib:

```bash
pixi add matplotlib
```

Direct pip installation is supported on machines with a working C++20 compiler
and BLAS/LAPACK installation:

```bash
python -m pip install "git+https://github.com/YOUR_ORG/lineartetrahedron.git"
```

On macOS, CMake usually finds Apple's Accelerate framework automatically. On
Linux, install BLAS/LAPACK development packages first, or use Pixi/conda-forge
as above. AdaptiveSimplex is fetched automatically during the build unless CMake
can already find an installed `adaptivesimplex` package.

## Fermi Surface Example

Fermi-surface extraction uses reduced Brillouin-zone coordinates on `[0, 1]^d`.
The returned `points` and `cells` can be plotted directly with matplotlib:

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from lineartetrahedron import fermi_surface


def hamiltonian(kx: float, ky: float) -> np.ndarray:
    x = 2.0 * np.pi * kx
    y = 2.0 * np.pi * ky
    scalar = -0.12 + 0.42 * np.cos(x) - 0.34 * np.cos(y) + 0.16 * np.cos(x + y)
    dx = 0.22 * np.sin(x) + 0.10 * np.sin(x - y)
    dz = 0.18 * np.cos(y) - 0.08 * np.cos(2.0 * x + y)
    return np.array([[scalar + dz, dx], [dx, scalar - dz]], dtype=complex)


surface = fermi_surface(hamiltonian, mu=0.0, min_feature_size=0.01)

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.add_collection(LineCollection(surface.points[surface.cells], colors="#b91c1c", linewidths=0.8))
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal")
ax.set_xlabel("kx")
ax.set_ylabel("ky")
plt.show()
```

The same example is available as `scripts/example_fermi_surface_plot.py`.

## Local Development

The Pixi `test` task installs the package in editable mode and runs pytest.

```bash
pixi run test
```

To test against an external AdaptiveSimplex install instead of the vendored
fallback, point CMake at that prefix:

```bash
CMAKE_PREFIX_PATH=/path/to/adaptivesimplex/prefix pixi run test
```
