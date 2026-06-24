# LinearTetrahedron

Adaptive Fermi-surface extraction for dense tight-binding and callable
Hamiltonians. The public Python API works in reduced Brillouin-zone coordinates
`[0, 1]^d`:

```python
surface = fermi_surface(H, mu=0.0, min_feature_size=0.01)
```

`surface.points` contains reduced-coordinate surface points, and
`surface.cells` indexes line segments into those points for 2D plots.

Coordinates passed to the Hamiltonian are normalized. In 2D, `kx=0` and
`kx=1` are the same reciprocal-lattice point, and the physical phase convention
for a tight-binding term with lattice vector `R` is
`exp(-2j * pi * dot(k, R))`.

## Install

Recommended: add the Git branch as a Pixi source package. Pixi will build the
native extension with the right compiler, CMake/Ninja, BLAS/LAPACK, and Python
dependencies:

```bash
pixi init
pixi add matplotlib
pixi add lineartetrahedron \
  --git https://gitlab.kwant-project.org/qt/lineartetrahedron.git \
  --branch inertia-marking
```

The build fetches a pinned AdaptiveSimplex source automatically, so installation
needs internet access.

## Plot A Fermi Surface

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from lineartetrahedron import fermi_surface


def H(kx, ky):
    # kx and ky are reduced coordinates in [0, 1].
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
    margin=0.0,
    return_states=False,
)
```

`hamiltonian` may be either:

- a tight-binding dictionary `{tuple[int, ...]: np.ndarray}`;
- a callable with explicit scalar arguments, such as `H(kx, ky)`.

For callable Hamiltonians:

- the number of required positional arguments is the dimension;
- each argument is a reduced coordinate in `[0, 1]`;
- the callable must return a dense square NumPy-compatible matrix;
- vector-style callables such as `H(k)` are intentionally not accepted.

For tight-binding dictionaries, keys are integer lattice vectors and values are
dense square hopping matrices. The Hamiltonian is evaluated as

```python
H(k) = sum(H_R * exp(-2j * pi * dot(k, R)) for R, H_R in hamiltonian.items())
```

where `k` is reduced-coordinate.

`min_feature_size` is the target smallest simplex edge length in reduced
Brillouin-zone units. For example, `min_feature_size=0.01` means the adaptive
mesh keeps refining visible or unresolved Fermi-surface regions until triangle
edges are about one percent of the reduced-zone side length, unless they are
certified safe earlier. Smaller values give sharper Fermi surfaces and require
more diagonalizations.

`max_diagonalizations` is optional and caps unique mesh-vertex diagonalizations.
If the cap is hit before the adaptive run finishes, `surface.converged` is
`False`.

`margin` is an optional energy-unit safety buffer for certification. The default
`margin=0.0` gives the usual certificate. A positive value requires certified
same-sign states to remain at least `margin` away from `mu`, so larger margins
force more refinement. Visible Fermi crossings are still detected independently
of this margin.

Set `return_states=True` to attach approximate eigenstates to the extracted
Fermi-surface points:

```python
surface = fermi_surface(H, mu=0.0, min_feature_size=0.01, return_states=True)
states = surface.states
```

For each Fermi point, the state is copied from the endpoint vertex of the
crossing edge whose band eigenvalue is closest to `mu`. This is cheap and useful
for local expectation values such as spin, orbital, sublattice, or layer weight.
The phases are whatever the diagonalizer returned, so these states should not be
used directly for Berry phases or derivatives along the Fermi contour.

When states are requested:

- `surface.states.band_indices`: crossing band for each point;
- `surface.states.eigenvalues`: selected endpoint eigenvalue;
- `surface.states.eigenvectors`: selected endpoint eigenvector.

Useful result fields:

- `surface.points`: array with shape `(n_points, ndim)`;
- `surface.cells`: array with shape `(n_cells, ndim)`, line segments in 2D;
- `surface.converged`: whether all active simplices reached a terminal decision;
- `surface.stats.evaluated_vertices`: number of unique Hamiltonian
  diagonalizations;
- `surface.stats.cut_simplices`, `feature_size_simplices`, and
  `unresolved_simplices`: compact terminal-outcome counters;
- `surface.parameters`: resolved `mu`, `min_feature_size`, dimension, and matrix
  size.

## Development

From a checkout:

```bash
pixi run test
```
