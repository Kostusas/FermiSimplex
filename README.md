# LinearTetrahedron

**Adaptive, occupation-certified spectral calculations on simplex meshes.**

LinearTetrahedron finds Fermi surfaces and computes zero-temperature charge and
density matrices without paying for a dense momentum grid. Its central object
is the local occupation

$$
N(k;\mu)=\operatorname{Tr}\Theta\!\left(\mu I-H(k)\right),
$$

and its central question is simple: *can the occupation be proved constant on
this simplex, or should we look more closely?*

![Adaptive Fermi-surface refinement](docs/assets/fermi_surface_refinement.gif)

- 🛡️ **Occupation certificates** prove gapped simplices and rigorous charge
  bounds from vertex eigensystems and a user-supplied curvature bound.
- ⚡ **Adaptive sampling** concentrates diagonalizations near unresolved Fermi
  surfaces instead of refining the entire Brillouin zone uniformly.
- ♻️ **Shared spectral cache** lets Fermi-surface, charge, and density-matrix
  calculations reuse the same mesh and eigensystems.
- 🎯 **Projected charge estimates** inspect the nonlinear Hamiltonian residual
  only in the bands whose occupation is still ambiguous.
- 🧩 **Python and C++** share one numerical core; models can be dense callables
  or translation-invariant tight-binding Hamiltonians.

## Quick start

From a source checkout with a C++20 compiler and BLAS/LAPACK available:

```bash
pip install .
```

The model below produces the three-dimensional surface shown above:

```python
import numpy as np

from lineartetrahedron import Hamiltonian, SpectralMesh


def hamiltonian(kx, ky, kz):
    phase = 2 * np.pi * np.array([kx, ky, kz])
    return np.array([[np.cos(phase).sum()]], dtype=complex)


mesh = SpectralMesh(Hamiltonian(hamiltonian))
surface = mesh.fermi_surface(
    mu=0.17,
    min_feature_size=0.07,
    curvature_bound=(2 * np.pi) ** 2,
)

surface.points      # (npoints, 3)
surface.cells       # (ntriangles, 3)
surface.cell_bands  # band index for every triangle
```

The coordinates are reduced coordinates in $[0,1]^d$. Here
$M=(2\pi)^2$ bounds every directional second derivative of the scalar
Hamiltonian. `Hamiltonian` infers the momentum-space dimension from the
function arguments and the matrix dimension by evaluating it at the origin.
Both model types are evaluated with separate coordinates: `model(kx, ky, ...)`.

![Two- and three-dimensional Fermi surfaces](docs/assets/fermi_surface_gallery.png)

The same `SpectralMesh` can drive the other observables:

```python
from lineartetrahedron import AdaptiveOptions

options = AdaptiveOptions(target_error=1e-2, max_refinements=10_000)

charge = mesh.integrate_charge(
    mu=0.17,
    options=options,
    curvature_bound=(2 * np.pi) ** 2,
)
density = mesh.integrate_density_matrix(
    mu=0.17,
    lattice_vectors=[(0, 0, 0), (1, 0, 0)],
    options=options,
)

charge.value
charge.stopping_error
charge.certified_error_bound
density.matrices  # (number of lattice vectors, ndof, ndof)
```

For a tight-binding model,

$$
H(k)=\sum_R H_R e^{-2\pi i k\cdot R},
$$

use `TightBinding({R: H_R, ...})`. Opposite hoppings are checked for
$H_{-R}=H_R^\dagger$.

## What is certified?

For a simplex of diameter $D$, a supplied curvature bound $M$ gives the
Hamiltonian interpolation bound

$$
\epsilon=\frac12 M D^2.
$$

Every charge and Fermi-surface simplex is passed through the certificate.
Omitting `curvature_bound`, passing `None`, and passing `0.0` are equivalent:
all assert zero curvature; none disables certification.

- `charge.certified_error_bound` is rigorous if the supplied $M$ is valid.
- `charge.stopping_error` is the projected/adaptive estimate used to decide
  refinement; it is not a certificate.
- `surface.coverage_certified` concerns occupation classification down to the
  requested feature size, not topology or geometric/Hausdorff error.
- density-matrix `stopping_error` is an adaptive quadrature estimate; density
  matrices are not currently certified.

The derivation—including the rotated-subspace certificate, partial occupation
bounds, asymmetric residual pairing, cut-simplex charge formula, and preview
error—is in [the mathematics guide](docs/mathematics.md).

## API at a glance

- `Hamiltonian`: wrap a dense Python callable.
- `TightBinding`: evaluate a Hermitian hopping expansion.
- `SpectralMesh`: own adaptive geometry and cached eigensystems.
- `certify_simplex`: certify supplied vertex eigenpairs directly.
- `integrate_charge`: adaptive filling and $dQ/d\mu$.
- `integrate_density_matrix`: real-space density-matrix components.
- `fermi_surface`: band-labelled points and cells in reduced coordinates.

See the runnable [quick start](examples/quick_start.py) and
[two-band plotting example](examples/fermi_surface.py), the
[visual-generation notes](docs/visuals.md), and the
[build and architecture guide](docs/development.md).

## Development

```bash
pixi run test
```

This builds the standalone C++ library, verifies an installed downstream CMake
consumer, rebuilds the Python extension, and runs the Python tests. The dense
60-band stress case lives in [benchmarks/fermi_surface_60.py](benchmarks/fermi_surface_60.py).
