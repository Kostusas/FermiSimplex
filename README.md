# FermiSimplex

**Adaptive, occupation-certified spectral calculations on simplex meshes.**

FermiSimplex finds Fermi surfaces and computes zero-temperature charge and
density matrices without paying for a dense momentum grid. Its central object
is the local occupation

$$
N(k; \mu) = \mathrm{Tr}\left[\Theta\left(\mu I - H(k)\right)\right],
$$

and its central question is simple: *can the occupation be proved constant on
this simplex, or should we look more closely?*

The upstream development repository is
[GitLab](https://gitlab.kwant-project.org/qt/lineartetrahedron); the
[GitHub repository](https://github.com/Kostusas/FermiSimplex) is a public
mirror.

![Adaptive Fermi-surface refinement](https://raw.githubusercontent.com/Kostusas/FermiSimplex/main/docs/assets/fermi_surface_refinement.gif)

See the [visual Python tour][visual-tour] for a presentation-ready
introduction with real adaptive sampling traces, multiband examples, and a
rotating noble-metal-inspired three-dimensional surface.

- 🛡️ **Gapped-region proofs** combine cached eigensystems with rigorous spectral
  bounds to exclude a Fermi-level crossing throughout a simplex.
- ⚡ **Adaptive sampling**, built on
  [AdaptiveSimplex](https://gitlab.kwant-project.org/qt/adaptivesimplex),
  concentrates diagonalizations near unresolved Fermi surfaces instead of
  refining the entire Brillouin zone uniformly.
- 🚀 **Numerical efficiency by design:** adaptive refinement, shared spectral
  caching, and the compiled numerical core avoid repeated work as the Fermi
  surface becomes progressively sharper.
- 🎯 **Recursive charge estimates** use certificate-selected active spaces,
  a corrected frozen-Schur reduction and actual-Hamiltonian microsimplex
  samples to target interpolation error near the Fermi level.
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

from fermisimplex import SpectralMesh


def hamiltonian(kx, ky, kz):
    phase = 2 * np.pi * np.array([kx, ky, kz])
    return np.array([[np.cos(phase).sum()]], dtype=complex)


mesh = SpectralMesh(hamiltonian)
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
Hamiltonian. `SpectralMesh` infers the momentum-space dimension from the
callable arguments and the matrix dimension by evaluating it at the origin.
Callables receive separate coordinates: `hamiltonian(kx, ky, ...)`.

![Two- and three-dimensional Fermi surfaces](https://raw.githubusercontent.com/Kostusas/FermiSimplex/main/docs/assets/fermi_surface_gallery.png)

The same `SpectralMesh` can drive the other observables and reuse every
eigensystem it has already computed:

```python
charge = mesh.integrate_charge(
    mu=0.17,
    target_error=1e-2,
    max_refinements=10_000,
    error_depth=1,
)
density = mesh.integrate_density_matrix(
    mu=0.17,
    lattice_vectors=[(0, 0, 0), (1, 0, 0)],
    target_error=1e-2,
    max_refinements=10_000,
)

charge.value
charge.stopping_error
charge.error_stats
density.matrices  # (number of lattice vectors, ndof, ndof)
```

For a tight-binding model,

$$
H(k)=\sum_R H_R e^{-2\pi i k\cdot R},
$$

pass `{R: H_R, ...}` directly to `SpectralMesh`. Opposite hoppings are checked
for $H_{-R}=H_R^\dagger$.

## What is certified?

The direct certificate and Fermi-surface calculation ask whether occupation can
change inside a simplex. They combine vertex eigensystems with
`curvature_bound`, which limits the Hamiltonian between samples. With a valid
bound, separated occupied and empty trial subspaces prove fixed occupation.

- **Certified:** no Fermi surface crosses the simplex.
- **Partially certified:** rigorous lower and upper occupation bounds remain.
- **Inconclusive:** this is not a gapless verdict; FermiSimplex refines and
  tries again.

`surface.coverage_certified` concerns classification down to
`min_feature_size`, not topology or geometric accuracy. Charge instead uses a
sampled recursive error estimate: it evaluates the actual Hamiltonian on
temporary microsimplices, reduces certificate-selected safe bands with one
corrected frozen-Schur step, and converts terminal midpoint defects into
shifted occupation volumes. `charge.stopping_error` is therefore useful for
adaptive
refinement but is not a rigorous bound; structure between sampled points can
still alias, and the frozen safe block is only a local approximation and does
not track its inertia away from the anchor. Density matrices also use adaptive
estimates.

Fermi-surface guarantees assume a valid `curvature_bound`. Omitting it, `None`,
and `0.0` all assert zero curvature; none disables certification. Charge has no
curvature argument. See the [mathematics guide][mathematics] for details.

## API at a glance

- `SpectralMesh`: accept a callable or tight-binding dictionary and own the
  adaptive geometry and cached eigensystems.
- `certify_simplex`: certify supplied vertex eigenpairs directly; eigenvalues
  must be finite and ascending, and eigenvector columns must be finite and
  orthonormal. These performance-sensitive numerical preconditions are not
  rechecked.
- `mesh.integrate_charge`: adaptive filling and $dQ/d\mu$.
- `mesh.integrate_density_matrix`: real-space density-matrix components.
- `mesh.fermi_surface`: band-labelled points and cells in reduced coordinates.

Adaptive controls are ordinary keyword arguments on the calculation that uses
them—there is no separate options object. For charge, `error_depth=1` permits
one complete temporary subdivision into $2^d$ microsimplices on each unresolved
branch. Certified branches stop early. Increasing the maximum depth adds
actual-Hamiltonian samples without refining the persistent mesh.
`charge.error_stats` reports the resulting reductions, solves, eigensystems,
temporary simplices, and fallbacks.

See the [visual Python tour][visual-tour], runnable
[quick start][quick-start], and
[two-band plotting example][fermi-example], the
[visual-generation notes][visuals], and the
[build and architecture guide][development].

## Development

AdaptiveSimplex provides the mesh geometry, refinement, vertex caching, and
cut-simplex integration; FermiSimplex adds the spectral models, certificates,
and observable-specific algorithms.

```bash
pixi run test
```

This builds the standalone C++ library, verifies an installed downstream CMake
consumer, rebuilds the Python extension, and runs the Python tests. The dense
60-band stress case lives in [benchmarks/fermi_surface_60.py][stress-benchmark].

FermiSimplex is licensed under the BSD 3-Clause license. If you use it in
research, please cite the metadata in [CITATION.cff][citation].


[visual-tour]: https://github.com/Kostusas/FermiSimplex/blob/main/docs/showcase.md
[mathematics]: https://github.com/Kostusas/FermiSimplex/blob/main/docs/mathematics.md
[quick-start]: https://github.com/Kostusas/FermiSimplex/blob/main/examples/quick_start.py
[fermi-example]: https://github.com/Kostusas/FermiSimplex/blob/main/examples/fermi_surface.py
[visuals]: https://github.com/Kostusas/FermiSimplex/blob/main/docs/visuals.md
[development]: https://github.com/Kostusas/FermiSimplex/blob/main/docs/development.md
[stress-benchmark]: https://github.com/Kostusas/FermiSimplex/blob/main/benchmarks/fermi_surface_60.py
[citation]: https://github.com/Kostusas/FermiSimplex/blob/main/CITATION.cff
