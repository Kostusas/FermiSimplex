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
- 🎯 **Shared occupation bounds** use a quadratic reduced matrix model for
  charge errors and gap classification. Temporary subdivision of the model
  requires no further Hamiltonian evaluations.
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
Callables receive separate coordinates: `hamiltonian(kx, ky, ...)`. They are
trusted to keep returning finite Hermitian matrices of the inferred shape.

![Two- and three-dimensional Fermi surfaces](https://raw.githubusercontent.com/Kostusas/FermiSimplex/main/docs/assets/fermi_surface_gallery.png)

The same `SpectralMesh` can drive the other observables and reuse every
eigensystem it has already computed:

```python
charge = mesh.integrate_charge(
    mu=0.17,
    target_error=1e-2,
    max_refinements=10_000,
    error_depth=2,
)
density = mesh.integrate_density_matrix(
    mu=0.17,
    lattice_vectors=[(0, 0, 0), (1, 0, 0)],
    target_error=1e-2,
    max_refinements=10_000,
)
selected_density = mesh.integrate_density_components(
    mu=0.17,
    lattice_vectors=[(0, 0, 0), (1, 0, 0)],
    components=[(0, 0, 0), (1, 0, 1)],
    target_error=1e-2,
)

charge.value
charge.stopping_error
charge.error_stats
density.matrices  # (number of lattice vectors, ndof, ndof)
selected_density.values  # follows the component request order

weights = mesh.occupied_weights(0.17)
mesh.points        # (active_vertices, ndim), read-only
mesh.simplices     # (active_simplices, ndim + 1), read-only
mesh.eigenvalues   # (active_vertices, ndof), read-only
mesh.eigenvectors  # (active_vertices, ndof, ndof), read-only
particle_number = weights.sum()
band_energy = np.sum(weights * mesh.eigenvalues)
projectors = np.einsum("vib,vjb->vbij", mesh.eigenvectors, mesh.eigenvectors.conj())
onsite_density = np.einsum("vb,vbij->ij", weights, projectors)
```

`occupied_weights` only uses cached eigensystems on the current active mesh;
it performs no Hamiltonian evaluations or refinement. If an active vertex
has not been cached yet, it raises instead of filling the cache implicitly.

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
`min_feature_size`, not topology or geometric accuracy. Charge and surface
classification share a quadratic occupation enclosure. A positive surface
`curvature_bound` supplies a uniform interpolation allowance. Omitting it,
`None`, or `0.0` uses a sampled remainder. Charge also samples its remainder.
For general Hamiltonians, features between probes can still be missed.
See the [occupation design](docs/occupation-enclosure.md) for the degree-four
polynomial guarantee, the Schur bound and the sampling limitations.

## API at a glance

- `SpectralMesh`: accept a callable or tight-binding dictionary and own the
  adaptive geometry and cached eigensystems.
- `certify_simplex`: certify supplied vertex eigenpairs directly; eigenvalues
  must be finite and ascending, and eigenvector columns must be finite and
  orthonormal. These performance-sensitive numerical preconditions are not
  rechecked.
- `mesh.integrate_charge`: adaptive filling and $dQ/d\mu$.
- `mesh.estimate_charge_on_current_mesh`: direct linear-simplex filling and
  $dQ/d\mu$ with no error estimation or refinement.
- `mesh.points`, `mesh.simplices`, `mesh.eigenvalues`, and
  `mesh.eigenvectors`: read-only snapshots of the current active mesh and
  its cached eigensystems.
- `mesh.occupied_weights`: current-mesh occupied barycentric weights.
- `mesh.integrate_density_components`: selected real-space density entries,
  requested as `(lattice_vector_index, row, column)`.
- `mesh.integrate_density_matrix`: real-space density-matrix components.
- `mesh.fermi_surface`: band-labelled points and cells in reduced coordinates.

Adaptive controls are ordinary keyword arguments on the calculation that uses
them—there is no separate options object. Charge defaults to `error_depth=2`.
Each level permits one complete temporary subdivision into $2^d$
microsimplices on each unresolved branch; certified branches stop early.
Increasing the maximum depth subdivides the fixed quadratic model without
additional Hamiltonian samples or persistent mesh refinement.
`charge.error_stats` reports the resulting reductions, solves, eigensystems,
and temporary simplices.

Density integration follows charge integration on the same mesh. Full matrices
and selected entries use [adaptive polynomial cubature](docs/density-p-cubature.md):
start with Q3-Q1, raise the degree through seven by default, then bisect unresolved
density cells. The density tree preserves the charge cuts and does not change
the retained charge mesh. `max_degree`, `max_refinements` (promotions), and
`max_h_refinements` (bisections) bound the work; both refinement budgets are
unlimited by default. The reported quadrature error excludes remaining cut error.

`estimate_density_on_current_mesh` evaluates selected affine density moments
without refinement or an error estimate. Use it for an explicitly prescribed
mesh; it is not an adaptive integration method.

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

### Reusing retained density-preview spectra

The `points`, `simplices`, `eigenvalues` and `eigenvectors` properties describe
only the active mesh. After density integration, use:

```python
snapshot = mesh.evaluated_snapshot(include_eigenvectors=True)
cell_points = snapshot.points[snapshot.simplices]
cell_energies = snapshot.eigenvalues[snapshot.simplices]
print(snapshot.cached_vertices, snapshot.preview_vertices)
```

This independent read-only snapshot exports all cached full spectra, including
preview points, and the finest complete already-evaluated partition. Export
never evaluates the Hamiltonian, diagonalizes, or refines. Incomplete previews
fall back to complete evaluated ancestors while their cached points remain
available. A mesh without an evaluated covering raises an error. Exact dyadic
coordinate keys support downstream midpoint/centroid deduplication. Temporary
charge-error data is excluded. See the [snapshot design](docs/evaluated-snapshot.md)
for fields and guarantees. This API works with the existing AdaptiveSimplex
dependency; no changes to AdaptiveSimplex are required.

## Experimental quadratic occupation enclosure

`mesh.integrate_charge(...)` uses one quadratic matrix
model for occupation tests and charge error intervals. Inspect its per-simplex
results with `mesh.occupation_enclosures(mu=...)`. Surface classification uses
the same enclosure. The remainder is sampled unless the inspection
API receives a valid uniform interpolation bound. See the
[design and measured comparisons](docs/occupation-enclosure.md) for assumptions,
limitations, numerical tests and runtime tradeoffs.
