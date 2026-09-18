# Evaluated spectral snapshot

## Goal and scope

`mesh.evaluated_snapshot(include_eigenvectors=True)` copies all retained full
spectra and the finest complete evaluated partition into a read-only snapshot.
Density-preview spectra are included. Temporary charge-error workspaces and
reduced-model spectra are excluded. Export performs no Hamiltonian evaluation,
diagonalization, refinement, or change to later integration behavior.

The active mesh controls the integrator. Its preview tree may contain finer
cells, including unevaluated cells. The complete spectral cache is independent
of either partition: export every retained entry, even if the selected partition
does not reference it. Existing `points`, `simplices`, `eigenvalues`,
`eigenvectors`, and `occupied_weights` retain their active-mesh meanings.

## Representation and algorithm

AdaptiveSimplex's `core::evaluated_partition` selects complete evaluated child
coverings recursively, retaining a parent if deeper data is incomplete. Its
frontier covers the active domain once, without overlapping parents and
children, in any simplex dimension. A mesh without an evaluated covering raises
an error; export never fills missing data. Selection assumes Geometry's valid
bisection trees and a spectral cache belonging to that geometry/model.

FermiSimplex enumerates cached vertex IDs in ascending order and copies spectra,
coordinates and exact dyadic keys. A single ID-to-row map translates connectivity
into direct indices into the exported point and spectral arrays. Vertex and
simplex IDs are stable within the owning mesh's lifetime, not globally across
meshes. Provenance is current active membership, not the historical reason a
point was first evaluated. Each selected simplex also records its active
ancestor ID. Counts distinguish all cached vertices, active vertices, preview-only
vertices and vertices used by the exported partition.

Python returns a frozen `EvaluatedSnapshot` with independent read-only arrays.
Later integration and deletion of the mesh cannot invalidate it. Eigenvectors
are optional to avoid their copy cost; omitting them never changes the cache.
Their axes follow numpy.linalg.eigh: (point, orbital, band).
No concurrent mutation of the same SpectralMesh is supported during export.

## Exact coordinate reuse

For point i and coordinate j, the exact reduced coordinate is
`dyadic_numerators[i, j] / 2**dyadic_levels[i]`. MeanFi can construct rational
midpoint and centroid keys from these integers to deduplicate across cells and
against the cache, without choosing a floating-point rounding tolerance.
Centroids need not be dyadic. Geometric endpoints 0 and 1 remain distinct;
periodic identification requires the caller's model and eigenvector convention.

The denominator for the requested reuse metric is `snapshot.cached_vertices`.
Its numerator is the number of unique quadrature keys absent from the cache,
shared across both rules before any new diagonalizations.

## Validation and development

Core tests establish tree coverage, incomplete-preview fallback and arbitrary
dimension traversal. Python tests check all retained spectra against known
Hamiltonians, zero evaluator calls, metadata/indexing, read-only lifetime,
repeated exports and subsequent integration, plus exact midpoint/centroid reuse.
Illustrative trapezoid/Simpson energy quadrature has an exact polynomial integral,
with absolute tolerance
1e-12 for floating-point accumulation in the small test fixtures.

The CMake dependency pin includes the AdaptiveSimplex evaluated-partition API
(commit `5ea4787abf801b15fc42c374da4c9418cba2f556`). For development against a
local checkout, use
`-DFETCHCONTENT_SOURCE_DIR_ADAPTIVESIMPLEX=/path/to/adaptivesimplex`.
No new dependencies are introduced.

Validation on the implementation: dimensions 1 through 4 have worst volume
error 1.45e-15. For the polynomial x^2 - 2 on [0, 1], composite trapezoid error
is 1/96 and Simpson error is zero in the test run. These illustrate sample reuse;
they do not prescribe MeanFi's Q2/Q3 rules. A nonuniform 2D density-preview case
has 48 midpoint/centroid requests, 40 unique coordinates, 4 existing preview hits
and 36 genuinely missing coordinates.
