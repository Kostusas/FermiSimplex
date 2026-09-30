# Performance measurement

FermiSimplex measures C++ cost relative to the same complex Hermitian LAPACK
eigensolve used by the library. Python startup, callbacks, and bindings are
deliberately excluded.

## Build and run

Use a Release build and one BLAS/LAPACK thread:

```sh
cmake -S . -B build/cpp-benchmarks -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFERMISIMPLEX_BUILD_PYTHON=OFF \
  -DFERMISIMPLEX_BUILD_TESTING=OFF \
  -DFERMISIMPLEX_BUILD_BENCHMARKS=ON
cmake --build build/cpp-benchmarks \
  --target fermisimplex_run_performance_benchmark
```

The custom target writes
`build/cpp-benchmarks/benchmarks/fermisimplex-performance.json`.
The equivalent command is:

```sh
pixi run benchmark-cpp
```

The executable can also be run directly:

```sh
build/cpp-benchmarks/cpp/fermisimplex_performance_benchmark \
  --preset quick --output result.json
```

Available presets are `quick`, `ci`, and `full`. Use `quick` as a smoke
test, `ci` for repeatable version tracking, and `full` for larger matrices
and hopping sets. Every preset performs an untimed LAPACK warm-up.

The larger-matrix charge scaling mode remains available:

```sh
build/cpp-benchmarks/cpp/fermisimplex_performance_benchmark \
  --preset ci --only charge-scaling --output charge-scaling.json
```

## Charge-estimator benchmark

Use the focused mode when changing the shared occupation enclosure:

```sh
build/cpp-benchmarks/cpp/fermisimplex_performance_benchmark \
  --preset quick --only charge-estimator \
  --output charge-estimator.json
```

This mode uses a deterministic 2D diagonal model containing an active cluster
of requested width `q` embedded between separated occupied and empty spectator
bands. The mesh is fixed at root level 2. For each matrix size it measures:

- `q = 1` at error depths 0 and 1;
- a small multiband cluster, up to `q = 4`, at depth 1;
- `q = N/2` at depth 1, which exercises a larger retained active space;
- for the largest non-quick matrix, the small cluster at depth 2.

The constructed cluster width is stored as `target_bands`. Certification may
reduce the actual uncertain space on individual simplices, so
`charge_initial_active_dimension_sum / charge_root_simplices` is the observed
mean root width. The same quadratic enclosure handles every active dimension.

The terminal table reports total milliseconds and full-eigensolve equivalents
per root simplex. One equivalent is the faster measured full `zheevd` path for
the same matrix size. It also reports actual Hamiltonian evaluations, reduced
eigensystems, quadratic Schur controls, and temporary and terminal simplices.
Depth changes
the fixed microsimplex tree; it is not AdaptiveSimplex mesh refinement.

Each root builds one quadratic matrix model and a sampled remainder from
quartic lattice probes. Safe states are eliminated using a diagonal anchor
block. Temporary cells restrict this fixed polynomial; they do not evaluate
the Hamiltonian again. The root shares one center frame between its block
sign proof and affine bounds. Only retained terminal cells integrate charge
intervals and cut disagreement. See [the design](occupation-enclosure.md).

Machine-readable charge fields include:

- `error_depth` and `stopping_error`;
- `density_cut_error`, the sampled sum of terminal shifted-cut widths and
  exact root-versus-child affine-cut disagreements;
- `charge_root_simplices`, `charge_micro_simplices`, and
  `charge_terminal_simplices`;
- `charge_hamiltonian_evaluations`;
- center-frame and sector-margin eigensystem counts;
- quadratic Schur-control and Schur-reduction counts;
- initial and terminal active-dimension sums and the minimum reduced dimension.

`lapack_equivalents_per_operation` is the full timed mesh pass divided by one
full eigensolve. Divide it by `charge_root_simplices` to recover the value shown
as `eig/root` in the terminal summary.

## Accuracy and failure benchmark

The maintained public-API accuracy sweep is reproducible with:

```sh
pixi run benchmark-charge-error
```

It compares current-mesh charge and the sampled estimate with dense off-dyadic
references for 1D scalar convergence; 2D avoided and clustered systems; systems
with nonzero active-safe Schur coupling embedded through 128 bands; varying
active-cluster sizes; and visible versus exact dyadic aliasing. It writes JSON and a
scalar convergence plot under `build/benchmarks/`. Reference-grid differences
are recorded, so these are accuracy diagnostics rather than proofs. Its
single-shot wall times are also diagnostic; use the repeated C++ benchmark for
stable performance comparisons.

## General measurements

The ordinary presets retain these benchmark families:

- reused-workspace and current-wrapper LAPACK eigensolves;
- cumulative model evaluation, trusted Hamiltonian dispatch, eigensystem, and
  cache insertion costs;
- tight-binding evaluation and eigensystem costs for several hopping counts;
- direct current-mesh charge and adaptive charge integration at the default
  temporary polynomial subdivision depth 2;
- full-matrix and selected-component density integration through one grouped
  contraction kernel;
- adaptive Fermi-surface extraction;
- controlled root-mesh evaluation and classification scaling;
- the separate occupation-bounds benchmark executable.

End-to-end results record total time, new spectral vertices, actual simplex
visits, refinements, time per vertex and visit, and LAPACK equivalents per
vertex and visit. Charge results additionally carry the occupation enclosure
counters above. The current-mesh charge pass performs only linear-simplex
integration; the adaptive charge pass forces AdaptiveSimplex preview depth zero
and refines using the distance to the sampled occupation interval.

The benchmark excludes `SpectralMesh` construction from timed regions. Reference
LAPACK matrices are prepared outside the timer. Raw timings should only be
compared on the same runner; LAPACK-equivalent ratios still require the same
LAPACK provider and thread configuration.

Selected density requests are grouped by matrix element. For each simplex
vertex, the band contraction for a unique `(row, column)` pair is evaluated
once and reused for every requested lattice-vector phase. Its contraction cost
therefore scales with the number of unique requested matrix elements rather
than with the number of returned components.

## CI use

On a fixed runner:

1. build in Release mode;
2. run the CI preset with one BLAS/LAPACK thread;
3. retain the JSON file as an artifact;
4. compare cases by
   `(name, ndim, ndof, root_level, target_bands, error_depth)`;
5. compare deterministic operation counters exactly;
6. report timing changes above 5% and fail only after a repeated change above
   10%.

Small matrices are dominated by fixed overhead and clock noise. Apply timing
thresholds to medium and large cases first, while retaining small cases to
track scaling and control flow.
