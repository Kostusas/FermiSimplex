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

Use the focused mode when changing recursive charge-error estimation:

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
- `q = N/2` at depth 1, which exercises the root large-active-space gate
  wherever certification retains that many uncertain states;
- for the largest non-quick matrix, the small cluster at depth 2.

The constructed cluster width is stored as `target_bands`. Certification may
reduce the actual uncertain space on individual simplices, so
`charge_initial_active_dimension_sum / charge_root_simplices` is the observed
mean root width. `charge_conservative_fallbacks` records how often the root
gate or a failed Schur layer selected the sampled occupation-range fallback.

The terminal table reports total milliseconds and full-eigensolve equivalents
per root simplex. One equivalent is the faster measured full `zheevd` path for
the same matrix size. It also reports actual Hamiltonian evaluations, reduced
eigensystems, corrected Schur evaluations, micro-simplices, and fallbacks.
Depth changes
the fixed microsimplex tree; it is not AdaptiveSimplex mesh refinement.

Each Schur layer stores a dense anchor resolvent and reuses two thin work
buffers. At a terminal node, an unchanged certificate is reused when its
chemical-potential interval contains the complete shifted-volume radius. These
are algebraic and control-flow optimizations; they do not change the sampled
charge interval.

Machine-readable charge fields include:

- `error_depth` and `stopping_error`;
- `charge_root_simplices`, `charge_micro_simplices`, and
  `charge_terminal_simplices`;
- `charge_hamiltonian_evaluations`;
- full, reduced, and norm eigensystem counts;
- corrected Schur-evaluation and Schur-reduction counts;
- `charge_conservative_fallbacks` and Schur-failure counts;
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
with nonzero active-safe Schur coupling embedded through 128 bands; root-gate
boundaries; and visible versus exact dyadic aliasing. It writes JSON and a
scalar convergence plot under `build/benchmarks/`. Reference-grid differences
are recorded, so these are accuracy diagnostics rather than proofs. Its
single-shot wall times are also diagnostic; use the repeated C++ benchmark for
stable performance comparisons.

## General measurements

The ordinary presets retain these benchmark families:

- reused-workspace and current-wrapper LAPACK eigensolves;
- cumulative model evaluation, Hamiltonian validation, eigensystem, and cache
  insertion costs;
- tight-binding evaluation and eigensystem costs for several hopping counts;
- fixed-mesh and adaptive charge integration with recursive error depth 1;
- adaptive Fermi-surface extraction;
- controlled root-mesh evaluation and classification scaling;
- the separate occupation-bounds benchmark executable.

End-to-end results record total time, new spectral vertices, actual simplex
visits, refinements, time per vertex and visit, and LAPACK equivalents per
vertex and visit. Charge results additionally carry the recursive estimator
counters above. The current-mesh charge pass uses no AdaptiveSimplex preview;
the adaptive charge pass also forces preview depth zero and refines using the
summed charge-error estimate.

The benchmark excludes `SpectralMesh` construction from timed regions. Reference
LAPACK matrices are prepared outside the timer. Raw timings should only be
compared on the same runner; LAPACK-equivalent ratios still require the same
LAPACK provider and thread configuration.

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
