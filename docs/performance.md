# Performance measurement

FermiSimplex measures computational cost relative to the same complex
Hermitian LAPACK eigensolve used by the library. The benchmark suite is C++
only; Python startup, callbacks, and bindings are deliberately excluded.

## Build and run

Use an optimized build and one BLAS/LAPACK thread:

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
`build/cpp-benchmarks/benchmarks/fermisimplex-performance.json`. The
equivalent Pixi command is:

```sh
pixi run benchmark-cpp
```

The executable also supports direct use:

```sh
build/cpp-benchmarks/cpp/fermisimplex_performance_benchmark \
  --preset quick \
  --output result.json
```

Use the focused larger-matrix charge benchmark when only end-to-end charge
scaling and its phase breakdown are needed:

```sh
build/cpp-benchmarks/cpp/fermisimplex_performance_benchmark \
  --preset ci --only charge-scaling --output charge-scaling.json
```

Available presets are `quick`, `ci`, and `full`. Use `quick` only as a smoke
test. The `ci` preset has stable cases and sample counts intended for version
tracking. `full` adds larger matrices and more tight-binding terms.

Each preset performs an untimed LAPACK warm-up before collecting samples. This
reduces process-start, dynamic-library, and CPU-frequency effects without
mixing warm-up work into reported timings.

## Measurements

The headline metric is total LAPACK-equivalents per newly evaluated vertex:
`total operation time / (new vertices * LAPACK time)`.

This includes vertex evaluation, diagonalization, caching, every simplex-rule
or classification call, refinement bookkeeping, and result construction. It
excludes `SpectralMesh` construction. The target is therefore stated directly:
`total_lapack_equivalents_per_vertex <= 2`.

The secondary metric divides the same total time by actual simplex visits.
Production charge cases use preview depth zero, so each estimate visits an
active simplex once. Explicit diagnostic cases with positive preview depth
also count preview contributions. For Fermi surfaces, visits count every
classified frontier simplex. Neither metric uses the final active-simplex
count as a proxy for work.

The terminal summary puts the primary metric first, reports the measured
LAPACK time beside it, and marks whether each case meets the two-solve target.
It also prints per-vertex pipeline and per-simplex phase scaling across every
matrix size in the selected preset. Visit counts and refinement diagnostics
are shown separately for the largest simplex matrix size.

The end-to-end workloads are:

- `charge_current_mesh_total`: one complete previewless charge estimate on a
  fixed level-2 mesh of the nonlinear crossing model. Every active simplex is
  certificate-selected for projected-error estimation, so this includes the
  exact center solve, edge projections, and occupation-shell conversion;
- `charge_adaptive_total`: converged previewless integration of the same
  crossing model from level 1 to the requested sampled stopping error;
- `fermi_surface_total`: adaptive 2D Fermi-surface refinement of a crossing
  dense model down to the requested feature size.

Supporting diagnostic measurements report:

- `lapack_reused_workspace`: `zheevd` with one workspace query and reused work
  arrays. Input matrices are prepared outside the timer, so this measures the
  LAPACK call itself;
- `lapack_current_wrapper`: the current FermiSimplex wrapper, including its
  per-call workspace query and allocations;
- `lapack_reference_best`: the faster of those two measured LAPACK paths for
  the current matrix size and LAPACK provider;
- cumulative vertex-pipeline stages: model evaluation, evaluated-and-validated
  Hamiltonian, complete eigensystem, and eigensystem-cache insertion. Subtract
  adjacent stages when an isolated incremental cost is needed;
- direct per-simplex timings for certification, the eigenvalues-only exact
  center, complete sampled projected error, occupation-shell conversion,
  ordinary band integration, and complete charge integration, using the same
  nonlinear crossing model and curvature as the overall charge workload;
- a profiled adaptive charge run that records vertex eigensystems,
  certificates, sampled projected error, occupation-shell conversion, band
  integration, and adaptive-framework time over the actual varying-width
  simplex visits. The terminal table also reports the maximum certificate-
  selected width encountered;
- Fermi classification per simplex;
- controlled root-mesh evaluation and classification with deterministic
  vertex and simplex counts;
- the complete 3D projected-error estimator for a 60-band Hamiltonian and
  selected widths $m=1,2,4,6,8,10,12,16,32,60$, including the eigenvalues-only
  exact center and all six principal-angle projected edge-midpoint probes.
  `--only projected-edges` runs just this focused diagnostic.

Within one charge integration, projected edge estimates are memoized by the
two endpoint vertex IDs and the selected band interval. Neighboring simplices
therefore reuse a shared edge calculation. When the selected interval is the
full $N$-band space, the edge projection is unitarily equivalent to the full
Hamiltonian, so the implementation skips frame construction and directly
performs an eigenvalues-only $N\times N$ solve. Both paths are mathematically
identical to the estimator described in [Mathematics](mathematics.md).

Diagnostic results include `lapack_equivalents_per_operation`, normalized to
`lapack_reference_best`. End-to-end results additionally include total time,
new vertices, actual simplex visits, refinements, time per vertex and visit,
and LAPACK-equivalents per vertex and visit. Both raw LAPACK paths remain in
the output because small-matrix behavior can depend on the LAPACK provider.

The JSON schema uses stable benchmark names and records the commit, dirty-tree
state, compiler, build type, system, LAPACK linkage, thread settings, counts,
charge stopping and certified errors, certificate-status counts, target status,
median, range, and median absolute deviation. Compare
`median_ns_per_operation` only between runs on the same runner. The
LAPACK-equivalent ratios are more portable, but still require the same LAPACK
implementation and thread configuration.

## CI use

Run performance jobs on a fixed or dedicated runner. Shared cloud runners can
be useful for collecting artifacts, but their timing variance makes strict
regression thresholds unreliable.

For each revision:

1. build in `Release` mode;
2. run `fermisimplex_run_performance_benchmark`;
3. retain the JSON file as a CI artifact;
4. compare cases by the tuple
   `(name, ndim, ndof, hopping_terms, root_level, target_bands)`;
5. graph `total_lapack_equivalents_per_vertex` for end-to-end workloads;
6. use per-simplex and phase timings to diagnose changes;
7. flag a regression only when it exceeds both a relative threshold and the
   observed run-to-run dispersion.

A practical initial policy is to report changes above 5% and fail only after a
repeat confirms a change above 10%. Counts such as vertices and simplices are
deterministic and may use exact comparisons.

Very small absolute timings remain sensitive to clock resolution, CPU state,
and allocator behavior even when operations are batched. Apply regression
thresholds to the medium and large matrix cases first; retain the smallest
cases primarily to track scaling and fixed overhead.
