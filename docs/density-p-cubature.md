# Density cubature on the charge mesh

`SpectralMesh.integrate_density_components_p` is an additive alternative to
`integrate_density_components`. The latter retains its existing h-refinement
behavior, including `preview_depth=0`, for comparisons and existing callers.
The new routine never splits simplices. Call `integrate_charge` first to
resolve occupations; at fixed filling use the converged chemical potential.

For a simplex of dimension d and volume V, the initial approximation is

```
Q1 = V/(d+1) * sum(f(vertex))
Q2 = V/((d+1)*(d+2)) * sum(f(vertex)) + V*(d+1)/(d+2)*f(centroid)
```

Q2 is exact through total degree two (degree three in 1D). This follows from
the normalized barycentric moments E[lambda_i] = 1/(d+1) and
E[lambda_i lambda_j] = (1+delta_ij)/((d+1)(d+2)). It need not improve an
arbitrary integrand. Its initial local indicator is max(abs(Q2-Q1)).

When this is insufficient, the routine promotes the cell with the largest
indicator to degrees 3, 5, ..., `max_degree` (default and maximum 21).
The Grundmann–Moeller rule of index s has degree 2s+1. For each t=0,...,s
and each nonnegative integer tuple beta summing to t, its nodes and
normalized weights are

```
lambda_i = (2*beta_i+1)/(d+1+2*t)
w = (-1)^(s-t) * d! * (d+1+2*t)^(2*s+1)
    / (2^(2*s) * (s-t)! * (d+1+s+t)!)
```

These interior node sets are nested. Coincident nodes use exact reduced
integer barycentric keys. The centroid is reused at every level; vertices
have zero weight in the higher rules, but supply the initial error estimate.
Coefficients are constructed once per dimension and requested degree per
integration call. No fitted tables or external cubature dependency is needed.
See [Grundmann and Moeller (1978)](https://doi.org/10.1137/0715019).

Each interior node evaluates sum_b f_b P_b(k) times the real-space Fourier
phase, where f_b is that band's linear-simplex occupied volume fraction.
For cut bands, let M_bi be the existing barycentric occupied moments and
F_b(v_i) the projector times Fourier phase at vertex i. The returned integral is

```
Q_p + sum_bi [M_bi - f_b*V/(d+1)] F_b(v_i).
```

Equivalently, integrate the vertex-linear interpolant over the occupied region
exactly, and apply fraction-weighted p-cubature only to its nonlinear remainder.
This removes the lowest-order occupation/projector correlation error and is
exact when F_b is affine on the simplex. The same moments and vertex spectra
are already available, so there are no additional Hamiltonian evaluations or
diagonalizations. Partial cells require a small additional component contraction;
full, empty and on-level bands need no correction. The correction is independent
of degree, so it cancels from consecutive p differences. The onsite trace remains
the linear-simplex charge, and exactly on-level bands retain half occupation.
Higher-order occupation correlation and charge-geometry errors remain outside
the cubature estimate. A degeneracy between bands with unequal frozen fractions
can also make individual-band projectors nonsmooth.

Only requested complex components are retained at interior nodes. Vertex
eigensystems come from the shared charge cache; interior eigensystems are
released immediately after projection. Samples are reused across polynomial
orders within a call, not across calls, chemical potentials or component
selections. The geometry and its persistent vertex cache do not gain any
interior cubature nodes.

For complex component correction vectors delta_sigma between successive rules,
the stopping estimate reuses the h-adaptive density policy:

```
max(sqrt(sum_sigma ||delta_sigma||_infinity^2),
    ||sum_sigma delta_sigma||_infinity,
    sum_sigma roundoff_sigma).
```

The coherent term retains systematic error; the statistical term guards against
cancellation between cells. This is less conservative than summing local norms,
but remains empirical and can miss aliased features. AdaptiveSimplex's existing
RefinementQueue selects the largest local indicators. Signed higher-order weights
are accumulated in long double, with a floating-point floor scaled by their
absolute weight sum. Results are not guaranteed positive semidefinite.

Promotions are serial by default. If OpenMP is available, the requested OpenMP
thread count exceeds one, and the Hamiltonian has at least 32 orbitals, batches
of up to 16 cells run in parallel. The size threshold reflects the measured
scheduling overhead for small eigensystems. Cells, accumulators and exceptions
are private to each worker; global error updates and queue changes are serial.
The batch cannot exceed the remaining promotion budget. One-cell batches use a
serial fast path. Python callback exceptions are rethrown on the calling thread.
Only density_p.cpp is compiled with OpenMP; charge threading stays unchanged.
Build with `FERMISIMPLEX_ENABLE_DENSITY_OPENMP=OFF` to disable this optional path;
missing OpenMP also falls back to serial compilation. Control native threads via
OpenMP or threadpoolctl; MeanFi's `num_threads=1` default keeps promotions serial.
OpenMP-based BLAS libraries may share the same runtime/thread limit, so benchmark
thread settings should be recorded and nested oversubscription avoided.

`max_refinements` bounds promotions after initial Q2 evaluation;
`max_degree=2` limits evaluation to the requested vertices-plus-centroid pair.
Neither a degree cap nor a budget exhaustion is success unless the estimated
tolerance is met. `stats.refinements` remains zero;
`stats.p_refinements` counts order promotions, `stats.max_degree` records the
largest degree used, and `stats.cubature_evaluations` counts new interior
spectra. `stats.evaluations` also includes missing charge-mesh vertex spectra.
The legacy `cached_vertices` count continues to describe persistent spectra.
