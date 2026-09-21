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

Each node evaluates sum_b f_b P_b(k) times the real-space Fourier phase,
where f_b is that band's linear-simplex occupied volume fraction, held fixed
on the cell. The same formula is used for bulk and cut simplices. It
preserves the onsite trace of the linear-simplex charge. Empty cells need
no interior evaluations. An exactly on-level band has half occupation.

Only requested complex components are retained at interior nodes. Vertex
eigensystems come from the shared charge cache; interior eigensystems are
released immediately after projection. Samples are reused across polynomial
orders within a call, not across calls, chemical potentials or component
selections. The geometry and its persistent vertex cache do not gain any
interior cubature nodes.

The stopping estimate is the sum over cells of the maximum component-wise
absolute difference between consecutive rules, with a floating-point floor
scaled by the absolute weight sum. There is no cancellation of cell errors.
Signed higher-order weights are accumulated in long double. They can amplify
roundoff and the result is not guaranteed positive semidefinite.
The estimate is empirical, not a rigorous error bound, and can miss aliased
features. It excludes both charge/occupation error and covariance between
the cut occupation and the varying projector/Fourier phase. Tightening p
alone cannot remove this cut error. A degeneracy between bands with unequal
frozen fractions can also make individual-band projectors nonsmooth.

`max_refinements` bounds promotions after initial Q2 evaluation;
`max_degree=2` limits evaluation to the requested vertices-plus-centroid pair.
Neither a degree cap nor a budget exhaustion is success unless the estimated
tolerance is met. `stats.refinements` remains zero;
`stats.p_refinements` counts order promotions, `stats.max_degree` records the
largest degree used, and `stats.cubature_evaluations` counts new interior
spectra. `stats.evaluations` also includes missing charge-mesh vertex spectra.
The legacy `cached_vertices` count continues to describe persistent spectra.
