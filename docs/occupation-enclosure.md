# Shared occupation enclosure

## Goal and contract

Use one matrix model for charge error and gap classification. The reported
charge remains the mesh's piecewise affine band integral. The enclosure bounds
its error and drives refinement. Both `integrate_charge` and `fermi_surface`
use this implementation; there is no algorithm selector.

A sign claim is conditional on a uniform interpolation remainder. By default
that remainder is sampled. Explicit uniform bounds can be supplied to
`occupation_enclosures`; a positive surface `curvature_bound` also provides a
uniform allowance. Arbitrary features between samples can still be missed.
The standalone vertex-only `certify_simplex` is a low-level affine-matrix
primitive with an explicit remainder contract; adaptive calculations no longer
use it to classify the Hamiltonian.

## Algorithm

1. On each simplex, interpolate `K = H - mu I` with a quadratic Bernstein matrix
   polynomial `K2`, evaluating the Hamiltonian at vertices and edge midpoints.
   Cached vertex eigenvectors still provide the anchor frame. Direct evaluation
   removes cubic-cost spectral reconstruction, at the cost of repeated vertex
   Hamiltonian calls; there is no additional matrix cache.
2. Enumerate the degree-four barycentric lattice `alpha/4`, with nonnegative
   integer `alpha` summing to four, in any dimension. Skip the vertices and edge
   midpoints used by the interpolant. Cache these geometry-only probe weights
   by vertex count. Set `eta = min(2**dimension,32)*max_sample_defect + roundoff`.
   There is no per-dimension probe list or algorithm selector.
   Subtract the polynomial directly into each evaluated probe matrix and reuse
   the norm workspace, without retaining extra full matrices.
3. In a vertex eigenbasis, establish negative and positive safe sectors by
   testing every Bernstein control with allowance `eta`. Gershgorin bounds
   precede matrix factorizations. A Cholesky failure at pivot `k` retains the
   first `k-1` states; the positive suffix is visited in reverse order.
   This replaces a binary search with at most one factorization per control
   and sign, followed by at most one eigenvalue computation for its margin.
   Their minimum margin is `Delta`. Any states
   that cannot be separated remain active, up to the full matrix.
4. Write the active/safe blocks as `A,B,D`, freeze the anchor's safe block
   `D0`, and linearly interpolate the vertex coupling as `B1`. Set
   `X = D0^-1 B1` and form the quadratic reduced model
   `P = A2 - B1† D0^-1 B1`.
5. Bound `b >= ||B-B1||`, `d >= ||D-D0||`, and `x >= ||X||` by Bernstein
   controls. Use
   `epsilon = eta + 2*b*x + d*x*x + (b+d*x)^2/Delta`.
6. In each temporary cell's center eigenbasis, bound diagonal curvature and
   off-diagonal row sums. These affine band bounds determine strict occupation
   bounds and whether to subdivide. The root reuses this same center basis for
   its block sign proof. Only retained terminal cells integrate the shifted
   affine cuts and their disagreement with the reported cuts. Restrict the
   original polynomial for each bisection, preserving the frame used to build
   child models. Subdivision requires no new Hamiltonian samples and never
   reduces `epsilon`.
   An edge bisection copies unchanged controls and updates only controls incident
   on the replaced vertex by midpoint averages. This rule applies in every
   dimension. A center that is already diagonal borrows the original controls;
   only a changed basis owns a rotated polynomial.
   Surface classification consumes only the strict occupation bounds from this
   same traversal; it does not integrate charge or cut disagreement.

The Schur identity `S = Y - F† D^-1 F`, where
`Y = A - B†X - X†B + X†DX` and `F = B-DX`, gives step 5. With a uniform safe
gap and a smooth local Hamiltonian, `b=O(h^2)`, `d,x=O(h)`, `eta=O(h^3)`.
The model allowance is cubic; its solve-residual term is quartic. The reported
affine-band charge generally remains second order.

The sampling factors are verified with exact rational Bernstein subdivision in
[`verify_remainder_factor.py`](../benchmarks/verify_remainder_factor.py).
For **any matrix polynomial of degree at most four**, the full interpolation
remainder is bounded by these samples in every dimension, in exact arithmetic.
The residual vanishes at the quadratic nodes; its other quartic lattice values
multiply the corresponding Lagrange cardinal polynomials. Bound the sum of their
absolute values. Exact subdivision gives `14/9 <= 2`, `4 <= 4`, `71/9 <= 8`,
and `142/9 <= 16` in dimensions one through four. For any higher dimension,
each degree-four Bernstein coefficient involves at most four vertices;
cardinal polynomials supported outside those vertices contribute zero. The
largest absolute coefficient row sum on four vertices is 32, so 32 bounds
every dimension. Together these establish the single formula `min(2**d,32)`.
This does not establish a uniform bound for general functions.
Roundoff allowances are numerical safeguards, not interval arithmetic proofs.

## Reusing safe-subspace certificates

One mesh owns scalar proof records keyed by its immutable simplex ids. The
Hamiltonian must stay fixed while that mesh is used, as for the vertex spectra.
A record contains the safe ranks, separate occupied/empty margins at `mu0`,
the interpolation allowance, and the bounds on coupling curvature and safe-block
variation. It contains no Hamiltonian, eigenvector, or reduced matrix. Retired
cells retain small records alongside the existing geometry; children have new ids.
Cache storage is constant per tested cell, independent of band count.

At a new `mu`, the margins become `m_minus + (mu-mu0)` and
`m_plus - (mu-mu0)`. Empty subspaces have infinite margins. Subtract a roundoff
allowance for the changed diagonal-subtraction scale. If either margin ceases
to be positive, or the supplied remainder contract changes, rebuild the proof.
Otherwise reuse the ranks and sampled remainder. A fully safe cell needs no
new Hamiltonian evaluations. A partially safe cell evaluates the quadratic
nodes and projects only the active columns, costing `O(N^2 q)` rather than
`O(N^3)` for full rotations. Recompute its small Schur model, inverse of the
anchor safe diagonal, and error allowance at the new chemical potential.
The bounds on `B-B1` and `D-D0` remain valid because scalar shifts cancel.

This also applies to cut cells: only their safe subspaces are retained; their
active occupation, charge interval, and cut discrepancy are recomputed.
Reuse can retain fewer safe states than a fresh proof, making an enclosure
wider. Measurements must compare accuracy and refinement work, not only time.
Queries on the same mesh are serialized, as for its existing geometry/cache.

## Charge, gaps and flat bands

For reported local charge `Q`, the indicator is
`max(abs(Q-Qlower), abs(Qupper-Q))`; sum these local indicators. The density-cut
indicator also measures spatial disagreement of the affine occupied regions,
so cancellation of total charge cannot hide displaced cuts.

When the strict occupation bounds agree on a count `n`, every ordered band has
a known occupation throughout the cell. Compare the reported cuts directly
with those constants: sum `volume - reported_volume` for the occupied bands
and `reported_volume` for the empty bands. Reuse the fixed count even when the
block sign proof is tighter than the affine row bounds. Otherwise a gapped
cell can retain a spurious uncertainty strip in its density-cut indicator.
Keep the ordinary cut-disagreement calculation for unresolved cells; equality
of integrated charge alone does not establish fixed occupation. Tolerance
rounding can still give a nonzero indicator in a cell with fixed occupation.

Smooth projector quadrature is a separate error source. MeanFi uses its
requested density tolerance independently of this cut indicator. A resolved
occupation does not imply constant projectors or remove their quadrature work.

Apply the mesh's level tolerance once to each root band cut: replace relative
vertex energies within that tolerance by zero. Charge, the cut indicator and
density integration then refer to this same affine field. Temporary enclosure
cells and density-only children restrict the field without applying a new
level tolerance. Include disagreement from bands removed as safely occupied or
empty, since their reported cut can still have half occupation at a larger user
tolerance. The matrix allowance and outward charge bounds keep their separate
roundoff tolerance; a loose mesh tolerance must not weaken the gap proof.

Scalar affine cuts share one occupation rule, using vertex energies relative
to `mu` for level classification, including structurally constant bands.
Sorted energies are knots of the projected uniform simplex distribution.
Its cumulative fraction is a convex recurrence on successive knot intervals;
differentiate that recurrence to obtain the reported charge's slope in the same
pass. Snapped knots stay at zero while off-level knots move by `-dmu`. Thus
a tolerance plateau has zero slope, including a rounded band endpoint. The
derivative describes a fixed snapping classification; it is not defined at a
threshold where that classification changes. With zero level tolerance, use
the left derivative at exact knots. Equal knots need no division by zero and
nearby distinct knots remain distinct, avoiding cancellation for clustered
energies.
These scalar calculations use `O(d^2)` work and `O(d)` storage and do not build
barycentric moments. Density-weight integration still computes the moments
it needs through AdaptiveSimplex.

For sorted relative energies `t_i`, the cumulative fraction on knots `i..j`
is zero if `t_i >= 0`, one if `t_j <= 0`, and otherwise
`F[i,j] = (-t_i*F[i,j-1] + t_j*F[i+1,j])/(t_j-t_i)`.
On a straddling interval, set `a=-t_i/(t_j-t_i)` and `b=t_j/(t_j-t_i)`.
Its derivative is `D[i,j]=a*D[i,j-1]+b*D[i+1,j]+(F[i,j-1]-F[i+1,j])/(t_j-t_i)`.
Full and empty intervals have zero derivative while their snapping class stays
fixed. Fraction-only queries omit derivative work. Half occupation is handled
separately when the entire simplex lies on the level.

Strict occupation bounds are separate from integrated charge endpoints. A
constant tight-binding matrix has no interpolation error, but its numerical
eigenvalues can still be uncertain. At construction, cache an eigenvalue
roundoff allowance `64*N*machine_epsilon*||H||_1`, independently of the mesh
level tolerance. For a Hermitian matrix the maximum absolute column sum bounds
its spectral norm. Eigenvalues within this allowance of `mu` contribute
occupation intervals `[0,1]`, including when the cached value equals `mu`.
Charge and cut-error bounds retain that uncertainty. This is a numerical
safeguard, not a verified eigensolver error bound.

A structurally diagonal constant matrix has exact stored eigenvalues, so its
allowance is zero. Read and sort its diagonal directly, with the corresponding
permutation basis, avoiding eigensolver scaling at large energy ranges.
An exactly on-level diagonal entry has half occupation and
does not certify a strict gap. For a resolved nonzero energy, rounding to the
level still contributes its discrepancy to both error indicators. Matrix
structure and scale are checked once; enclosure queries read the cached
allowance and eigenvalues, without rescanning or diagonalizing the matrix.
Constant charge integration never refines: if its irreducible error exceeds
the target, it raises a nonconvergence error immediately after evaluating the
current mesh. A constant callable is not assumed exact from finitely many
samples.
Surface classification also stops on the current cells of a constant matrix:
a resolved spectrum is gapped, while an unresolved spectrum is inconclusive.
A constant zero-energy band fills the cell and cannot be represented by a
codimension-one surface. An empty extracted mesh must not certify its absence.

Surface classification first compares cached vertex occupation intervals.
Their lower counts exclude near-level bands; their upper counts include them.
If the largest lower count exceeds the smallest upper count, continuity forces
a crossing and no matrix model is needed. Mere proximity to the level is not
such a witness. All other cells retain the full quartic model check, including
hidden pockets and ambiguous contacts. This rule is dimension independent.

For surface classification, a known affine-interpolation error `e` implies a
quadratic remainder at most `(1+2*d/(d+1))*e`: the edge controls of `K2-L` are
at most `2e`, and their Bernstein weights sum to at most `d/(d+1)`.

## Code and verification

- `occupation/model.cpp`: interpolation, certificate reuse and Schur allowance.
- `occupation/sectors.cpp`: safe-subspace sign tests and their margins.
- `occupation/certificate_cache.h`: scalar proof records owned by each mesh.
- `occupation/probes.h`: dimension-general quartic lattice and remainder factor.
- `occupation/polynomial.h`: matrix polynomial restriction and frame changes.
- `occupation/enclosure.cpp`: charge intervals and strict sign classification.
- `occupation/cut_disagreement.h`: occupied-volume disagreement.
- `occupation/affine_cut.h`: scalar affine charge, derivative and level classification.
- `integration/integration.cpp`: one adaptive charge calculation.

The obsolete recursive estimator, temporary spectral caches, projected hopping
backend and method switches were removed. Cached anchor eigenvalues avoid a
redundant rotation, and sector margins are reused without copying a matrix
when row bounds suffice.

Store each symmetric Bernstein control only once. Matrix norm and sector row
bounds scan contiguous columns, including both matrix triangles. For fixed
dimension and subdivision depth, model construction is `O(N^3)` with `O(N^2)`
storage; the sector search has no logarithmic number of factorizations. This
does not bound the number of persistent refinements needed by a physical model.

Tests compare exact interval/disk/sphere volumes, cubic Schur errors, complex
multiband models, cubic/quartic pockets and adversarial quartic interior
bubbles. A deliberately unsampled smooth bump remains a documented failure.
Tolerance regressions compare affine diagonal densities with their exact
integrals in 1D/2D, including cancellation of total charge and removed safe
bands. Density-only refinement must preserve the root particle number to
`1e-12`; its Fourier integral is checked against the analytic value to `1e-5`,
the requested quadrature accuracy. Bisection tests evaluate both parent and
child matrix polynomials at the same physical points in 1D through 8D, with
an absolute tolerance of `2e-14` for unit-sized controls.
Constant-spectrum regressions use rank-one matrices with exact zero bands,
real and complex entries, and scales from `1e-8` to `1e8`. Their enclosures
must contain the analytic half-filled charge without asserting a gap, and
nearby positive/negative perturbations must remain enclosed too. Separate
diagonal cases retain exact eigenvalues and particle number, including widely
separated energies. Irreducible constant errors must leave the mesh unchanged.
Derivative tests use exact affine references and central differences away from
snapping thresholds, through dimension 12 and across bandwidths from `1e-10`
to `1e10`. The scaled exact-reference tolerance is `2e-12`; finite differences
allow `2e-9` for subtraction and truncation error. Surface tests in 1D through
5D require zero additional Hamiltonian calls for cached strict crossings and
retain probing for near-level gaps, contacts and hidden quartic pockets.
For measurements and separate-build reproduction, see the
[MeanFi report](https://gitlab.kwant-project.org/qt/meanfi/-/blob/codex/occupation-enclosure/docs/occupation-enclosure.md).
