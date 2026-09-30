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
   polynomial `K2`, using cached vertex eigensystems and edge midpoints.
2. Probe the remaining degree-four lattice nodes: quarter edges, the three
   permutations of `(1/2,1/4,1/4)` on each triangular face, and the tetrahedron
   center. Set `eta = 2**dimension * max_sample_defect + roundoff`.
3. In a vertex eigenbasis, establish negative and positive safe sectors by
   testing every Bernstein control with allowance `eta`. Gershgorin bounds
   precede matrix factorizations. A Cholesky failure at pivot `k` retains the
   first `k-1` directions; the positive suffix is visited in reverse order.
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
6. Restrict this fixed polynomial by temporary bisection. At terminal cells,
   use its center eigenbasis, bound diagonal curvature and off-diagonal row
   sums, and integrate affine cuts shifted by those bounds and `epsilon`.
   This gives charge endpoints and strict occupation bounds. Subdivision
   requires no new Hamiltonian samples and never reduces `epsilon`.

The Schur identity `S = Y - F† D^-1 F`, where
`Y = A - B†X - X†B + X†DX` and `F = B-DX`, gives step 5. With a uniform safe
gap and a smooth local Hamiltonian, `b=O(h^2)`, `d,x=O(h)`, `eta=O(h^3)`.
The model allowance is cubic; its solve-residual term is quartic. The reported
affine-band charge generally remains second order.

The sampling factors are verified with exact rational Bernstein subdivision in
[`verify_remainder_factor.py`](../benchmarks/verify_remainder_factor.py).
For **any matrix polynomial of degree at most four**, the full interpolation
remainder is bounded by these samples in dimensions one through three, in exact
arithmetic. The verified cardinal-function bounds are `14/9 <= 2`, `4 <= 4`,
and `71/9 <= 8`. This does not establish a uniform bound for general functions.
Roundoff allowances are numerical safeguards, not interval arithmetic proofs.

## Charge, gaps and flat bands

For reported local charge `Q`, the indicator is
`max(abs(Q-Qlower), abs(Qupper-Q))`; sum these local indicators. The density-cut
indicator also measures spatial disagreement of the affine occupied regions,
so cancellation of total charge cannot hide displaced cuts.

Strict occupation bounds are separate from integrated charge endpoints. An
exactly constant tight-binding matrix has an exact charge, including half
occupation at a flat band. Its charge interval can collapse while its strict
gap test remains inconclusive. This structural constant case prevents endless
refinement of a known flat band. A constant callable is not assumed exact from
finitely many samples.

For surface classification, a known affine-interpolation error `e` implies a
quadratic remainder at most `(1+2*d/(d+1))*e`: the edge controls of `K2-L` are
at most `2e`, and their Bernstein weights sum to at most `d/(d+1)`.

## Code and verification

- `occupation/model.cpp`: interpolation, safe sectors and Schur allowance.
- `occupation/polynomial.h`: matrix polynomial restriction and frame changes.
- `occupation/enclosure.cpp`: charge intervals and strict sign classification.
- `occupation/cut_disagreement.h`: occupied-volume disagreement.
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
For measurements and separate-build reproduction, see the
[MeanFi report](https://gitlab.kwant-project.org/qt/meanfi/-/blob/codex/occupation-enclosure/docs/occupation-enclosure.md).
