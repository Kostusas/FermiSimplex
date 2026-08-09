# Charge-error algorithm and ownership

Charge integration reports the piecewise-linear charge on the persistent
spectral mesh. A separate sampled calculation estimates the local error of
that value and drives adaptive refinement. This document describes the
implementation structure of that estimator; the derivation and its
limitations are in the
[mathematics guide](mathematics.md#recursive-active-space-charge-error-estimator).

Here, a **root simplex** means one simplex of the persistent adaptive mesh for
which an error estimate is being calculated. Its recursive microsimplices are
temporary and never enter that mesh.

## Ownership

The estimator separates model-wide data from one root calculation:

```text
ChargeErrorEstimator
└── ModelBackend                         integration lifetime
    ├── spectral mesh and chemical potential
    └── immutable tight-binding acceleration data

estimate(root simplex)
└── RootErrorWorkspace                  one-root lifetime
    ├── base PointCache
    ├── base EffectiveModel (Base kernel)
    └── recursive ActiveSpaceReduction
        └── reduced EffectiveModel      lexical branch lifetime
            ├── its own PointCache
            └── DenseSchur or ProjectedTB kernel
```

`RootErrorWorkspace` is created on the stack at the start of `estimate` and
is destroyed on every return, including exceptions. All recursive
microsimplices, vertices, and edge midpoints of that root share its base
cache. There is deliberately no cache shared between different root
simplices.

The mesh's full vertex eigensystems remain persistent. At the start of a root
estimate, the relevant systems are copied into the workspace and shifted by
the chemical potential. Temporary Hamiltonians and reduced eigensystems are
not added to the persistent mesh.

### Point records

Every `PointCache` is keyed by the exact `DyadicVertex`, not a floating-point
coordinate. One record contains only representations that have actually been
requested:

```text
PointData
├── optional Hamiltonian matrix
└── optional CachedSpectrum
    ├── eigenvalues
    └── optional eigenvectors
```

An eigenvalue-only request does not retain eigenvectors. If vectors are later
needed, a full diagonalization replaces the spectrum with a consistent
eigenvalue/eigenvector pair. Cache memory diagnostics count the live and peak
payload of all base and reduced point caches.

A reduced model has a separate cache because the same momentum point under a
different active-space reduction represents a different matrix. Reduction
results are owned by the recursive call while each child model holds a
non-owning link to its lexical parent. A model and its cache therefore stay
alive for exactly the descendant traversal that uses them.

## Hamiltonian kernels

An `EffectiveModel` evaluates one explicit kernel:

- **Base:** evaluate $K(k)=H(k)-\mu I$ through the root workspace.
- **DenseSchur:** apply one corrected frozen-Schur layer to the parent
  matrix.
- **ProjectedTB:** evaluate a precomputed matrix-valued tight-binding
  polynomial directly in the active space.

For the dense kernel, an anchor eigensystem splits the parent space into an
active basis $U_a$ and a safe basis. With the frozen safe-space resolvent $R$,

$$
X=K U_a,\qquad Y=R X,
$$

and the corrected reduced matrix is

$$
S_1=U_a^\dagger X-2X^\dagger Y+Y^\dagger K Y.
$$

For a native tight-binding model, the first reduction can instead project the
hopping terms once and form

$$
S_1(k)=\sum_R \widetilde H_R e^{-2\pi i k\cdot R}.
$$

This avoids constructing the full $N\times N$ Hamiltonian at every temporary
point. An internal factory chooses `ProjectedTB` when that construction is
valid and otherwise chooses `DenseSchur`; recursion does not contain separate
tight-binding branches.

The `ModelBackend` retains only model-wide acceleration data, such as the
tight-binding hopping Gram matrix used for fast defect norms. It does not own
point-dependent results.

## Calculation flow

For an outer simplex $T$, the estimator receives its linear charge
$\widetilde Q_T$, root certificate, and prepared certificate data.

1. **Apply the root gate.** If the certificate leaves $q>2$ active bands and
   $2q\geq N$, skip the expensive recursion and use the certified occupation
   range as a conservative interval.
2. **Seed the workspace.** Copy the root vertex eigensystems already stored by
   the spectral mesh, shift their eigenvalues by $\mu$, and create the base
   effective model.
3. **Certify a node.** Obtain occupation bounds $[L,U]$ for the current
   effective model. Intersect them with the last valid fallback range.
4. **Reduce the active space.** When $0<U-L<m$, choose the vertex whose safe
   spectrum is farthest from zero and build an `ActiveSpaceReduction`. The
   traversal records the $L$ newly fixed occupied states and the reduction
   creates a child effective model of dimension $U-L$.
5. **Recurse or terminate.** A fixed-occupation node stops immediately. An
   unresolved node stops at `error_depth`; otherwise one logical subdivision
   performs $d$ binary bisections, producing $2^d$ microsimplices, and visits
   every child with the current reduced model.
6. **Sum leaf intervals.** Child lower and upper charges are accumulated, then
   compared with $\widetilde Q_T$ to obtain the root error estimate.
7. **Destroy the workspace.** All point records and reduced models for $T$ are
   released before the next outer simplex is estimated.

## Terminal interval

The terminal calculation is split into four operations.

### 1. Measure midpoint defects

For every edge $(i,j)$ with midpoint $m_{ij}$, compare the actual current
matrix with its endpoint interpolation,

$$
\delta^M_{ij}=\left\|K(m_{ij})-
\tfrac12\bigl(K(k_i)+K(k_j)\bigr)\right\|.
$$

An unresolved leaf also compares ordered midpoint eigenvalues with their
endpoint interpolation, producing $\delta^E_{ij}$. A fixed base
tight-binding leaf may first use the cheaper hopping-Gram Frobenius bound.

### 2. Certify the radius

The sampled terminal radius is

$$
\beta=\frac{2d}{d+1}
\max_{(i,j)}\max\left(\delta^M_{ij},\delta^E_{ij}\right).
$$

Prepared certificate data or an existing certificate are reused only when
they are valid for both $-\beta$ and $+\beta$ and the effective model has not
changed.

### 3. Optionally reduce again

A fixed leaf first tries the inexpensive matrix-defect radius. If that radius
reopens any bands, the calculation repeats with the full matrix and band
defects. When it reopens a strict subset, one final active-space reduction is
attempted before that repeat. If reduction is not useful, the current model is
retained.

### 4. Integrate the charge interval

Let $[L_S,U_S]$ be the occupation bounds valid at the terminal radius and
$n_0$ the number of occupied states frozen by earlier reductions. For every
remaining band, exact affine cut-simplex volumes at levels $-\beta$ and
$+\beta$ give a leaf interval $[Q_S^-,Q_S^+]$. Summing the leaves gives
$[Q_T^-,Q_T^+]$, and the reported local estimate is

$$
\eta_T=\max\left(
|\widetilde Q_T-Q_T^-|,
|Q_T^+-\widetilde Q_T|
\right).
$$

The global stopping error is $\sum_T\eta_T$. It is a sampled refinement
estimate, not a rigorous error bound.

## Failure and diagnostics rules

- A zero or non-finite safe anchor prevents that reduction; calculation
  continues with the current valid model.
- A failure while evaluating an already-created Schur layer returns the last
  valid occupation range for that node as a conservative charge interval.
- The root gate uses the root certificate's interval directly.
- Public `ChargeErrorStats` count algorithmic work and fallbacks. Internal
  profiling keeps Hamiltonian evaluation, Schur application, eigensystems,
  certification, defect work, and occupied-volume integration as exclusive
  regions, so their reported times can be added without nested double
  counting.
