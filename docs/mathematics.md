# Mathematics

FermiSimplex separates two different mathematical tasks:

1. **Certification:** prove whether the number of states below a chemical
   potential can change inside a simplex.
2. **Approximation:** linearly interpolate the vertex eigenvalues to construct
   a Fermi surface or integrate the charge.

The certification problem is a matrix-sign problem. It does not require
tracking individual eigenvalue curves through the simplex. Curvature enters
only afterward, as one possible way to bound the difference between the true
Hamiltonian and its affine interpolation.

The charge stopping criterion is different again: it is a sampled estimate of
the eigenvalue-interpolation error. It guides adaptive integration, but it is
not part of the rigorous occupation proof.

## Occupation is the inertia of a shifted Hamiltonian

Let

$$
T=\operatorname{conv}\{k_0,\ldots,k_d\}
$$

be a simplex in reduced momentum coordinates, and let
$H(k)\in\mathbb C^{N\times N}$ be Hermitian. At chemical potential $\mu$,
define

$$
A(k;\mu)=H(k)-\mu I.
$$

The inertia of a Hermitian matrix is the triple

$$
\operatorname{In} A=(n_-(A),n_0(A),n_+(A)),
$$

counting its negative, zero, and positive eigenvalues. Away from an exact
contact with $\mu$, the local occupation is simply

$$
\nu(k;\mu)=n_-\!\left(A(k;\mu)\right).
$$

A Fermi-level contact occurs precisely when $A(k;\mu)$ is singular. Thus the
relevant question on a simplex is not “how far can every eigenvalue move?” but

> Can the inertia of $A(k;\mu)$ change anywhere in $T$?

This formulation automatically ignores crossings between two occupied bands
or two unoccupied bands. Only a change of sign relative to $\mu$ matters.

### The fixed-subspace criterion

Suppose there are fixed, full-rank frames

$$
F_o\in\mathbb C^{N\times r_o},
\qquad
F_u\in\mathbb C^{N\times r_u}
$$

such that

$$
F_o^\dagger A(k;\mu)F_o\prec0,
\qquad
F_u^\dagger A(k;\mu)F_u\succ0.
$$

The first inequality proves that $A(k;\mu)$ has at least $r_o$ negative
eigenvalues; the second proves that it has at least $r_u$ positive
eigenvalues. This is the variational characterization of the positive and
negative indices of inertia.

Consequently,

$$
r_o\leq \nu(k;\mu)\leq N-r_u.
$$

If the two dimensions fill the Hilbert space,

$$
r_o+r_u=N,
$$

then $A(k;\mu)$ has exactly $r_o$ negative eigenvalues, exactly $r_u$
positive eigenvalues, and no zero eigenvalue. Therefore

$$
\boxed{
F_o^\dagger A(k;\mu)F_o\prec0,\quad
F_u^\dagger A(k;\mu)F_u\succ0,\quad
r_o+r_u=N
\Longrightarrow
\nu(k;\mu)=r_o.
}
$$

This is the core certificate used by FermiSimplex. The remaining problem is to
construct useful fixed frames and prove their signs everywhere in a simplex
using only vertex eigensystems.

## Why vertex tests extend across a simplex

Let $\lambda_i(k)$ be the barycentric coordinates of $k\in T$. Define the
vertex-affine Hamiltonian

$$
H_{\mathrm{lin}}(k)=\sum_{i=0}^d\lambda_i(k)H(k_i)
$$

and the non-affine residual

$$
R(k)=H(k)-H_{\mathrm{lin}}(k).
$$

Assume that a valid uniform bound is available:

$$
\boxed{
\sup_{k\in T}\lVert R(k)\rVert_2\leq\epsilon_T.
}
$$

The direct `certify_simplex` interface accepts $\epsilon_T$ itself. The
`fermi_surface` mesh API obtains it from `curvature_bound`; that conversion is
derived later.

For any fixed frame $F$,

$$
F^\dagger\!\left(H_{\mathrm{lin}}(k)-\mu I\right)F
=
\sum_i\lambda_i(k)
F^\dagger\!\left(H(k_i)-\mu I\right)F.
$$

The cone of positive-definite matrices is convex. Therefore, if a fixed
restricted block is positive definite at every vertex, its affine
interpolation is positive definite throughout the simplex.

The residual is controlled in the metric of the frame. If

$$
G_F=F^\dagger F,
$$

then

$$
-\epsilon_TG_F
\preceq
F^\dagger R(k)F
\preceq
\epsilon_TG_F.
$$

It is therefore sufficient to find fixed frames $F_o,F_u$ for which every
vertex satisfies

$$
\boxed{
-F_o^\dagger A_iF_o-\epsilon_TG_o\succ0,
\qquad
F_u^\dagger A_iF_u-\epsilon_TG_u\succ0,
}
$$

where

$$
A_i=H(k_i)-\mu I,
\qquad
G_o=F_o^\dagger F_o,
\qquad
G_u=F_u^\dagger F_u.
$$

Indeed, barycentric averaging preserves the strict inequalities for
$H_{\mathrm{lin}}$, and the $\epsilon_TG$ terms absorb the worst possible
residual. The implementation additionally requires the restricted blocks to
exceed the numerical margin

$$
\eta=\max(10^{-10},\text{tolerance})
$$

and verifies positive definiteness by Cholesky factorization.

This is why a simplex is special: one finite set of vertex inequalities proves
a continuum of matrix inequalities in its interior.

## Constructing the signed trial subspaces

### Vertex analysis and anchor selection

FermiSimplex begins with the cached eigensystem at every vertex:

$$
H(k_i)=U_i\operatorname{diag}
\left(\varepsilon_{i,0},\ldots,\varepsilon_{i,N-1}\right)U_i^\dagger,
$$

with the eigenvalues in ascending order.

At each vertex it counts eigenvalues strictly below $\mu$, outside the
numerical tolerance. The vertex data are marked `VisibleGapless` if

- an eigenvalue lies within the tolerance of $\mu$, or
- different vertices have different occupation counts.

In the second case continuity already forces a Fermi-level contact somewhere
between the vertices. In either case, the algorithm chooses as anchor the
vertex with the largest minimum spectral distance from $\mu$:

$$
a=\arg\max_i\min_n|\varepsilon_{i,n}-\mu|.
$$

This uses an eigensystem that is already part of the mesh. Certification does
not diagonalize the Hamiltonian at the simplex center. When the vertex data
are not visibly gapless, every vertex has the same occupation $n_o$ and the
full graph-subspace proof below is attempted. Visibly gapless simplices skip
that full proof, but still use the anchor frame to obtain partial occupation
bounds.

### All vertices in one spectral frame

Split the anchor eigenbasis into occupied and unoccupied columns,

$$
Q=[Q_o,Q_u].
$$

In this fixed basis, write every shifted vertex Hamiltonian as

$$
Q^\dagger A_iQ
=
\begin{pmatrix}
O_i & C_i^\dagger\\
C_i & U_i
\end{pmatrix}.
$$

At the anchor,

$$
O_a=-D_o,\qquad U_a=D_u,\qquad C_a=0,
$$

where $D_o$ and $D_u$ are positive diagonal matrices containing the occupied
and unoccupied distances from $\mu$.

A naive choice would keep the anchor sectors $Q_o$ and $Q_u$ fixed. That can
fail when the spectral subspaces rotate appreciably across the simplex, even
though the gap never closes. FermiSimplex therefore constructs a pair of graph
subspaces that can absorb occupied--unoccupied mixing.

Let

$$
\overline C=\frac{1}{d+1}\sum_{i=0}^d C_i.
$$

The rotation matrix $X\in\mathbb C^{n_u\times n_o}$ is

$$
X_{\alpha j}
=
\frac{\overline C_{\alpha j}}
{(D_u)_{\alpha\alpha}+(D_o)_{jj}}.
$$

Equivalently, $X$ is the elementwise solution of the Sylvester equation

$$
D_uX+XD_o=\overline C.
$$

It is a first-order guess for the average rotation of the occupied and
unoccupied spaces. In the original Hilbert space, define

$$
F_o=
Q\begin{pmatrix}
I\\-X
\end{pmatrix}
=Q_o-Q_uX,
\qquad
F_u=
Q\begin{pmatrix}
X^\dagger\\I
\end{pmatrix}
=Q_oX^\dagger+Q_u.
$$

These frames are orthogonal and complementary:

$$
F_o^\dagger F_u=0,
\qquad
\dim F_o+\dim F_u=N.
$$

Their Gram matrices are

$$
G_o=I+X^\dagger X,
\qquad
G_u=I+XX^\dagger.
$$

At vertex $i$, their exact restricted quadratic forms are

$$
F_u^\dagger A_iF_u
=
U_i+C_iX^\dagger+XC_i^\dagger+XO_iX^\dagger,
$$

and

$$
-F_o^\dagger A_iF_o
=
-O_i+C_i^\dagger X+X^\dagger C_i-X^\dagger U_iX.
$$

FermiSimplex tests

$$
F_u^\dagger A_iF_u-\epsilon_TG_u\succ\eta I
$$

and

$$
-F_o^\dagger A_iF_o-\epsilon_TG_o\succ\eta I
$$

at every vertex. If both pass, the fixed-subspace argument proves that

$$
\operatorname{In}A(k;\mu)=(n_o,0,N-n_o)
\qquad\text{for every }k\in T.
$$

The formula for $X$ is only a proof-search heuristic. The validity of the
result comes from the final positive-definiteness tests, not from perturbation
theory. A poor rotation can cause a valid gap to remain unproved, but it cannot
produce a false certificate.

## Partial occupation bounds

The full graph-subspace proof is deliberately sufficient rather than
necessary. If it fails, FermiSimplex still asks how many common negative and
positive directions can be proved.

It searches within the unrotated anchor sectors. Suppose it finds fixed
orthonormal frames

$$
P_o\subseteq Q_o,
\qquad
P_u\subseteq Q_u,
$$

of dimensions $r_o$ and $r_u$, such that at every vertex

$$
-P_o^\dagger A_iP_o-\epsilon_TI\succ\eta I,
$$

$$
P_u^\dagger A_iP_u-\epsilon_TI\succ\eta I.
$$

The same barycentric and residual argument proves these signs throughout the
simplex. Hence

$$
\boxed{
r_o\leq \nu(k;\mu)\leq N-r_u
\qquad(k\in T).
}
$$

The returned occupation bounds are

$$
[L,U]=[r_o,N-r_u].
$$

Internally, anchor directions are ordered by their worst diagonal margin over
all vertices. Cholesky factorizations then find a common passing leading
subspace. This deterministic ordering is not guaranteed to find the largest
possible subspace; it affects the tightness of $[L,U]$, not its validity.

The half-open band interval

$$
[L,U)
$$

has a useful interpretation. Bands below $L$ are proved occupied throughout
the simplex, bands at or above $U$ are proved unoccupied, and only the $U-L$
intervening ordered bands can affect the occupation uncertainty.

If $L=U$, the occupation is exact even if the graph-subspace search failed.

### Certificate statuses

- `CertifiedGapped`: constant occupation was proved, either by the
  complementary graph frames or by matching partial bounds after an otherwise
  inconclusive full proof.
- `VisibleGapless`: the sampled vertices already show a contact or an
  occupation change.
- `Inconclusive`: neither a crossing nor a constant occupation was proved.

`Inconclusive` is not evidence of a Fermi surface. It means that refinement is
needed before the available sufficient conditions can decide.

## Why this is sharper than a single Weyl radius

A direct Weyl argument would choose one reference point $k_0$, define its
smallest distance from $\mu$,

$$
g_\mu(k_0)=\min_n|\varepsilon_n(k_0)-\mu|,
$$

and try to prove

$$
\sup_{k\in T}\lVert H(k)-H(k_0)\rVert_2<g_\mu(k_0).
$$

This compresses the entire matrix motion into one norm and compares it with
one smallest gap. A rapidly moving remote band or harmless
occupied--unoccupied mixing can dominate that norm.

FermiSimplex instead separates the Hamiltonian into

$$
H(k)=H_{\mathrm{lin}}(k)+R(k).
$$

The possibly large vertex-to-vertex motion in $H_{\mathrm{lin}}$ is handled
explicitly through signed restricted blocks. Only the genuinely unsampled
part $R(k)$ is replaced by a scalar norm bound. Thus:

- affine matrix variation is treated exactly, no matter how large it is;
- positive and negative sectors are tested separately;
- motion away from $\mu$ strengthens rather than weakens a restricted block;
- occupied--unoccupied mixing can be absorbed by the graph rotation;
- remote bands enter through the actual block matrices instead of sharing one
  global smallest-gap comparison.

For a smooth Hamiltonian on a simplex of diameter $h$, vertex-to-vertex motion
is generally $O(h)$, while the residual of affine interpolation is $O(h^2)$.
This is the main reason to certify the affine matrix structure first and bound
only what remains.

## Obtaining the residual bound

The certificate itself requires only a valid $\epsilon_T$ satisfying

$$
\lVert H(k)-H_{\mathrm{lin}}(k)\rVert_2\leq\epsilon_T.
$$

`curvature_bound` is how the `fermi_surface` mesh API constructs this
quantity; it is not a bound on the curvature of an individual band.

Assume that $H$ is twice differentiable and that

$$
M\geq
\sup_{k,\,\lVert v\rVert_2=1}
\left\lVert D_v^2H(k)\right\rVert_2.
$$

Taylor expansion of $H(k_i)$ around an interior point $k$ gives a first-order
term proportional to $k_i-k$ and a remainder bounded by

$$
\frac{M}{2}\lVert k_i-k\rVert_2^2.
$$

The barycentric identity

$$
\sum_i\lambda_i(k)(k_i-k)=0
$$

cancels all first-order terms. Therefore

$$
\lVert R(k)\rVert_2
\leq
\frac{M}{2}
\sum_i\lambda_i(k)\lVert k_i-k\rVert_2^2
\leq
\frac{M D_T^2}{2},
$$

where $D_T$ is the simplex diameter. FermiSimplex consequently uses

$$
\boxed{
\epsilon_T=\frac12\,M D_T^2.
}
$$

Refinement reduces this residual allowance quadratically with the simplex
diameter.

### Tight-binding Hamiltonians

For reduced coordinates and

$$
H(k)=\sum_R H_Re^{-2\pi i k\cdot R},
$$

the second directional derivative is

$$
D_v^2H(k)
=
-(2\pi)^2
\sum_R(v\cdot R)^2H_Re^{-2\pi i k\cdot R}.
$$

The triangle inequality gives

$$
\sup_{k,\,\lVert v\rVert_2=1}\lVert D_v^2H(k)\rVert_2
\leq
(2\pi)^2
\sum_R\lVert H_R\rVert_2\lVert R\rVert_2^2.
$$

Choosing the right-hand side as $M$ is therefore sufficient.

The Frobenius norm may replace $\lVert H_R\rVert_2$ for an easier but more
conservative estimate.

A positive `curvature_bound` supplied by the surface caller gives this uniform
bound. Omitting it, `None`, or `0.0` selects sampled remainders within the same
quadratic occupation algorithm. An affine matrix can still have curved
eigenvalues, so exact matrix interpolation does not make its affine-band
charge exact. The shared enclosure accounts for both effects.

## Reusing bounds at nearby chemical potentials

Once a fixed frame has been certified, changing the chemical potential does
not require rebuilding every restricted block. For any frame $F$,

$$
F^\dagger A(k;\mu+\delta)F
=
F^\dagger A(k;\mu)F-\delta F^\dagger F.
$$

The unoccupied block therefore limits how far $\mu$ may be raised, while the
occupied block limits how far it may be lowered. FermiSimplex uses Gershgorin
row bounds to obtain conservative one-sided radii and returns an interval

$$
\mu'\in[\mu-r_o^\mu,\mu+r_u^\mu]
$$

over which the stored occupation bounds remain valid.

Only the bounds $[L,U]$ are reusable throughout this interval. The descriptive
status can change; for example, a vertex eigenvalue may become visibly equal
to a shifted $\mu$ even though the stored lower and upper occupation bounds
remain correct.

## Adaptive Fermi-surface determination

The occupation certificate controls which simplices can be discarded:

1. Evaluate and cache the missing vertex eigensystems.
2. Build the shared quadratic occupation enclosure at $\mu$.
3. Discard a simplex with strict fixed occupation in that enclosure.
4. Refine a `VisibleGapless` or `Inconclusive` simplex while its diameter
   exceeds `min_feature_size`.
5. At the requested feature size, intersect the ordered vertex-linear bands
   with $\mu$ on the retained simplices.

This prevents a small pocket from being silently discarded merely because all
coarse vertices happen to lie on the same side of $\mu$: such a simplex must
pass the matrix certificate before it is removed.

`coverage_certified` is true only when the run completes, no inconclusive
simplex survives at the terminal feature size, and the extracted result
contains no unrepresentable flat-band or lower-dimensional contact. Conditional
on a valid interpolation remainder, it certifies that discarded simplices contain
no true Fermi-level crossing.

It does **not** prove that the returned piecewise-linear surface has the exact
topology, a bounded Hausdorff distance, or a certified geometric interpolation
error. Those are stronger statements than spectral coverage at a requested
feature size.

## Piecewise-linear charge

At zero temperature,

$$
Q(\mu)=\int_{\mathrm{BZ}}\nu(k;\mu)\,dk.
$$

For each ordered band index $n$, FermiSimplex linearly interpolates the vertex
eigenvalues:

$$
\widetilde E_n(k)
=
\sum_i\lambda_i(k)\varepsilon_{i,n}.
$$

The simplex contribution is

$$
\widetilde Q_T(\mu)
=
\sum_n
\operatorname{vol}
\{k\in T:\widetilde E_n(k)\leq\mu\}.
$$

The scalar cut recurrence computes these affine cut-simplex volumes. A band
lying identically on the level is assigned half of the simplex volume. The
mesh level tolerance first snaps near-level vertex energies to the level.

For distinct vertex energies $e_0,\ldots,e_d$, the derivative of one band
contribution, away from tolerance-snapped vertices, is

$$
\frac{d\widetilde Q_T}{d\mu}
=
d|T|
\sum_{i=0}^d
\frac{(\mu-e_i)_+^{d-1}}
{\prod_{j\neq i}(e_j-e_i)}.
$$

The implementation differentiates the cumulative fraction recurrence directly,
holding snapped vertices on the level. It therefore returns the slope of the
reported charge, including zero on tolerance plateaus. The
[occupation design](occupation-enclosure.md) gives the recurrence and its
convention at exact knots when the level tolerance is zero.

## Shared occupation enclosure

Adaptive charge and surface classification use the same quadratic matrix
model, safe-sector proof, Schur remainder and occupation interval. The
[occupation design](occupation-enclosure.md) gives the algorithm and its
assumptions. Temporary subdivision evaluates only the fixed polynomial.
Increasing its depth improves integration of that model; persistent mesh
refinement is needed to reduce its interpolation remainder.

The reported charge is still the piecewise-linear integral above. Its local
error indicator is the largest distance to the enclosure endpoints. The
separate density-cut indicator also includes spatial disagreement of the
occupied regions, so a cancellation in total charge does not hide an error
in the density cut.

## Density matrices

Density matrices reuse the adaptive geometry and cached vertex eigensystems.
For active parent-child corrections $\Delta_T$, their stopping estimate is

$$
\max\left(
    \sqrt{\sum_T \lVert\Delta_T\rVert_\infty^2},
    \left\lVert\sum_T\Delta_T\right\rVert_\infty
\right).
$$

The first term guards against complete cancellation and the second retains the
signed componentwise correction. This estimate is not rigorous: the occupation
certificate does not bound variation of the spectral projector, so density
matrices are not currently certified.
`integrate_density_components` applies the same rule only to requested entries
$(r,i,j)$. Requests with the same $(i,j)$ share the projector contraction and
differ only by the lattice phase for $r$. The stopping estimate then covers
only those requested entries. Full density matrices are the special case that
requests every $(r,i,j)$.
With `preview_depth=0`, no parent-child correction is sampled: the current mesh
is integrated directly, `stopping_error` is zero by construction, and no
refinement occurs.

## What the reported quantities mean

| Quantity | Meaning | Rigorous? |
|---|---|---|
| `occupation_bounds = [L, U]` | Every point in the simplex has between $L$ and $U$ occupied states | Yes, if $\epsilon_T$ is valid |
| `CertifiedGapped` | The occupation is constant and no Fermi-level state exists in the simplex | Yes, if $\epsilon_T$ is valid |
| `VisibleGapless` | Vertex data show a contact or force an occupation change | Yes, assuming a continuous Hamiltonian |
| `Inconclusive` | The sufficient proof did not decide | No gap or crossing conclusion |
| charge `stopping_error` | Sum of distances to occupation-interval endpoints | Conditional on the remainder |
| current-mesh charge `value` | Direct linear-simplex integral with no accompanying error estimate | No |
| `charge.error_stats` | Estimator work and active-space counters | Diagnostic only |
| `coverage_certified` | The run completed with no terminal inconclusive or unrepresentable contact | Conditional on valid $\epsilon_T$; not a topology guarantee |
| density-matrix `stopping_error` | Adaptive quadrature estimate | No |

The certificate rows apply to the direct interface and Fermi-surface path when a
valid residual bound is supplied. Charge uses the shared polynomial enclosure. Its allowance is sampled by
default, so its indicator is not a general uniform error certificate.

The rigorous statements also assume finite ascending vertex eigenvalues and
finite column-orthonormal eigenvectors. The direct certification interface
checks container dimensions but intentionally does not revalidate these
performance-sensitive numerical preconditions.
