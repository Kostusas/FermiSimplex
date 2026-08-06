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

The Fermi-surface caller supplies this bound. Omitting `curvature_bound`, passing
`None`, or passing `0.0` all assert $M=0$, meaning that the Hamiltonian itself
is affine on every tested simplex. None of these values disables
certification.

An affine matrix Hamiltonian can still have curved eigenvalues because its
eigenspaces may rotate and its levels may avoid one another. Hence $M=0$ can
make the curvature-backed occupation certificate exact while the
linear-tetrahedron charge still needs refinement. This is why matrix
certification and band-interpolation error estimation are separate algorithms.

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
2. Build the simplex certificate at $\mu$.
3. Discard a `CertifiedGapped` simplex; it contains no Fermi-level state.
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
on the supplied residual bound, it certifies that discarded simplices contain
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

AdaptiveSimplex computes these affine cut-simplex volumes exactly. A band
lying identically on the level is assigned half of the simplex volume.

For distinct vertex energies $e_0,\ldots,e_d$, the derivative of one band
contribution is

$$
\frac{d\widetilde Q_T}{d\mu}
=
d|T|
\sum_{i=0}^d
\frac{(\mu-e_i)_+^{d-1}}
{\prod_{j\neq i}(e_j-e_i)}.
$$

The implementation uses the equivalent divided-difference expression and
confluent divided differences for repeated numerical knots.

## Recursive active-space charge-error estimator

The reported charge remains the current-mesh linear-tetrahedron value
$\widetilde Q_T$. A separate recursive calculation estimates how far that value
may move. It uses certification to identify a small active space, not as a
continuous charge certificate.

### Root active-space gate

Let the root vertex certificate (used here with zero residual) return
occupation bounds $[L_T,U_T]$ and

$$
q=U_T-L_T
$$

uncertain states out of $N$. If

$$
q>2
\qquad\text{and}\qquad
2q\geq N,
$$

the reduction is not worth its cost. The estimator returns the tighter
sampled occupation interval

$$
Q_T^- = L_T|T|,
\qquad
Q_T^+ = U_T|T|.
$$

The corresponding local estimate is

$$
\eta_T=\max\left(
|\widetilde Q_T-Q_T^-|,
|Q_T^+-\widetilde Q_T|
\right).
$$

The $q\leq2$ exemption is important: a scalar or two-band model always enters
the recursive estimator even though its active space is at least half of the
full matrix. The gate is applied only to the original simplex, never to its
descendants.

### Frozen Schur reduction with one correction

At a recursive node, let $K(k)$ denote the current full or already reduced
Hamiltonian relative to the chemical potential. Certification of its vertex
eigensystems selects active and safe directions. The vertex whose safe spectrum
is farthest from zero supplies fixed orthonormal bases $U_?$ and $U_s$. Because
these are eigenvectors at the anchor $k_0$,

$$
D_0=U_s^\dagger K(k_0)U_s=\operatorname{diag}(d_s),
\qquad
U_s^\dagger K(k_0)U_?=0.
$$

At a sampled point define

$$
A=U_?^\dagger K U_?,
\qquad
B=U_s^\dagger K U_?,
\qquad
D=U_s^\dagger K U_s.
$$

A frozen Schur approximation would use $X_0=D_0^{-1}B$ and
$S_0=A-B^\dagger X_0$. The estimator takes one inexpensive correction,

$$
X_1=X_0+D_0^{-1}(B-DX_0),
\qquad
S_1=A-B^\dagger X_1.
$$

Define the anchor safe-space resolvent

$$
R=U_sD_0^{-1}U_s^\dagger.
$$

With

$$
X=KU_?,
\qquad
Y=RX,
\qquad
F=X^\dagger Y,
$$

the implementation evaluates the equivalent expression

$$
\boxed{
S_1=U_?^\dagger X-2F+Y^\dagger K Y.
}
$$

The layer forms $R$ once, then discards $U_s$ and $D_0^{-1}$. Every sampled
matrix uses two reused $m\times q$ work buffers: after forming $Y$, the $X$
buffer is overwritten by $KY$. Only the resulting $q\times q$ matrices are
cached. Thus the estimator never forms or factors the large pointwise safe
block $D(k)$. Forming the dense resolvent costs one setup product per layer,
but makes repeated applications efficient when $q\ll m$.

If $B=O(h)$ and $D-D_0=O(h)$, plain freezing leaves a typical $O(h^3)$ Schur
error, while one correction leaves the next $O(h^4)$ term. This statement
assumes that the anchor safe gap stays separated and that the Neumann expansion
is useful. The corrected surrogate is not an exact Schur complement and does
not preserve inertia exactly. It can become inaccurate when safe-block
variation is comparable with the anchor gap or when a safe eigenvalue crosses
zero. The deliberately simple implementation adds no variation guard; exact
finite-stencil aliasing is a separate limitation discussed below.

Every block still uses the actual current matrix at that sampled point. At the
root this is the actual $H(k)-\mu I$; no affine Hamiltonian reconstruction is
used. Deeper recursion re-anchors the current small effective matrix, freezes
its newly safe spectrum, and applies the same one-correction construction.

### Logical microsimplex recursion

Each unresolved node is completely subdivided before certification repeats on
its children. One logical subdivision consists of $d$ longest-edge bisections,
so it creates $2^d$ child simplices. Consequently, `error_depth=1` allows
at most one complete subdivision along an unresolved branch, not one binary
bisection. The public charge API defaults to `error_depth=2`. A certified node
stops before subdividing.

On every child, the algorithm certifies the current small matrices, freezes any
new safe directions, and creates another corrected frozen-Schur layer when
possible.
This can reduce the active dimension repeatedly. A node terminates when its
occupation is fixed or when `error_depth` is reached. All new microvertices
evaluate the actual model Hamiltonian; temporary samples do not refine the
persistent mesh.

### Terminal midpoint envelope

Let $K_S$ be the terminal full or reduced matrix on a $d$-simplex $S$. For every
edge $(i,j)$ with midpoint $m_{ij}$, calculate the matrix disagreement

$$
\delta^{M}_{ij}
=
\left\|
K_S(m_{ij})-\frac{K_S(k_i)+K_S(k_j)}{2}
\right\|_2.
$$

For an unresolved leaf, also compare its ordered eigenvalues. A leaf fixed at
zero radius first uses the cheaper Frobenius upper bound for $\delta^M$ and
repeats certification with the provisional $\beta_S$. If that radius reopens
any bands, the estimator evaluates the ordered midpoint eigenvalues, recomputes
$\delta^M$ in operator norm, and certifies again. Otherwise it takes
$\delta^E_{ij}=0$.

$$
\delta^{E}_{ij}
=
\max_n\left|
\lambda_n(K_S(m_{ij}))
-\frac{\lambda_n(K_S(k_i))+\lambda_n(K_S(k_j))}{2}
\right|.
$$

The second term detects eigenvalue curvature from internal band mixing even when
the matrix itself is affine. Set

$$
\delta_S
=
\max_{(i,j)}\max(\delta^M_{ij},\delta^E_{ij}),
\qquad
\boxed{
\beta_S=\frac{2d}{d+1}\,\delta_S.
}
$$

The prefactor converts the sampled edge defect into the dimension-generic
simplex interpolation scale. Because the edge midpoints are only samples,
$\beta_S$ is an estimator rather than a supremum bound.

### Shifted occupation volumes

The terminal certificate is needed at both shifted levels $-\beta_S$ and
$+\beta_S$. If the existing certificate's chemical-potential interval contains
both levels and the effective model has not been re-anchored, its occupation
bounds are reused. Otherwise certification is repeated with radius $\beta_S$.
This restriction prevents a parent certificate from being applied to a changed
Schur surrogate.

Let the resulting reduced occupation bounds be $[L_S,U_S]$, let $n_0$ be the
number of previously frozen occupied states, and let $e_{n,i}$ be terminal
vertex energies relative to $\mu$. Define the affine cut volume

$$
\Phi_S(a;e_{n,0},\ldots,e_{n,d})
=
\operatorname{vol}\left\{
 k\in S:
 \sum_i\lambda_i(k)e_{n,i}\leq a
\right\}.
$$

The leaf interval is

$$
Q_S^-
=
(n_0+L_S)|S|
+
\sum_{n=L_S}^{U_S-1}\Phi_S(-\beta_S;e_{n,0},\ldots,e_{n,d}),
$$

$$
Q_S^+
=
(n_0+L_S)|S|
+
\sum_{n=L_S}^{U_S-1}\Phi_S(+\beta_S;e_{n,0},\ldots,e_{n,d}).
$$

The signs follow because the effective energies are measured relative to
$\mu$: lowering the cut level gives the lower occupied volume. A leaf whose
$\beta_S$ certificate remains fixed has $L_S=U_S$ and contributes its integer
occupation exactly within the sampled model. A candidate layer with a zero or
non-finite anchor safe eigenvalue is discarded, leaving the current small
matrix. A non-finite corrected sample uses the last
valid sampled occupation range.

Sum the leaf endpoints to obtain $Q_T^-$ and $Q_T^+$. The root estimate remains

$$
\boxed{
\eta_T=\max\left(
|\widetilde Q_T-Q_T^-|,
|Q_T^+-\widetilde Q_T|
\right).
}
$$

The global stopping estimate and refinement priority use

$$
\boxed{
E_{\mathrm{stop}}=\sum_T\eta_T.
}
$$

`charge.value` remains the current-mesh linear-tetrahedron charge.
`stats.target_reached` means only that this sampled sum met `target_error`.
The separate `estimate_charge_on_current_mesh` operation evaluates the same
linear-simplex charge and derivative without running this estimator; its result
therefore contains no `stopping_error`.

### Expected scaling and limitations

For a smooth regular Fermi crossing, midpoint defects are normally $O(h^2)$.
The shifted shell is then $O(h^{d+1})$ on each cut $d$-simplex, giving roughly
$O(h^2)$ global decay. The full unresolved-volume fallback is instead $O(h^d)$
locally and usually only $O(h)$ globally.

Actual-Hamiltonian microvertex evaluations catch non-affinity and higher Fourier
modes whenever those modes are visible at the samples. They cannot prevent
exact finite-stencil aliasing. For example, on a unit root interval,

$$
E(k)=\frac12+\sin(4\pi k)
$$

is positive at both vertices and the edge midpoint but crosses zero between
them. That root appears fixed and terminates, so increasing `error_depth` adds
no samples there. Greater depth improves resistance only along branches that
remain unresolved; a continuous guarantee requires an independent variation
bound.

`charge.error_stats` exposes the work and failure modes: root, micro-, and
terminal-simplex counts; actual Hamiltonian evaluations; full, reduced, and
norm eigensystems; corrected Schur evaluations and Schur reductions; the
`conservative_fallbacks` and `schur_failures` counters; and initial,
terminal, and minimum active dimensions. A `minimum_active_dimension` value of
zero means that no Schur reduction was recorded.

## Density matrices

Density matrices reuse the adaptive geometry and cached vertex eigensystems.
Their `stopping_error` is an adaptive quadrature estimate. The occupation
certificate alone does not bound the variation of the spectral projector; a
rigorous density-matrix certificate would require additional gap-dependent
control. Density matrices are therefore not currently certified.
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
| charge `stopping_error` | Sum of recursive Schur shifted-volume estimates | No |
| current-mesh charge `value` | Direct linear-simplex integral with no accompanying error estimate | No |
| `charge.error_stats` | Estimator work, active-space, and fallback counters | Diagnostic only |
| `coverage_certified` | The run completed with no terminal inconclusive or unrepresentable contact | Conditional on valid $\epsilon_T$; not a topology guarantee |
| density-matrix `stopping_error` | Adaptive quadrature estimate | No |

The certificate rows apply to the direct interface and Fermi-surface path when a
valid residual bound is supplied. Charge reuses the same statuses to organize
its sampled active space; it does not turn them into a continuous charge
certificate.

The rigorous statements also assume finite ascending vertex eigenvalues and
finite column-orthonormal eigenvectors. The direct certification interface
checks container dimensions but intentionally does not revalidate these
performance-sensitive numerical preconditions.
