# Mathematics

FermiSimplex combines two ideas:

1. certify how many states are occupied throughout a simplex;
2. integrate the piecewise-linear bands and refine where their error appears
   largest.

The first statement is rigorous when the supplied curvature bound is valid.
The projected charge estimator is deliberately cheaper and is not itself a
certificate.

## Problem and notation

Let

$$
S=\mathrm{conv}\{k_0,\ldots,k_d\}
$$

be a simplex with barycentric coordinates $\lambda_i(k)$, volume $|S|$, and
diameter $D_S$. The vertex-linear Hamiltonian and its residual are

$$
H_{\mathrm{lin}}(k)=\sum_i\lambda_i(k)H(k_i),
\qquad
R(k)=H(k)-H_{\mathrm{lin}}(k).
$$

At chemical potential $\mu$, the exact local occupation is

$$
N(k; \mu)=\mathrm{Tr}\left[\Theta\left(\mu I-H(k)\right)\right].
$$

All coordinates used by the package are reduced coordinates in $[0,1]^d$.

## From curvature to a simplex error bound

The user supplies a global directional-curvature bound

$$
M\geq
\sup_{k,\,\lVert v\rVert_2=1}
\left\lVert D_v^2H(k)\right\rVert_2.
$$

Barycentric interpolation then gives

$$
\lVert R(k)\rVert_2
\leq \frac{M}{2}\sum_i\lambda_i(k)\lVert k_i-k\rVert_2^2
\leq \frac{M D_S^2}{2}.
$$

The certificate therefore receives the scalar

$$
\boxed{\epsilon_S=\frac12 M D_S^2}.
$$

For a tight-binding Hamiltonian

$$
H(k)=\sum_R H_R e^{-2\pi i k\cdot R},
$$

one simple, usually conservative, choice is

$$
M=(2\pi)^2\sum_R\lVert H_R\rVert_F\lVert R\rVert_2^2.
$$

Certification always runs. Omitting `curvature_bound`, passing `None`, or
passing `0` all mean $M=0$: the caller is asserting that the Hamiltonian is
affine. The package cannot verify $M$, so every rigorous conclusion below is
conditional on that assertion being true.

## Occupation certification

### Vertex analysis

At every vertex, the algorithm counts eigenvalues strictly below $\mu$. A
simplex is visibly gapless when a vertex eigenvalue lies at $\mu$ within the
numerical tolerance, or when the vertex occupation counts differ. Otherwise,
the vertex with the largest minimum distance from $\mu$ becomes the anchor.

Let the anchor eigenbasis be split into occupied and unoccupied columns,
$Q=[Q_o,Q_u]$. At each vertex,

$$
Q^\dagger(H_i-\mu I)Q=
\begin{pmatrix}
A_i & C_i^\dagger\\
C_i & D_i
\end{pmatrix},
$$

where the anchor has $n_o$ occupied and $n_u$ unoccupied states.

### A fixed pair of graph subspaces

The off-diagonal blocks suggest a perturbative rotation

$$
X_{\alpha j}=
\frac{(C_{\mathrm{avg}})_{\alpha j}}
{\delta^u_\alpha+\delta^o_j}.
$$

This defines two orthogonal graph frames,

$$
F_o=\begin{pmatrix}I\\-X\end{pmatrix},
\qquad
F_u=\begin{pmatrix}X^\dagger\\I\end{pmatrix}.
$$

For every vertex, the implementation tests

$$
-F_o^\dagger(H_i-\mu I)F_o
-\epsilon_S F_o^\dagger F_o \succ 0
$$

and

$$
F_u^\dagger(H_i-\mu I)F_u
-\epsilon_S F_u^\dagger F_u \succ 0.
$$

The error terms are valid because

$$
-\epsilon_S F^\dagger F
\preceq F^\dagger R(k)F
\preceq \epsilon_S F^\dagger F.
$$

The same frames are used at every vertex. Their restrictions of
$H_{\mathrm{lin}}$ are barycentric combinations, so positivity at all vertices
implies positivity everywhere inside the simplex. The two frame dimensions add
to the full Hilbert-space dimension; consequently the inertia is fixed and the
occupation is exactly $n_o$ throughout $S$.

The rotation is only a proof-search heuristic. If this test fails, the simplex
has not been proved gapless.

### Partial occupation bounds

When the full proof fails, the code searches for common negative directions in
the anchor occupied sector and common positive directions in the unoccupied
sector. Suppose it proves $r_o$ negative directions and $r_u$ positive
directions. Then

$$
\boxed{r_o\leq N(k;\mu)\leq n_{\mathrm{dof}}-r_u}
\qquad (k\in S).
$$

The result is stored as `occupation_bounds = [L, U]`. Directions are ordered by
their worst diagonal margin over the vertices, and leading subspaces are
accepted by Cholesky tests. This ordering changes how tight the result is, not
its validity. If $L=U$, the occupation is exact even if the rotated full proof
failed.

The three statuses mean:

- `CertifiedGapped`: constant occupation was proved;
- `VisibleGapless`: the vertex data display a contact or occupation change;
- `Inconclusive`: neither conclusion was proved.

Positive-definiteness tests use the numerical margin
$\max(10^{-10},\text{tolerance})$.

### Reuse across chemical potentials

Changing $\mu$ shifts a restricted block by its frame Gram matrix,

$$
B(\mu+\delta)=B(\mu)-\delta G,
\qquad G=F^\dagger F.
$$

Gershgorin row margins provide conservative one-sided radii. If
$r_o^\mu$ comes from the occupied sector and $r_u^\mu$ from the unoccupied
sector, the occupation bounds remain valid for

$$
\boxed{\mu'\in[\mu-r_o^\mu,\ \mu+r_u^\mu]}.
$$

Only the occupation bounds are reusable over this interval; the status can
change within it.

## Piecewise-linear simplex charge

The zero-temperature filling is

$$
Q(\mu)=\int_{\mathrm{BZ}}N(k;\mu)\,dk.
$$

For each sorted band index $n$, FermiSimplex interpolates the vertex
energies,

$$
\widetilde E_n(k)=\sum_i\lambda_i(k)E_{in},
$$

and computes

$$
\widetilde Q_S(\mu)=
\sum_n \mathrm{vol}
\{k\in S:\widetilde E_n(k)\leq\mu\}.
$$

AdaptiveSimplex evaluates these affine cut-simplex volumes exactly. A band
lying entirely on the level is assigned half of the simplex volume.

For distinct vertex energies $e_0,\ldots,e_d$, the derivative of one band
contribution is

$$
\frac{d\widetilde Q_S}{d\mu}
=d|S|\sum_{i=0}^d
\frac{(\mu-e_i)_+^{d-1}}
{\prod_{j\neq i}(e_j-e_i)}.
$$

The implementation uses the equivalent divided-difference formula and
confluent divided differences for repeated numerical knots.

## The certified charge bound

If a certificate establishes $L\leq N(k;\mu)\leq U$, the linearly interpolated
occupation also stays between those ranks. Therefore

$$
\left|Q_S-\widetilde Q_S\right|\leq(U-L)|S|.
$$

The reported global quantity is

$$
\boxed{B_{\mathrm{cert}}=
\sum_{S\in\mathcal P}(U_S-L_S)|S|},
$$

where $\mathcal P$ is the preview partition used for the returned charge. This
is `certified_error_bound`. It is rigorous when $M$ is valid, but it is not the
criterion used to stop refinement. Even when $M=0$, the bound can be positive
if the occupation remains ambiguous.

## The projected charge estimator

Let the occupation certificate select exactly the half-open band interval
$J=[L,U)$, with width $m=U-L$. No neighboring guard bands are added. At vertex
$k_i$, write the selected eigenvalues as

$$
\varepsilon_{i,r}=\varepsilon_{L+r}(k_i),
\qquad r=0,\ldots,m-1,
$$

and collect their orthonormal eigenvectors in the frame
$V_i\in\mathbb C^{N\times m}$.

At the simplex barycenter $k_c$, compute the complete ordered spectrum using
an eigenvalues-only Hermitian solve,

$$
\eta_{c,0}\le\cdots\le\eta_{c,N-1},
\qquad \theta_{c,r}=\eta_{c,L+r}.
$$

For the midpoint of edge $(i,j)$, compute the principal-angle SVD

$$
V_i^\dagger V_j=A\Sigma C^\dagger
$$

and use the aligned midpoint frame

$$
Q_{ij}=(V_iA+V_jC)(2I+2\Sigma)^{-1/2}.
$$

This construction depends only on the endpoint subspaces, not on phase or
unitary gauge choices within their selected frames.

At each edge midpoint, diagonalize only the projected matrix

$$
H_{ij}^{\mathrm{proj}}=Q_{ij}^\dagger H(k_{ij})Q_{ij}
$$

for its $m$ ordered eigenvalues $\theta_{ij,r}$. Thus the center is exact at
its sampled point while the edges use $m$-dimensional Rayleigh--Ritz values.
Compare both kinds of ordered values with the vertex-linear band interpolation

$$
\ell_{p,r}=\sum_i\lambda_i(k_p)\varepsilon_{i,r},
\qquad
\delta_{p,r}=\theta_{p,r}-\ell_{p,r}.
$$

The directional estimates are

$$
\rho_+=\max_{p,r}[\delta_{p,r}]_+,
\qquad
\rho_-=\max_{p,r}[-\delta_{p,r}]_+.
$$

The probes are the barycenter and every edge midpoint, after removing
duplicates: one point in 1D, the center plus three edges in 2D, and the center
plus six edges in 3D. Edge midpoints are mathematically natural because the
error of scalar linear interpolation of a quadratic along an edge is
proportional to $t(1-t)$ and therefore has its largest magnitude at the
midpoint. The edge midpoints together with the vertices are the standard
quadratic interpolation nodes. The barycenter adds a symmetric interior
probe, although the estimator does not construct a quadratic interpolant.

A positive difference raises the sampled energy relative to its
linear interpolation and can reduce occupation; a negative difference can
increase it. The existing occupation-volume shell therefore uses

$$
E_{\mathrm{proj},S}=\sum_{n=L}^{U-1}
\left[V_{n,S}(\mu+\rho_-)-V_{n,S}(\mu-\rho_+)\right]_+,
$$

where $V_{n,S}(a)$ is the occupied cut volume of interpolated band $n$ at
level $a$.

This is a sampled estimate—not a proof. It can miss unsampled extrema, and the
edge values can miss errors caused by components outside their projected
subspaces. The exact center also removes any accidental inflation that a
projected center's Ritz error could have supplied. Consequently,
`target_reached` means that the sampled stopping criterion was met; it is not a
continuous interpolation-error certificate.

## Adaptive stopping and optional preview

Production charge calculations use `preview_depth=0`. In that case the
AdaptiveSimplex preview value is the coarse value and its correction vanishes:

$$
Q_S^{\mathrm{preview}}=Q_S^{\mathrm{coarse}}=\widetilde Q_S,
\qquad \Delta Q_S=0.
$$

The local and global stopping estimates are therefore

$$
\boxed{E_S=E_{\mathrm{proj},S}},
\qquad
E_{\mathrm{stop}}=\sum_S E_S.
$$

The same local projected-error estimate prioritizes refinement. This avoids
an additional refined evaluation because the charge rule already measures its
own sampled interpolation error.

A positive `preview_depth=p` remains available for tests and diagnostics. Let
$\mathcal P_p(S)$ be its depth-$p$ preview leaves. Then

$$
Q_S^{\mathrm{coarse}}=\widetilde Q_S,
\quad
Q_S^{\mathrm{preview}}=\sum_{T\in\mathcal P_p(S)}\widetilde Q_T,
\quad
\Delta Q_S=Q_S^{\mathrm{preview}}-Q_S^{\mathrm{coarse}}.
$$

The local stopping contribution is

$$
\boxed{E_S=
\sum_{T\in\mathcal P_p(S)}E_{\mathrm{proj},T}
+|\Delta Q_S|},
$$

The returned charge is the preview value. With the production depth zero this
is exactly the current-mesh charge. `stats.target_reached` only says that the
sampled estimate met the target.

Increasing $M$ increases $\epsilon_S$, which can widen $[L,U)$, enlarge both
reported errors, and require more refinement. Refinement reduces the curvature
term quadratically through $D_S^2$.

## Fermi surfaces and density matrices

Fermi-surface extraction discards certified-gapped simplices and refines
visible or inconclusive ones to `min_feature_size`. It then intersects the
piecewise-linear bands with $\mu$. `coverage_certified` concerns classification
down to that feature size; it does not certify topology, Hausdorff distance, or
geometric interpolation error.

Density matrices use the same adaptive mesh and cached eigensystems, but their
`stopping_error` is an adaptive quadrature estimate. A rigorous projector bound
would additionally require spectral-gap control, so density matrices are not
currently certified.
