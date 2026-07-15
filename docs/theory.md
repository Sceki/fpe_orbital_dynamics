# Theory notes — code ↔ paper map

Reference: G. Acciarini, C. Greco, M. Vasile, *Uncertainty propagation in
orbital dynamics via Galerkin projection of the Fokker–Planck Equation*,
Advances in Space Research 73 (2024) 53–63,
[doi:10.1016/j.asr.2023.11.042](https://doi.org/10.1016/j.asr.2023.11.042).

## 1. Fokker–Planck equation (paper Sec. 2)

For the Itô SDE `dXₜ = f(Xₜ) dt + σ(Xₜ) dWₜ` the pdf `p(x, t)` obeys

```
∂p/∂t = −Σᵢ ∂/∂xᵢ (fᵢ p) + Σᵢ,ₗ ∂²/(∂xᵢ∂xₗ) (Dᵢₗ p),    D = σσᵀ/2        (paper Eq. 2)
```

Neither `f` nor `D` may depend explicitly on time (required for the
matrix-exponential solution of Eq. 13).

## 2. Galerkin projection (paper Sec. 3.1)

Ansatz `p(x, t) = Σⱼ aⱼ(t) Φⱼ(x)` (Eq. 3) with multivariate bases built as
tensor products of 1D B-splines (Eq. 4), `N = Πᵢ Nᵢ` (Eq. 5). Projecting the
FPE onto each `Φₖ` with the L2 inner product (Eq. 7, weight `w ≡ 1` here)
gives (Eq. 8):

```
B ȧ = M a                                                        (Eq. 11)
Bₖⱼ = ⟨Φₖ, Φⱼ⟩                                                   (Eq. 9)
Mₖⱼ = ⟨Φₖ, −Σᵢ ∂(fᵢΦⱼ)/∂xᵢ + Σᵢ,ₗ ∂²(DᵢₗΦⱼ)/(∂xᵢ∂xₗ)⟩             (Eq. 10)
```

Code: `fpe.basis.TensorBSplineBasis.gram_kron()` builds `B` **exactly** as the
Kronecker product of the per-dimension Gram matrices (the basis is
separable); `fpe._core.assemble_M` computes `M` by quadrature. The expanded
integrand implemented in `cpp/include/fpe/assembly.hpp` is, per pair `(k, j)`:

```
Φₖ · [ −f·∇Φⱼ − (∇·f) Φⱼ + Σᵢ,ₗ Dᵢₗ ∂²Φⱼ/(∂xᵢ∂xₗ)
       + 2 Σᵢ (Σₗ ∂Dᵢₗ/∂xₗ) ∂Φⱼ/∂xᵢ + (Σᵢ,ₗ ∂²Dᵢₗ/(∂xᵢ∂xₗ)) Φⱼ ]
```

with the last two terms vanishing for constant diffusion.

Initial coefficients (Eq. 12): `cₖ = ⟨Φₖ, p₀⟩` computed by quadrature, then
`a₀` solves `B a₀ = c` — the correct L2 projection for a *non-orthonormal*
basis (Eq. 12 is exact as written only when `B = I`). Propagation (Eq. 13):
`a(t) = expm(B⁻¹M t) a₀`, implemented densely (Padé-13, scaling & squaring)
or matrix-free (Arnoldi/Krylov `expm`-action with adaptive sub-stepping,
sparse LDLT of `B`).

## 3. B-spline bases (paper Sec. 4.1, Eqs. 14–17)

Order-`k` (degree `k−1`) B-splines on a clamped uniform knot vector,
default `k = 3` as in the paper. Values/derivatives are computed with the
numerically stable triangular scheme (Piegl & Tiller A2.2–A2.3), which is
algebraically identical to the Cox–de Boor recursion (Eq. 14) and the
derivative formulas (Eqs. 16–17).

Sparsity: basis `i` is supported on `[tᵢ, tᵢ₊ₖ]`, so any product
`Φₖ Φⱼ` (or its derivatives) vanishes unless `|kᵢ − jᵢ| < k` in **every**
dimension — the paper's Sec. 4.1 condition (`≥ 3` differences give zero for
`k = 3`). `B` and `M` are therefore banded Kronecker-structured sparse
matrices; the number of nonzeros per row is `(2k−1)ⁿ`, independent of `N`.

## 4. Integrals (paper Sec. 4.2)

Two quadratures feed the same assembly kernel:

- **Halton quasi-Monte Carlo** (the paper's choice, Eq. 19): equal-weight
  points over the whole box; scales to higher dimensions.
- **Tensor Gauss–Legendre per knot span** (`quadrature="gauss"`): exact for
  `B` (polynomial integrand) and near-exact for smooth `f`; recommended in
  low dimension.

Assembly is element-grouped: all points inside one knot-span element share
the same `kⁿ` active basis functions, so each point adds a rank-1 update to a
dense local block and each element scatters once into the global sparse
matrix. Cost: `O(n_points · k²ⁿ)`, independent of `N`, multithreaded.

## 5. Derivatives of the dynamics (paper Sec. 3.2)

`M` needs `∇·f` at every quadrature point. Built-in C++ dynamics use
**forward-mode automatic differentiation with dual numbers**
(`cpp/include/fpe/dual.hpp`) — the C++ analogue of the paper's JAX forward
mode — including through the orbit-average quadrature of the equinoctial
model. Python dynamics can supply the divergence analytically, by finite
differences, or exactly via `fpe.dynamics.from_jax`.

## 6. Boundary treatment (this implementation)

The truncated-domain FPE is meaningful only while the pdf effectively
vanishes at the box boundary. On top of that, the *unconstrained* Galerkin
operator on a box has genuine unstable eigenmodes that the real-line problem
excludes by integrability: e.g. for the 1D OU drift `f = −θx`,
`L[xᵐ] = θ(m+1)xᵐ`, so polynomial modes grow like `e^{θ(m+1)t}`. Any
projection error seeding them is amplified exponentially.

`FokkerPlanckSolver(boundary="dirichlet")` (default) removes these modes by
restricting the Galerkin space to basis functions that vanish on the
boundary — for clamped B-splines this is exactly "drop the first/last basis
function of each dimension". The restricted OU spectrum reproduces the exact
FPE spectrum `0, −θ, −2θ, …` to machine precision (see
`tests/test_fpe_analytic.py::test_dirichlet_removes_unstable_boundary_modes`).
`boundary="free"` recovers the paper's original unconstrained formulation.

## 7. Probability metrics (paper Sec. 4.3, Eqs. 20–21)

`fpe.metrics.hellinger` and `fpe.metrics.kl_divergence` operate on densities
evaluated on a common point set, after the paper's post-processing
(clip at zero, normalize). `fpe.metrics.ks_statistic` provides the
binning-free Kolmogorov–Smirnov distance between Monte Carlo samples and a
reference CDF — with the DKW inequality it turns "MC agrees with the FPE"
into a quantitative statement: `KS·√N` must stay at the DKW constant until
the FPE discretization floor is reached (see `examples/07_mc_convergence.py`). `FokkerPlanckSolver.integral` computes the exact
integral of the spline pdf (a Kronecker contraction of 1D integrals) — the
paper's quality monitor; `moments` and `marginal` similarly return exact
means/covariances and exact 1D marginals of the approximation: because the
basis is separable, integrating dimensions out is a contraction with the
per-dimension basis integrals, not a numerical quadrature.