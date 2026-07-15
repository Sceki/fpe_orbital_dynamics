"""Sum-of-separables (Kronecker) form of the Galerkin FPE operator.

The reason the assembled matrix M dies beyond ~4 dimensions is storage and
assembly work: ``(2k-1)^n`` nonzeros per row and ``k^{2n}`` operations per
quadrature point (something that can probably be improved as well).
But whenever the drift is a *sum of separable terms*,

    f_i(x) = sum_r  c_r  prod_d  g_{r,d}(x_d)          (exact for polynomials),

each term's Galerkin contribution factorizes into a Kronecker product of
small 1D matrices computed by *one-dimensional* quadrature:

    -<Phi_k, d/dx_i (f_i Phi_j)>  =  -c_r  prod_d  W_d[k_d, j_d],
    W_d = / int phi_k g (phi_j)' + phi_k g' phi_j dx   (d = i)
          \\ int phi_k g phi_j dx                       (d != i),

and likewise for constant diffusion (plain first/second-derivative
matrices). M is then never assembled: it is stored as a few small factors
and applied matrix-free, dimension by dimension, the same structure the
solver already exploits for the Gram matrix B. Combined with the Krylov
expm-action, this removes both walls at once and makes 6-7 dimensions
practical for polynomial (or polynomial-fitted) dynamics.

The Dirichlet boundary restriction slices every 1D factor independently
(the active index set is a Cartesian product), so the Kronecker structure
survives it, exactly as for B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.polynomial import Polynomial

__all__ = ["SeparableTerm", "SeparableDynamics", "KroneckerOperator"]

# A 1D factor: a numpy Polynomial (differentiable -> usable in the derivative
# dimension), a (g, g') pair of callables, or a plain callable (usable only in
# non-derivative dimensions).
FactorSpec = Union[Polynomial, Tuple[Callable, Callable], Callable]


def _g_pair(g: Optional[FactorSpec]):
    if g is None:
        return None, None
    if isinstance(g, Polynomial):
        return g, g.deriv()
    if isinstance(g, tuple):
        if len(g) != 2:
            raise ValueError("a factor tuple must be (g, gprime)")
        return g
    return g, None


@dataclass
class SeparableTerm:
    """One separable drift term: contributes ``coeff * prod_d g_d(x_d)`` to
    component ``f_{dim_out}`` of the drift. Dimensions absent from
    ``factors`` have ``g_d = 1``."""

    dim_out: int
    coeff: float
    factors: Dict[int, FactorSpec]


class SeparableDynamics:
    """Drift ``f(x)`` given as a sum of separable terms.

    Every polynomial drift is exactly of this form; smooth non-polynomial
    dynamics can be fitted per dimension (Chebyshev/Taylor) first.
    """

    def __init__(self, dim: int, terms: Sequence[SeparableTerm]):
        self.dim = int(dim)
        self.terms = list(terms)
        for t in self.terms:
            if not 0 <= t.dim_out < self.dim:
                raise ValueError("term dim_out out of range")
            if any(not 0 <= d < self.dim for d in t.factors):
                raise ValueError("factor dimension out of range")

    @classmethod
    def linear(cls, A) -> "SeparableDynamics":
        """f(x) = A x, term by nonzero matrix entry."""
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        x = Polynomial([0.0, 1.0])
        terms = [
            SeparableTerm(i, A[i, j], {j: x})
            for i in range(A.shape[0])
            for j in range(A.shape[1])
            if A[i, j] != 0.0
        ]
        return cls(A.shape[0], terms)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the drift at points (n_pts, dim) -- for Monte Carlo and
        cross-checks against the quadrature-assembled path."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = np.zeros_like(X)
        for t in self.terms:
            prod = np.full(X.shape[0], t.coeff)
            for d, g in t.factors.items():
                gf, _ = _g_pair(g)
                prod = prod * np.asarray(gf(X[:, d]), dtype=float)
            F[:, t.dim_out] += prod
        return F


class KroneckerOperator:
    """``sum_r c_r  (x)_d A_{r,d}`` acting on flattened row-major tensors.

    Factors equal to the identity are simply omitted from a term's list;
    application is dimension-wise (``O(N n_d)`` per factor), so the matvec
    costs ``O(R N n)`` instead of touching ``(2k-1)^n N`` stored entries.
    """

    def __init__(self, shape: Sequence[int],
                 terms: Sequence[Tuple[float, Sequence[Tuple[int, np.ndarray]]]]):
        self.shape = tuple(int(s) for s in shape)
        self.n = int(np.prod(self.shape))
        self.terms = []
        for coeff, mats in terms:
            checked = []
            for d, Md in mats:
                Md = np.ascontiguousarray(Md, dtype=float)
                if Md.shape != (self.shape[d], self.shape[d]):
                    raise ValueError(f"factor for dim {d} has shape {Md.shape}, "
                                     f"expected {(self.shape[d],) * 2}")
                checked.append((int(d), Md))
            self.terms.append((float(coeff), checked))

    def matvec(self, v: np.ndarray) -> np.ndarray:
        T0 = np.asarray(v, dtype=float).reshape(self.shape)
        out = np.zeros(self.shape)
        for coeff, mats in self.terms:
            T = T0
            for d, Md in mats:
                T = np.moveaxis(np.tensordot(Md, np.moveaxis(T, d, 0), axes=(1, 0)), 0, d)
            out += coeff * T
        return out.ravel()

    def to_dense(self, max_n: int = 5000) -> np.ndarray:
        """Materialize the full matrix (tests / small problems only)."""
        if self.n > max_n:
            raise ValueError(f"refusing to densify a {self.n} x {self.n} operator")
        A = np.zeros((self.n, self.n))
        for coeff, mats in self.terms:
            full = [np.eye(s) for s in self.shape]
            for d, Md in mats:
                full[d] = Md
            term = full[0]
            for f in full[1:]:
                term = np.kron(term, f)
            A += coeff * term
        return A


# ---------------------------------------------------------------------- #
# 1D weighted Galerkin matrices (per-span Gauss-Legendre quadrature)
# ---------------------------------------------------------------------- #
def span_quadrature(spline, q: int):
    """Gauss-Legendre points/weights, ``q`` per non-empty knot span."""
    knots = np.asarray(spline.knots)
    p = spline.order - 1
    gx, gw = np.polynomial.legendre.leggauss(int(q))
    xs, ws = [], []
    for i in range(p, spline.n_basis):
        a, b = knots[i], knots[i + 1]
        if b <= a:
            continue
        xs.append(0.5 * (a + b) + 0.5 * (b - a) * gx)
        ws.append(0.5 * (b - a) * gw)
    return np.concatenate(xs), np.concatenate(ws)


def build_kron_terms(splines, dynamics: "SeparableDynamics", D, interior: bool,
                     q: int, sink_1d=None):
    """Kronecker-term representation of the restricted operator ``B^{-1} M``.

    Used by the full-tensor solver (:meth:`FokkerPlanckSolver.
    assemble_separable`). Dimensions untouched by a term contribute their
    1D Gram matrix to ``M``;
    pre-multiplying every factor by the (restricted) ``G_d^{-1}`` turns
    those into true identities, so untouched dimensions cost nothing.

    Returns ``(shape, terms)`` with ``terms = [(coeff, [(d, matrix), ...])]``.
    """
    from scipy.linalg import cho_factor, cho_solve

    dim = len(splines)
    if dynamics.dim != dim:
        raise ValueError(f"dynamics has dim {dynamics.dim}, basis has {dim}")
    D = np.zeros((dim, dim)) if D is None else np.asarray(D, dtype=float)
    sl = slice(1, -1) if interior else slice(None)
    quads = [Basis1DQuadrature(sp, q) for sp in splines]
    chols = [cho_factor(np.asarray(sp.gram())[sl, sl]) for sp in splines]
    pre = lambda d, Md: cho_solve(chols[d], Md)  # noqa: E731

    terms = []
    # drift: -<Phi_k, d/dx_i (f_i Phi_j)>, one Kronecker term per separable
    # term (the derivative dimension always carries a factor)
    for t in dynamics.terms:
        mats = []
        for d in range(dim):
            if d == t.dim_out:
                mats.append((d, pre(d, quads[d].advect(t.factors.get(d))[sl, sl])))
            elif d in t.factors:
                mats.append((d, pre(d, quads[d].mass(t.factors[d])[sl, sl])))
        terms.append((-t.coeff, mats))
    # constant diffusion: sum_il D_il <Phi_k, d2 Phi_j / dx_i dx_l>
    for i in range(dim):
        for l in range(i, dim):
            Dil = D[i, i] if i == l else D[i, l] + D[l, i]
            if Dil == 0.0:
                continue
            if i == l:
                terms.append((Dil, [(i, pre(i, quads[i].diff2()[sl, sl]))]))
            else:
                terms.append((Dil, [(i, pre(i, quads[i].diff1()[sl, sl])),
                                    (l, pre(l, quads[l].diff1()[sl, sl]))]))
    # separable absorption (boundary sponge / first-passage sink)
    if sink_1d is not None:
        for d in range(dim):
            sd = quads[d].mass((lambda x, d=d: sink_1d(x, d), None))[sl, sl]
            terms.append((-1.0, [(d, pre(d, sd))]))

    shape = tuple((sp.n_basis - 2) if interior else sp.n_basis for sp in splines)
    return shape, terms


class Basis1DQuadrature:
    """Cached basis values/derivatives of one 1D spline at quadrature points."""

    def __init__(self, spline, q: int):
        self.x, self.w = span_quadrature(spline, q)
        self.B0 = np.asarray(spline.basis_matrix(self.x, 0))
        self.B1 = np.asarray(spline.basis_matrix(self.x, 1))
        self.B2 = np.asarray(spline.basis_matrix(self.x, 2))

    def _wvals(self, g) -> np.ndarray:
        gf, _ = _g_pair(g)
        return self.w if gf is None else self.w * np.asarray(gf(self.x), dtype=float)

    def mass(self, g=None) -> np.ndarray:
        """int phi_k g phi_j dx."""
        return np.einsum("pk,p,pj->kj", self.B0, self._wvals(g), self.B0)

    def advect(self, g=None) -> np.ndarray:
        """int phi_k d/dx( g phi_j ) dx  (g = 1 if None)."""
        gf, gp = _g_pair(g)
        if gf is None:
            return np.einsum("pk,p,pj->kj", self.B0, self.w, self.B1)
        if gp is None:
            raise ValueError(
                "the factor in the derivative dimension needs a derivative: "
                "pass a numpy Polynomial or a (g, gprime) tuple"
            )
        out = np.einsum("pk,p,pj->kj", self.B0, self.w * np.asarray(gp(self.x), float), self.B0)
        out += np.einsum("pk,p,pj->kj", self.B0, self.w * np.asarray(gf(self.x), float), self.B1)
        return out

    def diff1(self) -> np.ndarray:
        """int phi_k phi_j' dx."""
        return np.einsum("pk,p,pj->kj", self.B0, self.w, self.B1)

    def diff2(self) -> np.ndarray:
        """int phi_k phi_j'' dx."""
        return np.einsum("pk,p,pj->kj", self.B0, self.w, self.B2)
