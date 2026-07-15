"""Tensor-product B-spline basis (paper Sec. 4.1)."""

from __future__ import annotations

from functools import reduce
from typing import List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from . import _core

IntOrSeq = Union[int, Sequence[int]]


def _per_dim(value: IntOrSeq, dim: int, name: str) -> List[int]:
    if np.isscalar(value):
        return [int(value)] * dim
    values = [int(v) for v in value]
    if len(values) != dim:
        raise ValueError(f"{name} must be a scalar or have one entry per dimension")
    return values


class TensorBSplineBasis:
    """Multivariate B-spline basis built as a tensor product of 1D bases.

    Parameters
    ----------
    domain:
        Sequence of ``(lo, hi)`` intervals, one per state dimension. The
        approximated pdf is assumed to (numerically) vanish at the domain
        boundary, so choose the box large enough to contain the pdf over the
        whole propagation horizon.
    n_basis:
        Number of basis functions per dimension (scalar or per-dimension).
    order:
        B-spline order ``k`` (polynomial degree ``k - 1``); the paper uses
        ``k = 3``. Scalar or per-dimension.

    Notes
    -----
    Multi-indices are flattened in C (row-major) order, so a coefficient
    vector ``a`` reshaped to :attr:`shape` recovers the tensor layout.
    """

    def __init__(self, domain: Sequence[Tuple[float, float]], n_basis: IntOrSeq, order: IntOrSeq = 3):
        domain = [(float(lo), float(hi)) for lo, hi in domain]
        dim = len(domain)
        self.domain = domain
        self.n_basis = _per_dim(n_basis, dim, "n_basis")
        self.order = _per_dim(order, dim, "order")
        lo = [d[0] for d in domain]
        hi = [d[1] for d in domain]
        self._cb = _core.TensorBasis(lo, hi, self.n_basis, self.order)

    # -- structure ---------------------------------------------------------
    @property
    def dim(self) -> int:
        return self._cb.dim

    @property
    def n_total(self) -> int:
        """Total number of multivariate basis functions N (paper Eq. 5)."""
        return int(self._cb.n_total)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._cb.shape)

    @property
    def volume(self) -> float:
        return float(np.prod([hi - lo for lo, hi in self.domain]))

    def spline(self, d: int) -> "_core.BSpline1D":
        """The 1D B-spline basis of dimension ``d``."""
        return self._cb.spline(d)

    # -- quadrature --------------------------------------------------------
    def element_quadrature(self, q: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tensor Gauss-Legendre points/weights, ``q`` per knot span per dim.

        Exact for the Gram matrix when ``q >= order`` and the recommended
        rule in low dimension; cost grows as ``(spans * q)^dim``.
        """
        X, W = self._cb.element_quadrature(int(q))
        return np.asarray(X), np.asarray(W)

    def halton_quadrature(self, n_points: int, skip: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Quasi-Monte Carlo (Halton) points/weights over the domain box.

        Equal-weight rule ``I(f) ~ V/n sum f(x_i)`` (paper Sec. 4.2, Eq. 19);
        scales to higher dimensions where tensor quadrature is unaffordable.
        """
        u = np.asarray(_core.halton(int(n_points), self.dim, int(skip)))
        lo = np.array([d[0] for d in self.domain])
        hi = np.array([d[1] for d in self.domain])
        X = lo + u * (hi - lo)
        W = np.full(int(n_points), self.volume / float(n_points))
        return X, W

    # -- exact integrals ---------------------------------------------------
    def grams(self) -> List[np.ndarray]:
        """Per-dimension exact Gram matrices ``G_d[i, j] = <phi_i, phi_j>``."""
        return [np.asarray(self._cb.spline(d).gram()) for d in range(self.dim)]

    def gram_kron(self) -> sp.csc_matrix:
        """Multivariate Gram matrix ``B = kron(G_0, G_1, ...)`` (paper Eq. 9).

        Exact thanks to the separability of the tensor-product basis; sparse
        and banded because 1D basis functions overlap only within ``order``
        neighbours (paper Sec. 4.1).
        """
        mats = [sp.csc_matrix(g) for g in self.grams()]
        return reduce(lambda a, b: sp.kron(a, b, format="csc"), mats)

    def integral_tables(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per-dimension exact ``(I0, I1, I2)`` moment tables of the basis."""
        return [tuple(np.asarray(v) for v in self._cb.spline(d).integrals()) for d in range(self.dim)]

    # -- evaluation --------------------------------------------------------
    def evaluate(self, a: np.ndarray, X: np.ndarray, n_threads: int = 0) -> np.ndarray:
        """Evaluate ``p(x) = sum_j a_j Phi_j(x)`` at each row of ``X``."""
        a = np.ascontiguousarray(a, dtype=float).ravel()
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=float)
        return np.asarray(_core.evaluate_pdf(self._cb, a, X, int(n_threads)))

    # -- (de)serialization ---------------------------------------------------
    def spec(self) -> dict:
        return {
            "domain": self.domain,
            "n_basis": self.n_basis,
            "order": self.order,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "TensorBSplineBasis":
        return cls(spec["domain"], spec["n_basis"], spec["order"])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TensorBSplineBasis(dim={self.dim}, shape={self.shape}, "
            f"order={self.order}, N={self.n_total})"
        )
