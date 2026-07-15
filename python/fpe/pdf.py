"""Probability density helpers."""

from __future__ import annotations

import numpy as np


class GaussianPDF:
    """Multivariate Gaussian density, callable on ``(n_pts, dim)`` arrays.

    Implemented via the correlation matrix (states are standardized by their
    marginal standard deviations first), so it stays numerically exact for
    the extreme scale disparities typical of orbital states -- e.g. a
    semi-major axis in km next to equinoctial elements of order 1e-4, where
    covariance condition numbers overflow double-precision PSD checks.
    """

    def __init__(self, mean, cov):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 1:
            cov = np.diag(cov)
        if cov.shape != (self.mean.size, self.mean.size):
            raise ValueError("cov must be (dim, dim) or a length-dim diagonal")
        if not np.allclose(cov, cov.T):
            raise ValueError("cov must be symmetric")
        self.cov = cov
        self._scale = np.sqrt(np.diag(cov))
        if np.any(self._scale <= 0.0):
            raise ValueError("cov must have strictly positive diagonal")
        corr = cov / np.outer(self._scale, self._scale)
        try:
            self._chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("cov is not positive definite") from exc
        # log of the normalization constant
        self._log_norm = (
            -0.5 * self.mean.size * np.log(2.0 * np.pi)
            - np.sum(np.log(self._scale))
            - np.sum(np.log(np.diag(self._chol)))
        )

    @property
    def dim(self) -> int:
        return self.mean.size

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1 and self.dim == 1:
            X = X[:, None]  # 1D density called on an array of positions
        X = np.atleast_2d(X)
        if X.shape[1] != self.dim:
            raise ValueError(f"points must have {self.dim} columns")
        z = (X - self.mean) / self._scale
        # Solve L y = z^T for the standardized Mahalanobis distance.
        y = np.linalg.solve(self._chol, z.T)
        return self._log_norm - 0.5 * np.sum(y * y, axis=0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(X))

    def sample(self, n: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.dim))
        return self.mean + (z @ self._chol.T) * self._scale

    def __repr__(self) -> str:  # pragma: no cover
        return f"GaussianPDF(mean={self.mean.tolist()})"
