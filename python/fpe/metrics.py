"""Probability metrics (paper Sec. 4.3, Eqs. 20-21).

Both metrics operate on densities evaluated on a common set of points. As in
the paper, the discrete densities are post-processed first, clipped at zero
and normalized to unit (discrete) mass, since the Galerkin approximation
does not enforce non-negativity or unit integral by construction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["postprocess", "hellinger", "kl_divergence", "ks_statistic"]


def postprocess(p: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Clip negative values and normalize to unit mass.

    With ``weights`` (quadrature weights of the evaluation points), the
    continuous integral ``sum w_i p_i`` is normalized to one; without, the
    plain discrete sum is normalized (as in the paper's grid-based metrics).
    """
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    mass = float(np.sum(p * weights)) if weights is not None else float(np.sum(p))
    if mass <= 0.0:
        raise ValueError("density is non-positive everywhere; cannot normalize")
    return p / mass


def hellinger(p: np.ndarray, q: np.ndarray, normalize: bool = True) -> float:
    """Hellinger distance (paper Eq. 20); 0 <= H <= 1, 0 iff identical."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if normalize:
        p = postprocess(p)
        q = postprocess(q)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def ks_statistic(samples: np.ndarray, cdf) -> float:
    """Kolmogorov-Smirnov statistic between samples and a reference CDF.

    ``D_n = sup_x |ECDF_n(x) - F(x)|`` with ``F`` given as a callable (or a
    pre-evaluated array aligned with ``sort(samples)``). Binning-free, so it
    is the natural statistic for validating a propagated pdf against Monte
    Carlo samples: by the DKW inequality ``D_n`` decays as ``~1/sqrt(n)``
    when the samples are drawn from ``F``, until it hits the floor set by
    the approximation error of ``F`` itself.
    """
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = x.size
    if n == 0:
        raise ValueError("samples must be non-empty")
    F = np.asarray(cdf(x) if callable(cdf) else cdf, dtype=float)
    if F.shape != (n,):
        raise ValueError("cdf must yield one value per sample")
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1) / n)
    return float(max(d_plus, d_minus))


def kl_divergence(p: np.ndarray, q: np.ndarray, normalize: bool = True, eps: float = 1e-300) -> float:
    """Kullback-Leibler divergence D_KL(p || q) (paper Eq. 21).

    Terms with ``p == 0`` contribute zero; ``q`` is floored at ``eps`` to
    avoid division by zero where the approximation underflows.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if normalize:
        p = postprocess(p)
        q = postprocess(q)
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))
