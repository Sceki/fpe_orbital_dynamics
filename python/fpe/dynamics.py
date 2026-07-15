"""Drift models f(x) for the Fokker-Planck drift term.

Built-in C++ models (fast, exact divergence via forward-mode AD):

- :class:`DampedOscillator` -- stochastic damped harmonic oscillator
  (paper Sec. 5.1).
- :class:`EquinoctialAveragedDrag` -- orbit-averaged equinoctial dynamics
  with in-plane atmospheric drag (paper Sec. 5.2, Eqs. 24-27).

Arbitrary dynamics can be supplied as vectorized Python callables via
:class:`CallableDynamics` (divergence given analytically, by finite
differences, or exactly through JAX with :func:`from_jax`, mirroring the
paper's use of automatic differentiation).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ._core import DampedOscillator, EquinoctialAveragedDrag
from .separable import SeparableDynamics, SeparableTerm

__all__ = [
    "DampedOscillator",
    "EquinoctialAveragedDrag",
    "CallableDynamics",
    "SeparableDynamics",
    "SeparableTerm",
    "from_jax",
]


class CallableDynamics:
    """Wrap a vectorized Python drift ``f`` (and optionally its divergence).

    Parameters
    ----------
    f:
        Callable mapping states ``X`` of shape ``(n_pts, dim)`` to drifts of
        shape ``(n_pts, dim)``.
    div_f:
        Optional callable mapping ``X`` to ``sum_i df_i/dx_i`` of shape
        ``(n_pts,)``. If omitted, the divergence is approximated with
        central finite differences (2*dim extra evaluations of ``f``).
    dim:
        State dimension.
    fd_step:
        Relative step for the finite-difference divergence.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        div_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dim: Optional[int] = None,
        fd_step: float = 1e-6,
    ):
        if dim is None:
            raise ValueError("CallableDynamics requires the state dimension `dim`")
        self.f = f
        self.div_f = div_f
        self.dim = int(dim)
        self.fd_step = float(fd_step)

    def eval_batch(self, X: np.ndarray, n_threads: int = 0):
        del n_threads  # vectorized NumPy call; threading is up to the callable
        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = np.asarray(self.f(X), dtype=float)
        if F.shape != X.shape:
            raise ValueError(f"f(X) must return shape {X.shape}, got {F.shape}")
        if self.div_f is not None:
            div = np.asarray(self.div_f(X), dtype=float).ravel()
            if div.shape != (X.shape[0],):
                raise ValueError("div_f(X) must return one value per point")
        else:
            div = np.zeros(X.shape[0])
            for i in range(self.dim):
                h = self.fd_step * (1.0 + np.abs(X[:, i]))
                Xp = X.copy()
                Xm = X.copy()
                Xp[:, i] += h
                Xm[:, i] -= h
                div += (np.asarray(self.f(Xp))[:, i] - np.asarray(self.f(Xm))[:, i]) / (2.0 * h)
        return F, div


def from_jax(f, dim: int) -> CallableDynamics:
    """Build a :class:`CallableDynamics` with an exact, JAX-derived divergence.

    ``f`` must map a single state of shape ``(dim,)`` to a drift of shape
    ``(dim,)`` using ``jax.numpy`` operations. Vectorization over points and
    the divergence (trace of the forward-mode Jacobian, paper Sec. 3.2) are
    generated automatically.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "from_jax requires the optional dependency jax: pip install 'fpe[jax]'"
        ) from exc

    f_batch = jax.jit(jax.vmap(f))
    div_single = lambda x: jnp.trace(jax.jacfwd(f)(x))  # noqa: E731
    div_batch = jax.jit(jax.vmap(div_single))

    return CallableDynamics(
        f=lambda X: np.asarray(f_batch(jnp.asarray(X))),
        div_f=lambda X: np.asarray(div_batch(jnp.asarray(X))),
        dim=dim,
    )
