"""Galerkin Fokker-Planck solver (paper Secs. 3-4).

The pdf is expanded as ``p(x, t) = sum_j a_j(t) Phi_j(x)`` on a tensor
B-spline basis; Galerkin projection turns the Fokker-Planck PDE into the
linear ODE system ``B da/dt = M a`` (Eq. 11), solved through the matrix
exponential ``a(t) = expm(B^{-1} M t) a0`` (Eq. 13).

The expensive spatial integrals live entirely in ``B`` and ``M``: assemble
them once (offline), then propagate *any* initial pdf to *any* future time at
negligible cost -- the key practical advantage highlighted in the paper.
:meth:`FokkerPlanckSolver.save` / :meth:`FokkerPlanckSolver.load` persist the
assembled operator for exactly this offline/online split.
"""

from __future__ import annotations

import json
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

from . import _core
from .basis import TensorBSplineBasis
from .dynamics import CallableDynamics

__all__ = ["FokkerPlanckSolver"]

# Below this size the dense expm path is typically faster than Krylov.
_DENSE_N_MAX = 1200


class FokkerPlanckSolver:
    """Galerkin-projected Fokker-Planck propagator.

    Parameters
    ----------
    basis:
        A :class:`~fpe.TensorBSplineBasis` covering the region where the pdf
        lives over the whole propagation horizon.
    dynamics:
        Deterministic drift ``f(x)``: a built-in C++ model from
        :mod:`fpe.dynamics`, a :class:`~fpe.dynamics.CallableDynamics`, a
        plain vectorized callable ``X -> F`` (divergence then by finite
        differences), or ``None`` for pure diffusion.
    diffusion:
        Constant diffusion matrix ``D = sigma sigma^T / 2`` (paper Eq. 2),
        shape ``(dim, dim)``. Mutually exclusive with ``sigma``.
    sigma:
        Constant noise matrix ``sigma`` of the SDE ``dX = f dt + sigma dW``
        (shape ``(dim, m)``); ``D`` is formed internally.
    boundary:
        ``"dirichlet"`` (default) restricts the Galerkin space to basis
        functions vanishing on the domain boundary, enforcing ``p = 0`` on
        the box edge. ``"free"`` keeps the full, unconstrained space of the
        paper's original formulation.

    Notes
    -----
    As required by the method (paper Sec. 3.1), neither the drift nor the
    diffusion may depend explicitly on time.

    The pdf must (numerically) vanish at the domain boundary for the
    truncated-domain FPE to be meaningful. With ``boundary="free"`` the
    discretized operator additionally carries spurious *unstable* modes
    supported near the boundary (e.g. the polynomial eigenfunctions
    ``x^m`` of the advection operator, which the real-line FPE excludes by
    integrability); any projection error seeding them grows exponentially.
    ``boundary="dirichlet"`` removes them by construction -- with clamped
    B-splines, dropping the first/last basis function of each dimension is
    exactly the ``p|_boundary = 0`` constraint.
    """

    def __init__(
        self,
        basis: TensorBSplineBasis,
        dynamics=None,
        diffusion=None,
        sigma=None,
        boundary: str = "dirichlet",
    ):
        self.basis = basis
        self.dynamics = self._wrap_dynamics(dynamics)
        if diffusion is not None and sigma is not None:
            raise ValueError("pass either `diffusion` (= sigma sigma^T / 2) or `sigma`, not both")
        n = basis.dim
        if sigma is not None:
            s = np.atleast_2d(np.asarray(sigma, dtype=float))
            if s.shape[0] != n:
                raise ValueError(f"sigma must have {n} rows")
            D = 0.5 * s @ s.T
        elif diffusion is not None:
            D = np.asarray(diffusion, dtype=float)
            if D.shape != (n, n):
                raise ValueError(f"diffusion must have shape ({n}, {n})")
            if not np.allclose(D, D.T):
                raise ValueError("diffusion matrix must be symmetric")
        else:
            D = np.zeros((n, n))
        self.D = D

        if boundary not in ("dirichlet", "free"):
            raise ValueError("boundary must be 'dirichlet' or 'free'")
        if boundary == "dirichlet" and any(nb < 3 for nb in basis.n_basis):
            raise ValueError("boundary='dirichlet' needs at least 3 basis functions per dimension")
        self.boundary = boundary

        self.B: Optional[sp.csc_matrix] = None
        self.M: Optional[sp.csc_matrix] = None
        self._op_kron = None  # KroneckerOperator: B^{-1} M on the active subspace
        self._reset_caches()

    def _reset_caches(self) -> None:
        self._krylov = None
        self._restricted = None
        self._kron_chol = None
        self._mass_vector: Optional[np.ndarray] = None

    @property
    def active_indices(self) -> np.ndarray:
        """Flat indices of the basis functions the solver evolves.

        All of them for ``boundary="free"``; the interior ones (multi-index
        components in ``1 .. N_d - 2``) for ``boundary="dirichlet"``.
        """
        if self.boundary == "free":
            return np.arange(self.n)
        idx = np.arange(self.n).reshape(self.basis.shape)
        return idx[tuple(slice(1, -1) for _ in range(self.dim))].ravel()

    def _restricted_matrices(self):
        """Sparse B and M restricted to the active subspace (cached).

        Only the sparse-matrix code paths need this; the Kronecker paths
        (projection, dim >= 4 propagation, separable operators) never build
        the assembled B, whose Kronecker of 1D Grams explodes in nnz beyond
        ~4 dimensions.
        """
        if self._restricted is None:
            if self.B is None:
                self.B = self.basis.gram_kron()
            idx = self.active_indices
            if self.boundary == "free":
                Bres = self.B.tocsc()
                Mres = self.M.tocsc() if self.M is not None else None
            else:
                Bres = self.B.tocsr()[idx][:, idx].tocsc()
                Mres = self.M.tocsr()[idx][:, idx].tocsc() if self.M is not None else None
            self._restricted = (idx, Bres, Mres)
        return self._restricted

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _wrap_dynamics(self, dynamics):
        if dynamics is None or hasattr(dynamics, "eval_batch"):
            return dynamics
        if callable(dynamics):
            return CallableDynamics(dynamics, dim=self.basis.dim)
        raise TypeError("dynamics must be None, a fpe.dynamics model, or a callable")

    def _quadrature(self, quadrature: str, q: Optional[int], n_points: Optional[int], halton_skip: int):
        if quadrature == "gauss":
            if q is None:
                q = max(self.basis.order) + 1
            return self.basis.element_quadrature(q)
        if quadrature == "halton":
            if n_points is None:
                n_points = 200_000
            return self.basis.halton_quadrature(n_points, skip=halton_skip)
        raise ValueError("quadrature must be 'gauss' or 'halton'")

    def assemble(
        self,
        quadrature: str = "gauss",
        q: Optional[int] = None,
        n_points: Optional[int] = None,
        halton_skip: int = 1,
        n_threads: int = 0,
        check_coverage: bool = True,
    ) -> "FokkerPlanckSolver":
        """Assemble the Galerkin matrices ``B`` (Eq. 9) and ``M`` (Eq. 10).

        Parameters
        ----------
        quadrature:
            ``"gauss"`` (tensor Gauss-Legendre per knot span; near-exact for
            smooth dynamics, recommended for dim <= 3-4) or ``"halton"``
            (quasi-Monte Carlo as in the paper, for higher dimensions).
        q:
            Gauss points per span per dimension (default: ``max(order) + 1``).
        n_points:
            Number of Halton points (default 200000).
        n_threads:
            C++ worker threads (0 = all cores).
        check_coverage:
            For Halton quadrature, warn if some knot-span element contains no
            quadrature point (locally under-resolved integrals).

        ``B`` is always computed exactly from the 1D Gram matrices via their
        Kronecker product, exploiting the separability of the basis.
        """
        est_nnz = self.n * int(np.prod([2 * k - 1 for k in self.basis.order]))
        if est_nnz > 2e8:
            raise MemoryError(
                f"the assembled sparse M would hold ~{est_nnz:.1e} nonzeros "
                f"((2k-1)^dim per row); express the drift as a sum of separable "
                f"terms and use assemble_separable() instead"
            )
        X, W = self._quadrature(quadrature, q, n_points, halton_skip)

        if self.dynamics is not None:
            F, divF = self.dynamics.eval_batch(X, n_threads)
            F = np.ascontiguousarray(F, dtype=float)
            divF = np.ascontiguousarray(divF, dtype=float).ravel()
        else:
            F = np.zeros_like(X)
            divF = np.zeros(X.shape[0])

        if quadrature == "halton" and check_coverage:
            counts = np.asarray(_core.points_per_element(self.basis._cb, X))
            n_empty = int(np.sum(counts == 0))
            if n_empty > 0:
                import warnings

                warnings.warn(
                    f"{n_empty}/{counts.size} knot-span elements contain no Halton point; "
                    "increase n_points for a reliable M matrix",
                    stacklevel=2,
                )

        self.M = _core.assemble_M(self.basis._cb, X, W, F, divF, self.D, n_threads=n_threads)
        self.B = self.basis.gram_kron()
        self._op_kron = None
        self._reset_caches()
        return self

    def assemble_separable(self, dynamics, q: Optional[int] = None, sink_1d=None) -> "FokkerPlanckSolver":
        """Assemble M as a sum of Kronecker products -- never materialized.

        For drifts given as sums of separable terms
        (:class:`~fpe.separable.SeparableDynamics`; exact for polynomial and
        in particular linear dynamics), every Galerkin term factorizes into
        small 1D matrices computed by one-dimensional quadrature. Assembly
        cost and operator storage become negligible and the operator is
        applied matrix-free, which is what makes dim >= 5 practical
        (the sparse M carries ``(2k-1)^dim`` nonzeros per row).

        Diffusion: the constant matrix D from the constructor.

        Parameters
        ----------
        q:
            Gauss points per knot span for the 1D integrals (default
            ``max(order) + 4``; exact for the cubic-and-lower polynomial
            factors of linear dynamics).
        sink_1d:
            Optional per-dimension absorption rate ``sigma_d = sink_1d(x, d)``,
            added as ``-sum_d sigma_d(x_d) p``. Use for boundary sponges in
            high dimensions (a general ``add_sink`` would need the full-
            dimensional quadrature this path exists to avoid).

        Notes
        -----
        ``save()``/``add_sink()`` and the QMC/Gauss quadrature ``assemble()``
        features operate on the sparse form and are unavailable in this
        mode. Use :meth:`project_separable` for initial conditions (the
        generic :meth:`project` needs a full-dimensional quadrature).
        """
        from .separable import KroneckerOperator, SeparableDynamics, build_kron_terms

        if not isinstance(dynamics, SeparableDynamics):
            raise TypeError("assemble_separable expects a fpe.separable.SeparableDynamics")
        if q is None:
            q = max(self.basis.order) + 4
        # The stored operator is B^{-1} M directly: dimensions untouched by a
        # term contribute their 1D Gram to M, which G_d^{-1} turns into a true
        # identity -- so untouched dimensions cost nothing in the matvec, and
        # no separate B-solve is needed during propagation.
        splines = [self.basis.spline(d) for d in range(self.dim)]
        shape, terms = build_kron_terms(
            splines, dynamics, self.D, interior=self.boundary == "dirichlet",
            q=q, sink_1d=sink_1d,
        )
        self._op_kron = KroneckerOperator(shape, terms)
        self.M = None
        self._reset_caches()
        return self

    def add_sink(
        self,
        sigma: Callable[[np.ndarray], np.ndarray],
        quadrature: str = "gauss",
        q: Optional[int] = None,
        n_points: Optional[int] = None,
        n_threads: int = 0,
    ) -> "FokkerPlanckSolver":
        """Add an absorption (killing) term ``-sigma(x) p`` to the operator.

        The propagated ``p`` becomes a *sub*-probability density whose mass
        ``integral(a)`` decays where ``sigma > 0``. This is the standard
        tool for first-passage problems: pair a smooth sink ramp inside a
        domain buffer (a "sponge layer") with an event region, and read the
        survival probability off the remaining pdf mass -- the pdf is
        removed inside the domain instead of interacting with the boundary.

        ``sigma`` maps points ``(n_pts, dim)`` to non-negative rates
        ``(n_pts,)``. Must be called after :meth:`assemble`; persisted by
        :meth:`save` like the rest of the operator.
        """
        self._require_assembled()
        if self._op_kron is not None:
            raise RuntimeError(
                "add_sink() operates on the sparse operator; for Kronecker-form "
                "operators pass sink_1d= to assemble_separable() instead"
            )
        if q is None:
            q = max(self.basis.order) + 1
        X, W = self._quadrature(quadrature, q, n_points, 1)
        svals = np.ascontiguousarray(np.asarray(sigma(X), dtype=float).ravel())
        if svals.size != X.shape[0]:
            raise ValueError("sigma(X) must return one rate per quadrature point")
        if np.any(svals < 0):
            raise ValueError("sigma must be non-negative")
        # <Phi_k, -sigma Phi_j> is exactly the -divF Phi_j term of the
        # assembly kernel with zero drift and diffusion.
        M_sink = _core.assemble_M(
            self.basis._cb, X, W, np.zeros_like(X), svals,
            np.zeros((self.dim, self.dim)), n_threads=n_threads,
        )
        self.M = (self.M + M_sink).tocsc()
        self._reset_caches()
        return self

    # ------------------------------------------------------------------ #
    # initial conditions
    # ------------------------------------------------------------------ #
    def project(
        self,
        p0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        quadrature: str = "gauss",
        q: Optional[int] = None,
        n_points: Optional[int] = None,
        n_threads: int = 0,
    ) -> np.ndarray:
        """L2 projection of an initial pdf onto the basis (paper Eq. 12).

        Computes ``c_k = <Phi_k, p0>`` by quadrature and returns the Galerkin
        coefficients from ``B a0 = c`` (for a non-orthonormal basis the Gram
        system must be solved; for orthonormal bases ``B = I`` and
        ``a0 = c``, recovering Eq. 12 verbatim). With
        ``boundary="dirichlet"`` the projection is onto the constrained
        subspace, so the boundary coefficients are exactly zero.

        ``p0`` is a callable evaluated at the quadrature points (e.g. a
        :class:`~fpe.GaussianPDF`) or a coefficient-sized array is passed
        through unchanged.
        """
        self._require_assembled(need_M=False)
        p0_arr = np.asarray(p0, dtype=float) if not callable(p0) else None
        if p0_arr is not None:
            if p0_arr.shape != (self.n,):
                raise ValueError("array initial condition must be a coefficient vector")
            return p0_arr
        if q is None:
            q = max(self.basis.order) + 2
        X, W = self._quadrature(quadrature, q, n_points, 1)
        pvals = np.ascontiguousarray(np.asarray(p0(X), dtype=float).ravel())
        if pvals.size != X.shape[0]:
            raise ValueError("p0(X) must return one density value per quadrature point")
        c = np.asarray(_core.project_rhs(self.basis._cb, X, W, pvals, n_threads))
        idx, _, _ = self._restricted_matrices()
        a = np.zeros(self.n)
        a[idx] = self._solve_B(c[idx])
        return a

    def project_separable(self, marginals: Sequence[Callable[[np.ndarray], np.ndarray]],
                          q: Optional[int] = None) -> np.ndarray:
        """L2 projection of a product-form initial pdf ``p0 = prod_d p_d(x_d)``.

        Because both ``p0`` and ``B`` factorize, the projection reduces to
        one small Gram solve per dimension (``a = kron_d G_d^{-1} c_d``) --
        no full-dimensional quadrature, so this is the initial-condition
        path for high-dimensional problems. Independent (diagonal-
        covariance) initial uncertainties are exactly of this form; each 1D
        factor may be any density (Gaussian, lognormal, ...).
        """
        if len(marginals) != self.dim:
            raise ValueError(f"need one marginal per dimension ({self.dim})")
        if q is None:
            q = max(self.basis.order) + 6
        from .separable import span_quadrature

        chols, shape = self._kron_factors()
        sl = slice(1, -1) if self.boundary == "dirichlet" else slice(None)
        T = None
        for d in range(self.dim):
            spd = self.basis.spline(d)
            x, w = span_quadrature(spd, q)
            B0 = np.asarray(spd.basis_matrix(x, 0))
            c = (B0.T @ (w * np.asarray(marginals[d](x), dtype=float)))[sl]
            a_d = cho_solve(chols[d], c)
            T = a_d if T is None else np.multiply.outer(T, a_d)
        out = np.zeros(self.n)
        out[self.active_indices] = T.ravel()
        return out

    # ------------------------------------------------------------------ #
    # propagation
    # ------------------------------------------------------------------ #
    def propagate(
        self,
        a0: np.ndarray,
        times: Sequence[float],
        method: str = "auto",
        krylov_m: int = 40,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Propagate coefficients to the requested times (paper Eq. 13).

        Parameters
        ----------
        a0:
            Initial coefficients (from :meth:`project`), taken at ``t = 0``.
        times:
            Non-decreasing, non-negative times at which to return ``a(t)``.
        method:
            ``"dense"`` (cache ``expm(B^{-1} M dt)`` per distinct step;
            best for small N), ``"krylov"`` (matrix-free Arnoldi
            ``expm``-action; scales to large sparse systems), or ``"auto"``.

        Returns
        -------
        Array of shape ``(len(times), N)`` with ``a(t_i)`` per row.
        """
        self._require_assembled()
        a0 = np.asarray(a0, dtype=float).ravel()
        if a0.size != self.n:
            raise ValueError(f"a0 must have {self.n} entries")
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty 1D sequence")
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times must be non-negative and non-decreasing")

        idx = self.active_indices
        if method == "auto":
            method = "dense" if idx.size <= _DENSE_N_MAX else "krylov"

        out = np.zeros((times.size, self.n))
        a = a0[idx].copy()  # boundary coefficients (zero by construction) drop out
        prev_t = 0.0
        if method == "dense":
            A = self._dense_operator()
            cache = {}
            for i, t in enumerate(times):
                dt = t - prev_t
                if dt > 0:
                    E = cache.get(dt)
                    if E is None:
                        E = np.asarray(_core.expm(A * dt))
                        cache[dt] = E
                    a = E @ a
                prev_t = t
                out[i, idx] = a
        elif method == "krylov":
            # Matrix-free paths: Kronecker-form operators always; sparse M
            # with the Kronecker B-solve beyond 3D (a sparse LDLT of B
            # suffers severe fill-in there). C++ propagator otherwise.
            if self._op_kron is not None:
                matvec = self._op_kron.matvec  # already B^{-1} M
            elif self.dim >= 4:
                _, _, Mres = self._restricted_matrices()
                matvec = lambda x_: self._solve_B(Mres @ x_)  # noqa: E731
            else:
                matvec = None
                prop = self._krylov_propagator(krylov_m, tol)
            for i, t in enumerate(times):
                dt = t - prev_t
                if dt > 0:
                    if matvec is not None:
                        a = self._expm_action(matvec, a, dt, krylov_m, tol)
                    else:
                        a = np.asarray(prop.apply(a, dt))
                prev_t = t
                out[i, idx] = a
        else:
            raise ValueError("method must be 'auto', 'dense', or 'krylov'")
        return out

    # ------------------------------------------------------------------ #
    # evaluation & diagnostics
    # ------------------------------------------------------------------ #
    def evaluate(self, a: np.ndarray, X: np.ndarray, n_threads: int = 0) -> np.ndarray:
        """Evaluate the approximated pdf at each row of ``X``."""
        return self.basis.evaluate(a, X, n_threads)

    @property
    def n(self) -> int:
        return self.basis.n_total

    @property
    def dim(self) -> int:
        return self.basis.dim

    @property
    def mass_vector(self) -> np.ndarray:
        """Exact ``w`` with ``integral(p) = w . a`` (Kronecker of 1D integrals)."""
        if self._mass_vector is None:
            tables = self.basis.integral_tables()
            w = tables[0][0]
            for t in tables[1:]:
                w = np.kron(w, t[0])
            self._mass_vector = w
        return self._mass_vector

    def integral(self, a: np.ndarray) -> float:
        """Exact integral of the pdf approximation over the domain.

        Monitoring how far this drifts from 1 tracks the quality of the
        representation over time (paper Secs. 3.1 and 5).
        """
        return float(self.mass_vector @ np.asarray(a, dtype=float).ravel())

    def normalize(self, a: np.ndarray) -> np.ndarray:
        """Rescale coefficients so the pdf integrates to one."""
        return np.asarray(a, dtype=float) / self.integral(a)

    def moments(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exact mean and covariance of the (normalized) pdf approximation.

        The tensor-product structure makes every moment a Kronecker
        contraction of per-dimension 1D integrals -- no sampling involved.
        """
        a = np.asarray(a, dtype=float).reshape(self.basis.shape)
        tables = self.basis.integral_tables()
        n = self.dim

        def contract(selector) -> float:
            out = a
            for d in range(n):
                vec = tables[d][selector(d)]
                out = np.tensordot(vec, out, axes=(0, 0))
            return float(out)

        mass = contract(lambda d: 0)
        if mass == 0.0:
            raise ValueError("pdf has zero mass; cannot compute moments")
        mean = np.array([contract(lambda d, i=i: 1 if d == i else 0) for i in range(n)]) / mass
        raw2 = np.empty((n, n))
        for i in range(n):
            for l in range(i, n):
                if i == l:
                    raw2[i, i] = contract(lambda d, i=i: 2 if d == i else 0)
                else:
                    raw2[i, l] = raw2[l, i] = contract(
                        lambda d, i=i, l=l: 1 if d in (i, l) else 0
                    )
        cov = raw2 / mass - np.outer(mean, mean)
        return mean, cov

    def marginal(self, a: np.ndarray, dim, x: np.ndarray) -> np.ndarray:
        """Marginal pdf of one or two state dimensions.

        All other dimensions are integrated out *exactly* (the basis is
        separable, so marginalization is a Kronecker contraction with the
        1D basis integrals), leaving a low-dimensional spline evaluated at
        the requested points.

        Parameters
        ----------
        dim:
            An ``int`` for a 1D marginal (``x`` of shape ``(n_pts,)``), or a
            pair of dimensions for a 2D joint marginal (``x`` of shape
            ``(n_pts, 2)``, columns ordered as ``dim``).
        """
        dims = (dim,) if np.isscalar(dim) else tuple(int(d) for d in dim)
        if len(dims) not in (1, 2) or len(set(dims)) != len(dims):
            raise ValueError("dim must be one dimension or a pair of distinct dimensions")
        if any(not 0 <= d < self.dim for d in dims):
            raise ValueError(f"dimensions must be in [0, {self.dim})")
        T = np.asarray(a, dtype=float).reshape(self.basis.shape)
        tables = self.basis.integral_tables()
        for d in range(self.dim - 1, -1, -1):
            if d in dims:
                continue
            T = np.tensordot(T, tables[d][0], axes=(d, 0))
        # remaining axes of T are ordered by ascending dimension index
        if len(dims) == 1:
            x = np.asarray(x, dtype=float).ravel()
            return self.basis.spline(dims[0]).basis_matrix(x, 0) @ T
        if dims[0] > dims[1]:
            T = T.T
        X = np.atleast_2d(np.asarray(x, dtype=float))
        if X.shape[1] != 2:
            raise ValueError("for a 2D marginal, x must have shape (n_pts, 2)")
        Ba = np.asarray(self.basis.spline(dims[0]).basis_matrix(X[:, 0], 0))
        Bb = np.asarray(self.basis.spline(dims[1]).basis_matrix(X[:, 1], 0))
        return np.einsum("pa,pb,ab->p", Ba, Bb, T)

    # ------------------------------------------------------------------ #
    # persistence: assemble offline once, reuse online
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist basis spec and assembled matrices to a ``.npz`` file.

        A loaded solver can project, propagate, and evaluate without the
        dynamics object: all dynamical information is baked into ``M``.
        """
        self._require_assembled()
        if self._op_kron is not None:
            raise NotImplementedError(
                "save() is not supported for Kronecker-form operators yet; "
                "re-create them with assemble_separable() (assembly is cheap)"
            )
        B = self.B.tocsr()
        M = self.M.tocsr()
        np.savez_compressed(
            path,
            spec=json.dumps(self.basis.spec()),
            boundary=self.boundary,
            D=self.D,
            B_data=B.data,
            B_indices=B.indices,
            B_indptr=B.indptr,
            M_data=M.data,
            M_indices=M.indices,
            M_indptr=M.indptr,
            shape=np.array(B.shape),
        )

    @classmethod
    def load(cls, path: str) -> "FokkerPlanckSolver":
        """Recreate a solver from :meth:`save` output (no dynamics needed)."""
        with np.load(path, allow_pickle=False) as z:
            spec = json.loads(str(z["spec"]))
            basis = TensorBSplineBasis.from_spec(spec)
            boundary = str(z["boundary"]) if "boundary" in z else "dirichlet"
            solver = cls(basis, dynamics=None, diffusion=z["D"], boundary=boundary)
            shape = tuple(z["shape"])
            solver.B = sp.csr_matrix((z["B_data"], z["B_indices"], z["B_indptr"]), shape=shape).tocsc()
            solver.M = sp.csr_matrix((z["M_data"], z["M_indices"], z["M_indptr"]), shape=shape).tocsc()
        return solver

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _require_assembled(self, need_M: bool = True) -> None:
        if need_M and self.M is None and self._op_kron is None:
            raise RuntimeError("call assemble() or assemble_separable() first")

    # -- Kronecker Gram solves ------------------------------------------- #
    # B (restricted or not) is exactly kron(G_0, G_1, ...) of small 1D Gram
    # matrices (the Dirichlet restriction slices each dimension, so the
    # Kronecker structure survives). Hence B^{-1} = kron(G_0^{-1}, ...) is
    # applied dimension-wise with tiny Cholesky solves in O(N sum N_d) --
    # no sparse factorization, no fill-in, which is what makes dim >= 4
    # tractable.
    def _kron_factors(self):
        if self._kron_chol is None:
            interior = self.boundary == "dirichlet"
            chols, shape = [], []
            for d in range(self.dim):
                G = np.asarray(self.basis.spline(d).gram())
                if interior:
                    G = G[1:-1, 1:-1]
                chols.append(cho_factor(G))
                shape.append(G.shape[0])
            self._kron_chol = (chols, tuple(shape))
        return self._kron_chol

    def _solve_B(self, c: np.ndarray) -> np.ndarray:
        """Solve the (restricted) Gram system B a = c.

        Accepts a vector ``(n,)`` or a matrix of columns ``(n, k)``.
        """
        chols, shape = self._kron_factors()
        vec = c.ndim == 1
        T = np.ascontiguousarray(c, dtype=float).reshape(shape + (-1,))
        for d, ch in enumerate(chols):
            T = np.moveaxis(T, d, 0)
            sh = T.shape
            T = cho_solve(ch, T.reshape(sh[0], -1)).reshape(sh)
            T = np.moveaxis(T, 0, d)
        out = T.reshape(int(np.prod(shape)), -1)
        return out[:, 0] if vec else out

    def _dense_operator(self) -> np.ndarray:
        if self._op_kron is not None:
            return self._op_kron.to_dense()  # already B^{-1} M
        _, _, Mres = self._restricted_matrices()
        return self._solve_B(Mres.toarray())

    def _krylov_propagator(self, m: int, tol: float):
        key = (m, tol)
        if self._krylov is None or self._krylov[0] != key:
            _, Bres, Mres = self._restricted_matrices()
            prop = _core.KrylovPropagator(Bres, Mres, m=m, tol=tol)
            self._krylov = (key, prop)
        return self._krylov[1]

    def _expm_action(self, matvec, v: np.ndarray, t: float, m: int, tol: float) -> np.ndarray:
        """expm(t A) v with matrix-free Arnoldi for A given as ``matvec``:
        the high-dimensional propagation path (a sparse LDLT of B suffers
        severe fill-in beyond 3D; Kronecker-form operators are never
        assembled at all). Same adaptive sub-stepping and augmented-matrix
        error estimate as the C++ KrylovPropagator."""
        n = v.size
        m = int(min(m, n - 1))
        w = v.astype(float).copy()
        if t == 0.0:
            return w
        t_done, tau, rejections = 0.0, t, 0
        V = np.empty((n, m + 1))
        H = np.zeros((m + 1, m))
        while t_done < t:
            beta = float(np.linalg.norm(w))
            if beta == 0.0:
                return w
            H[:] = 0.0
            V[:, 0] = w / beta
            mj, happy = m, False
            for j in range(m):
                p = matvec(V[:, j])
                for _ in range(2):  # MGS + one reorthogonalization
                    coeffs_ = V[:, : j + 1].T @ p
                    H[: j + 1, j] += coeffs_
                    p -= V[:, : j + 1] @ coeffs_
                hn = float(np.linalg.norm(p))
                H[j + 1, j] = hn
                if hn < 1e-14 * (1.0 + np.abs(H).max()):
                    mj, happy = j + 1, True
                    break
                V[:, j + 1] = p / hn
            while True:
                tau_try = min(tau, t - t_done)
                if happy:
                    E = np.asarray(_core.expm(tau_try * H[:mj, :mj]))
                    y, err = beta * E[:, 0], 0.0
                else:
                    Haug = np.zeros((mj + 1, mj + 1))
                    Haug[:mj, :mj] = H[:mj, :mj]
                    Haug[mj, mj - 1] = H[mj, mj - 1]
                    E = np.asarray(_core.expm(tau_try * Haug))
                    y, err = beta * E[:mj, 0], beta * abs(E[mj, 0])
                budget = tol * max(beta, 1.0) * (tau_try / t)
                if happy or err <= budget or tau_try <= 1e-14 * abs(t):
                    w = V[:, :mj] @ y
                    t_done = t if happy else t_done + tau_try
                    tau = tau_try * 2.0
                    break
                tau = tau_try * 0.5
                rejections += 1
                if rejections > 200:
                    raise RuntimeError("expm action: step size underflow")
        return w
