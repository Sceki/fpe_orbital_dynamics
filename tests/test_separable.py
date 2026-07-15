"""Kronecker-form (sum-of-separables) operators: equivalence with the
quadrature-assembled sparse path, and high-dimensional validation against
exact linear-SDE solutions."""

import numpy as np
import scipy.linalg

import fpe
from fpe.separable import KroneckerOperator, SeparableDynamics, SeparableTerm
from numpy.polynomial import Polynomial


def _exact_moments(A, GGt, m0, P0, t):
    n = A.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -A
    C[:n, n:] = GGt
    C[n:, n:] = A.T
    E = scipy.linalg.expm(C * t)
    Phi = E[n:, n:].T
    return Phi @ m0, Phi @ P0 @ Phi.T + Phi @ E[:n, n:]


def test_kronecker_operator_matches_explicit_kron():
    rng = np.random.default_rng(0)
    shape = (4, 3, 5)
    A0, A2 = rng.standard_normal((4, 4)), rng.standard_normal((5, 5))
    op = KroneckerOperator(shape, [(2.5, [(0, A0), (2, A2)]), (-1.0, [])])
    dense = 2.5 * np.kron(A0, np.kron(np.eye(3), A2)) - np.eye(60)
    v = rng.standard_normal(60)
    np.testing.assert_allclose(op.matvec(v), dense @ v, atol=1e-12)
    np.testing.assert_allclose(op.to_dense(), dense, atol=1e-12)


def test_separable_drift_evaluation():
    A = np.array([[0.0, 1.0], [-2.0, -0.7]])
    dyn = SeparableDynamics.linear(A)
    X = np.random.default_rng(1).standard_normal((40, 2))
    np.testing.assert_allclose(dyn(X), X @ A.T, atol=1e-14)
    # a genuinely nonlinear separable term: f_v += -x^3
    dyn2 = SeparableDynamics(2, dyn.terms + [SeparableTerm(1, -1.0, {0: Polynomial([0, 0, 0, 1])})])
    np.testing.assert_allclose(dyn2(X)[:, 1], X @ A.T[:, 1] - X[:, 0] ** 3, atol=1e-13)


def test_separable_operator_equals_sparse_assembly():
    """The Kronecker-form operator (stored as B^{-1} M) must equal the
    quadrature-assembled sparse operator for the same (polynomial)
    dynamics, both restricted."""
    A = np.array([[0.0, 1.0], [-1.3, -0.8]])
    D = np.array([[0.0, 0.0], [0.0, 0.07]])
    for boundary in ("dirichlet", "free"):
        basis = fpe.TensorBSplineBasis([(-2.5, 2.5), (-2.0, 3.0)], n_basis=[9, 8], order=3)
        # sparse reference (exact quadrature for the linear integrand)
        s_ref = fpe.FokkerPlanckSolver(
            basis,
            fpe.dynamics.CallableDynamics(
                f=lambda X: X @ A.T, div_f=lambda X: np.trace(A) * np.ones(X.shape[0]), dim=2
            ),
            diffusion=D,
            boundary=boundary,
        )
        s_ref.assemble(quadrature="gauss", q=4)
        A_ref = s_ref._dense_operator()  # B^{-1} M, restricted

        s_kron = fpe.FokkerPlanckSolver(basis, diffusion=D, boundary=boundary)
        s_kron.assemble_separable(SeparableDynamics.linear(A))
        np.testing.assert_allclose(s_kron._op_kron.to_dense(), A_ref, atol=1e-9)


def test_separable_sink_equals_add_sink():
    basis = fpe.TensorBSplineBasis([(-2.0, 2.0), (-2.0, 2.0)], n_basis=[8, 8], order=3)

    def s1d(x, d):
        return 0.3 * np.clip(np.abs(x) - 1.0, 0.0, None) ** 2

    ref = fpe.FokkerPlanckSolver(basis)
    ref.assemble(quadrature="gauss", q=6)
    ref.add_sink(lambda X: s1d(X[:, 0], 0) + s1d(X[:, 1], 1), q=6)
    A_ref = ref._dense_operator()

    kron = fpe.FokkerPlanckSolver(basis)
    kron.assemble_separable(SeparableDynamics(2, []), q=6, sink_1d=s1d)
    np.testing.assert_allclose(kron._op_kron.to_dense(), A_ref, atol=1e-9)


def test_separable_propagation_matches_sparse():
    A = np.array([[0.0, 1.0], [-1.0, -1.5]])
    D = np.array([[0.0, 0.0], [0.0, 0.05]])
    basis = fpe.TensorBSplineBasis([(-4.0, 4.0), (-4.0, 4.0)], n_basis=20, order=3)
    p0 = fpe.GaussianPDF([1.0, -0.5], np.diag([0.15, 0.15]))
    ts = np.array([0.4, 1.2])

    s_ref = fpe.FokkerPlanckSolver(
        basis,
        fpe.dynamics.CallableDynamics(
            f=lambda X: X @ A.T, div_f=lambda X: np.trace(A) * np.ones(X.shape[0]), dim=2
        ),
        diffusion=D,
    )
    s_ref.assemble(quadrature="gauss", q=4)
    ref = s_ref.propagate(s_ref.project(p0), ts, method="dense")

    s_kron = fpe.FokkerPlanckSolver(basis, diffusion=D)
    s_kron.assemble_separable(SeparableDynamics.linear(A))
    a0 = s_kron.project_separable(
        [fpe.GaussianPDF([1.0], [[0.15]]), fpe.GaussianPDF([-0.5], [[0.15]])]
    )
    # both projections use (different) high-order quadratures of the same
    # non-polynomial Gaussian: agreement to quadrature error, not exactness
    np.testing.assert_allclose(a0, s_ref.project(p0), atol=5e-7)
    for method in ("dense", "krylov"):
        got = s_kron.propagate(a0, ts, method=method)
        np.testing.assert_allclose(got, ref, atol=1e-6 * np.abs(ref).max())


def test_5d_linear_vs_van_loan():
    """End-to-end 5D: coupled damped linear SDE, Kronecker operator +
    matrix-free Krylov, moments against the exact Van Loan solution."""
    rng = np.random.default_rng(2)
    # random stable, coupled A: -I + skew + small mixing
    S = rng.standard_normal((5, 5)) * 0.3
    A = -np.eye(5) + (S - S.T)
    assert np.linalg.eigvals(A).real.max() < 0
    GGt = np.diag([0.3, 0.25, 0.2, 0.3, 0.25])
    P_inf = scipy.linalg.solve_continuous_lyapunov(A, -GGt)
    # start at a diagonal covariance matching the stationary variances
    # (product form -> project_separable), offset mean
    v0 = np.diag(P_inf).copy()
    sig0 = np.sqrt(v0)
    m0 = np.array([1.0, -1.0, 0.5, 0.0, -0.5]) * sig0

    domain = [(-6.0 * sig0[d], 6.0 * sig0[d]) for d in range(5)]
    basis = fpe.TensorBSplineBasis(domain, n_basis=13, order=3)
    solver = fpe.FokkerPlanckSolver(basis, diffusion=0.5 * GGt)

    def sink_1d(x, d):  # boundary sponge (see docs/theory.md Sec. 9)
        s = np.clip((np.abs(x) / sig0[d] - 4.8) / 1.2, 0.0, 1.0)
        return 3.0 * s * s * (3 - 2 * s)

    solver.assemble_separable(SeparableDynamics.linear(A), sink_1d=sink_1d)
    a0 = solver.project_separable(
        [fpe.GaussianPDF([m0[d]], [[v0[d]]]) for d in range(5)]
    )

    times = np.array([0.0, 0.4, 1.0])
    coeffs = solver.propagate(a0, times, method="krylov")
    P0 = np.diag(v0)
    for t, a in zip(times, coeffs):
        m_ref, P_ref = _exact_moments(A, GGt, m0, P0, t)
        sig_ref = np.sqrt(np.diag(P_ref))
        assert abs(solver.integral(a) - 1.0) < 2.5e-2
        mean, cov = solver.moments(solver.normalize(a))
        # tolerances measured at the 13-basis-per-dim resolution (h ~ 1.1 sigma)
        np.testing.assert_allclose(mean / sig_ref, m_ref / sig_ref, atol=0.03)
        np.testing.assert_allclose(np.sqrt(np.clip(np.diag(cov), 0, None)) / sig_ref,
                                   np.ones(5), atol=0.07)
