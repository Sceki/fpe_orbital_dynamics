"""End-to-end validation against closed-form Fokker-Planck solutions.

For linear SDEs dX = A X dt + G dW with Gaussian initial conditions the FPE
solution stays Gaussian with
    m(t)   = expm(A t) m0
    P(t)   = expm(A t) P0 expm(A t)^T + int_0^t expm(A s) G G^T expm(A s)^T ds,
the integral being computed exactly with Van Loan's block-matrix method.
This exercises the whole pipeline: assembly, projection, propagation,
evaluation, moments, and metrics.
"""

import numpy as np
import scipy.linalg

import fpe


def _exact_gaussian_moments(A, GGt, m0, P0, t):
    n = A.shape[0]
    if t == 0.0:
        return m0.copy(), P0.copy()
    # Van Loan (1978): expm([[ -A, GGt ], [ 0, A^T ]] t) = [[F1, G1], [0, F2]]
    # with expm(A t) = F2^T and int_0^t e^{As} GGt e^{A^T s} ds = F2^T G1.
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -A
    C[:n, n:] = GGt
    C[n:, n:] = A.T
    E = scipy.linalg.expm(C * t)
    F2 = E[n:, n:]
    G1 = E[:n, n:]
    Phi = F2.T
    Q = Phi @ G1
    return Phi @ m0, Phi @ P0 @ Phi.T + Q


class TestOrnsteinUhlenbeck1D:
    """dX = -theta X dt + s dW: the canonical analytically solvable FPE."""

    theta = 1.0
    s = 0.5
    m0, v0 = 1.5, 0.2**2

    def _solve(self, method):
        basis = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=40, order=3)
        dyn = fpe.dynamics.CallableDynamics(
            f=lambda X: -self.theta * X,
            div_f=lambda X: -self.theta * np.ones(X.shape[0]),
            dim=1,
        )
        solver = fpe.FokkerPlanckSolver(basis, dyn, sigma=[[self.s]])
        solver.assemble(quadrature="gauss")
        a0 = solver.project(fpe.GaussianPDF([self.m0], [[self.v0]]))
        times = np.linspace(0.0, 2.0, 9)
        coeffs = solver.propagate(a0, times, method=method)
        return solver, times, coeffs

    def _check(self, solver, times, coeffs):
        A = np.array([[-self.theta]])
        GGt = np.array([[self.s**2]])
        for t, a in zip(times, coeffs):
            m_ref, P_ref = _exact_gaussian_moments(
                A, GGt, np.array([self.m0]), np.array([[self.v0]]), t
            )
            assert abs(solver.integral(a) - 1.0) < 5e-3
            mean, cov = solver.moments(solver.normalize(a))
            assert abs(mean[0] - m_ref[0]) < 2e-3
            assert abs(cov[0, 0] - P_ref[0, 0]) < 2e-3
            # Full-shape check at the final time.
        x = np.linspace(-3.5, 3.5, 400)[:, None]
        p_num = solver.evaluate(coeffs[-1], x)
        p_ref = fpe.GaussianPDF(m_ref, P_ref)(x)
        assert fpe.metrics.hellinger(p_num, p_ref) < 0.01
        assert fpe.metrics.kl_divergence(p_ref, np.maximum(p_num, 1e-12)) < 1e-3

    def test_dense(self):
        self._check(*self._solve("dense"))

    def test_krylov(self):
        self._check(*self._solve("krylov"))


class TestDampedOscillator2D:
    """Paper Sec. 5.1 test case (K=1, gamma=2.1, sigma=0.08), validated
    against the exact linear-SDE solution instead of Monte Carlo."""

    K, gamma, sigma = 1.0, 2.1, 0.08

    def test_moments_and_shape(self):
        A = np.array([[0.0, 1.0], [-self.K, -self.gamma]])
        GGt = np.array([[0.0, 0.0], [0.0, 2.0 * self.sigma]])  # G = [0, sqrt(2 sigma)]^T
        m0 = np.array([-4.0, 0.001])
        P0 = np.diag([0.09, 0.09])

        # 40 basis per dimension: the paper's finer configuration (Sec. 5.1).
        basis = fpe.TensorBSplineBasis([(-6.0, 1.5), (-2.5, 2.5)], n_basis=[40, 40], order=3)
        dyn = fpe.dynamics.DampedOscillator(k=self.K, gamma=self.gamma)
        solver = fpe.FokkerPlanckSolver(
            basis, dyn, diffusion=[[0.0, 0.0], [0.0, self.sigma]]
        )
        solver.assemble(quadrature="gauss")
        a0 = solver.project(fpe.GaussianPDF(m0, P0))
        times = np.linspace(0.0, 2.05, 6)  # paper horizon: t0=0.95 to tf=3
        coeffs = solver.propagate(a0, times)

        for t, a in zip(times, coeffs):
            m_ref, P_ref = _exact_gaussian_moments(A, GGt, m0, P0, t)
            assert abs(solver.integral(a) - 1.0) < 2e-3
            mean, cov = solver.moments(solver.normalize(a))
            np.testing.assert_allclose(mean, m_ref, atol=3e-3)
            np.testing.assert_allclose(cov, P_ref, atol=8e-3)

        # Full pdf comparison at the final time, on the +-5 sigma region
        # where the pdf lives (far tails only compare spline ripple against
        # ~1e-90 Gaussian values and carry no information).
        sx, sv = np.sqrt(np.diag(P_ref))
        xg = np.linspace(m_ref[0] - 5 * sx, m_ref[0] + 5 * sx, 60)
        vg = np.linspace(m_ref[1] - 5 * sv, m_ref[1] + 5 * sv, 60)
        XX, VV = np.meshgrid(xg, vg, indexing="ij")
        grid = np.column_stack([XX.ravel(), VV.ravel()])
        p_num = solver.evaluate(coeffs[-1], grid)
        p_ref = fpe.GaussianPDF(m_ref, P_ref)(grid)
        # Consistent with the Hellinger levels the paper reports for this
        # configuration (Fig. 1, 40-basis case).
        assert fpe.metrics.hellinger(p_num, p_ref) < 0.05


def test_dirichlet_removes_unstable_boundary_modes():
    """On a truncated domain without boundary conditions the OU advection
    operator has genuine polynomial eigenfunctions x^m with eigenvalues
    theta*(m+1) > 0 (excluded on the real line by integrability). The
    Dirichlet restriction must remove them and recover the OU Fokker-Planck
    spectrum 0, -theta, -2 theta, ..."""
    theta, s = 1.0, 0.5
    basis = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=40, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: -theta * X, div_f=lambda X: -theta * np.ones(X.shape[0]), dim=1
    )

    free = fpe.FokkerPlanckSolver(basis, dyn, sigma=[[s]], boundary="free")
    free.assemble()
    ev_free = np.linalg.eigvals(free._dense_operator())
    assert ev_free.real.max() > 0.9 * theta, "free formulation has growing modes"

    diri = fpe.FokkerPlanckSolver(basis, dyn, sigma=[[s]], boundary="dirichlet")
    diri.assemble()
    ev = np.linalg.eigvals(diri._dense_operator())
    assert ev.real.max() < 1e-6, "Dirichlet restriction must be stable"
    lead = np.sort(ev.real)[-4:]
    np.testing.assert_allclose(lead, [-3.0 * theta, -2.0 * theta, -theta, 0.0], atol=2e-2)


def test_pure_diffusion_widens_gaussian():
    """f = 0, D = const: variance grows linearly, var(t) = var0 + 2 D t."""
    Dval = 0.05
    basis = fpe.TensorBSplineBasis([(-5.0, 5.0)], n_basis=36, order=4)
    solver = fpe.FokkerPlanckSolver(basis, dynamics=None, diffusion=[[Dval]])
    solver.assemble()
    a0 = solver.project(fpe.GaussianPDF([0.5], [[0.15]]))
    times = np.array([0.0, 0.5, 1.0, 2.0])
    coeffs = solver.propagate(a0, times)
    for t, a in zip(times, coeffs):
        mean, cov = solver.moments(solver.normalize(a))
        assert abs(mean[0] - 0.5) < 2e-3
        assert abs(cov[0, 0] - (0.15 + 2 * Dval * t)) < 2e-3
