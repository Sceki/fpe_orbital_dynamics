"""4D machinery: Kronecker Gram solves, Python Krylov path, 2D marginals,
and an end-to-end 4D validation against an exact linear-SDE solution."""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

import fpe



# Planar Clohessy-Wiltshire relative motion under PD station-keeping
# feedback (u = -kp r - kd v) with stochastic disturbance accelerations:
# state (x, y, vx, vy), a fully coupled, Hurwitz 4D linear system. (Without
# position feedback the along-track coordinate is translation-invariant --
# A singular, no stationary covariance.)
N_ORBIT = 1.0586e-3  # mean motion [1/s] (~700 km altitude)
KP = 4.0 * N_ORBIT**2
KD = 2.0 * N_ORBIT


def _cw_matrices(g=1.0):
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [3 * N_ORBIT**2 - KP, 0.0, -KD, 2 * N_ORBIT],
        [0.0, -KP, -2 * N_ORBIT, -KD],
    ])
    assert np.linalg.eigvals(A).real.max() < 0, "closed-loop dynamics must be stable"
    G = np.zeros((4, 2))
    G[2, 0] = g
    G[3, 1] = g
    return A, G


def _exact_moments(A, GGt, m0, P0, t):
    n = A.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -A
    C[:n, n:] = GGt
    C[n:, n:] = A.T
    E = scipy.linalg.expm(C * t)
    Phi = E[n:, n:].T
    return Phi @ m0, Phi @ P0 @ Phi.T + Phi @ E[:n, n:]


def test_kron_solve_matches_direct():
    for boundary in ("dirichlet", "free"):
        basis = fpe.TensorBSplineBasis([(-1, 1), (0, 2), (-3, 1)], n_basis=[6, 5, 7], order=3)
        solver = fpe.FokkerPlanckSolver(basis, boundary=boundary)
        solver._require_assembled(need_M=False)
        _, Bres, _ = solver._restricted_matrices()
        rng = np.random.default_rng(0)
        c = rng.standard_normal(Bres.shape[0])
        np.testing.assert_allclose(
            solver._solve_B(c), spla.spsolve(Bres.tocsc(), c), atol=1e-11
        )
        C = rng.standard_normal((Bres.shape[0], 3))
        ref = np.column_stack([spla.spsolve(Bres.tocsc(), C[:, j]) for j in range(3)])
        np.testing.assert_allclose(solver._solve_B(C), ref, atol=1e-11)


def test_python_krylov_matches_dense_4d():
    A, G = _cw_matrices(g=2e-4)
    basis = fpe.TensorBSplineBasis(
        [(-0.5, 0.5)] * 2 + [(-8e-4, 8e-4)] * 2, n_basis=6, order=3
    )
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: X @ A.T, div_f=lambda X: np.trace(A) * np.ones(X.shape[0]), dim=4
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, sigma=G)
    solver.assemble(quadrature="gauss", q=3)
    a0 = solver.project(
        fpe.GaussianPDF(np.zeros(4), np.diag([0.012**2, 0.012**2, 2.4e-8, 2.4e-8]))
    )
    ts = np.array([300.0, 900.0])
    dense = solver.propagate(a0, ts, method="dense")
    krylov = solver.propagate(a0, ts, method="krylov")  # dim >= 4 -> python path
    np.testing.assert_allclose(krylov, dense, atol=1e-8 * np.abs(dense).max())


def test_marginal_2d_matches_gaussian():
    basis = fpe.TensorBSplineBasis([(-4, 4), (-3, 5), (-4, 4)], n_basis=[18, 16, 15], order=3)
    solver = fpe.FokkerPlanckSolver(basis)
    cov = np.array([[0.5, 0.15, 0.0], [0.15, 0.4, -0.1], [0.0, -0.1, 0.6]])
    mean = np.array([0.2, 1.0, -0.3])
    a0 = solver.project(fpe.GaussianPDF(mean, cov))
    xg = np.linspace(-1.5, 1.9, 25)
    yg = np.linspace(-0.6, 2.6, 24)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    ours = solver.marginal(a0, (0, 1), pts)
    # 1) the contraction itself must be exact: compare against brute-force
    #    integration of the same spline pdf over the remaining dimension
    zg = np.linspace(-4.0, 4.0, 1201)
    for p_i, (px, py) in zip(ours[::75], pts[::75]):
        vals = solver.evaluate(a0, np.column_stack([
            np.full(zg.size, px), np.full(zg.size, py), zg
        ]))
        from scipy.integrate import trapezoid

        assert abs(p_i - trapezoid(vals, zg)) < 1e-8
    # 2) ... and reproduce the analytic Gaussian marginal to basis resolution
    ref = fpe.GaussianPDF(mean[:2], cov[:2, :2])(pts)
    np.testing.assert_allclose(ours, ref, atol=1.5e-2)
    # order of the requested dimensions must be honoured
    swapped = solver.marginal(a0, (1, 0), pts[:, ::-1])
    np.testing.assert_allclose(swapped, ours, atol=1e-12)


def test_4d_cw_moments_match_van_loan():
    """End-to-end 4D: coupled damped CW + noise, IC at the stationary
    covariance (so spreads stay constant and the box resolves them), offset
    mean decaying. Moments must track the exact Van Loan solution."""
    A, G = _cw_matrices(g=1.0)
    GGt = G @ G.T
    P_inf = scipy.linalg.solve_continuous_lyapunov(A, -GGt)
    scale = 0.1 / np.sqrt(P_inf[0, 0])  # position sigma -> 0.1 km
    G = G * scale
    GGt = G @ G.T
    P_inf = P_inf * scale**2
    sig = np.sqrt(np.diag(P_inf))

    m0 = np.array([1.0 * sig[0], -1.0 * sig[1], 0.5 * sig[2], 0.0])
    domain = [(-6.0 * sig[d], 6.0 * sig[d]) for d in range(4)]
    basis = fpe.TensorBSplineBasis(domain, n_basis=12, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: X @ A.T, div_f=lambda X: np.trace(A) * np.ones(X.shape[0]), dim=4
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, sigma=G)
    # q=3 integrates the linear-drift integrand exactly (degree 5)
    solver.assemble(quadrature="gauss", q=3)

    # boundary sponge: damps the weakly growing spurious outflow modes of
    # the truncated-domain transport operator (see docs/theory.md)
    def sponge(X):
        r = np.zeros(X.shape[0])
        for d in range(4):
            s = np.clip((np.abs(X[:, d]) / sig[d] - 4.6) / 1.4, 0.0, 1.0)
            r = np.maximum(r, s * s * (3.0 - 2.0 * s))
        return 2e-2 * r

    solver.add_sink(sponge)
    a0 = solver.project(fpe.GaussianPDF(m0, P_inf))

    times = np.array([0.0, 400.0, 800.0])
    coeffs = solver.propagate(a0, times, method="krylov")
    for t, a in zip(times, coeffs):
        m_ref, P_ref = _exact_moments(A, GGt, m0, P_inf, t)
        # tolerances at the 12-basis-per-dim resolution (h = 1.2 sigma):
        # this test checks the 4D pipeline end to end; accuracy versus
        # resolution and horizon is quantified in example 09.
        assert abs(solver.integral(a) - 1.0) < 2.5e-2
        mean, cov = solver.moments(solver.normalize(a))
        np.testing.assert_allclose(mean / sig, m_ref / sig, atol=0.07)
        np.testing.assert_allclose(np.sqrt(np.diag(cov)), np.sqrt(np.diag(P_ref)),
                                   rtol=0.08)
