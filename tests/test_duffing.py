"""Nonlinear (Duffing) dynamics: the exact stationary FPE solution must be
(numerically) invariant under propagation.

    dx = v dt;  dv = (-gamma v + alpha x - beta x^3) dt + sqrt(2 D) dW
    p_inf(x, v) ~ exp(-(gamma/D) (v^2/2 - alpha x^2/2 + beta x^4/4))

This exercises a genuinely non-Gaussian, nonlinear-drift case end to end
(assembly with a Python callable drift, projection, propagation, moments).
"""

import numpy as np

import fpe

GAMMA, ALPHA, BETA, D = 0.5, 1.0, 1.0, 0.05
DOMAIN = [(-2.2, 2.2), (-1.6, 1.6)]


def _stationary(X):
    x, v = X[:, 0], X[:, 1]
    U = -0.5 * ALPHA * x**2 + 0.25 * BETA * x**4
    return np.exp(-(GAMMA / D) * (0.5 * v**2 + U))


def _solver():
    basis = fpe.TensorBSplineBasis(DOMAIN, n_basis=[36, 30], order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: np.column_stack(
            [X[:, 1], -GAMMA * X[:, 1] + ALPHA * X[:, 0] - BETA * X[:, 0] ** 3]
        ),
        div_f=lambda X: -GAMMA * np.ones(X.shape[0]),
        dim=2,
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, D]])
    solver.assemble(quadrature="gauss")
    return solver


def test_stationary_solution_is_invariant():
    solver = _solver()
    a_inf = solver.normalize(solver.project(_stationary))
    coeffs = solver.propagate(a_inf, np.array([2.0, 5.0]))

    xg = np.linspace(-2.0, 2.0, 70)
    vg = np.linspace(-1.4, 1.4, 60)
    XX, VV = np.meshgrid(xg, vg, indexing="ij")
    grid = np.column_stack([XX.ravel(), VV.ravel()])
    p0 = solver.evaluate(a_inf, grid)
    m0, c0 = solver.moments(a_inf)
    for a in coeffs:
        assert abs(solver.integral(a) - 1.0) < 2e-3
        p = solver.evaluate(solver.normalize(a), grid)
        assert fpe.metrics.hellinger(p, p0) < 1e-2
        m, c = solver.moments(solver.normalize(a))
        np.testing.assert_allclose(m, m0, atol=2e-3)
        np.testing.assert_allclose(c, c0, atol=2e-3)
    # sanity: the stationary density really is bimodal in x
    marg = solver.marginal(a_inf, 0, xg)
    mid = marg[np.argmin(np.abs(xg))]
    peak = marg.max()
    assert mid < 0.35 * peak, "double-well stationary pdf must be bimodal"


def test_relaxation_towards_stationary():
    """A Gaussian at the barrier top must spread into the bimodal stationary
    density (monotically decreasing Hellinger distance to p_inf)."""
    solver = _solver()
    a0 = solver.project(fpe.GaussianPDF([0.0, 0.0], np.diag([0.15**2, 0.15**2])))
    coeffs = solver.propagate(a0, np.array([0.0, 2.0, 6.0, 14.0]))
    a_inf = solver.normalize(solver.project(_stationary))
    xg = np.linspace(-2.0, 2.0, 70)
    vg = np.linspace(-1.4, 1.4, 60)
    XX, VV = np.meshgrid(xg, vg, indexing="ij")
    grid = np.column_stack([XX.ravel(), VV.ravel()])
    p_inf = solver.evaluate(a_inf, grid)
    dists = [fpe.metrics.hellinger(solver.evaluate(a, grid), p_inf) for a in coeffs]
    assert dists[0] > 0.5, "initial condition is far from stationary"
    assert all(d2 < d1 for d1, d2 in zip(dists, dists[1:])), "must relax monotonically"
    assert dists[-1] < 0.05, "long-time solution must approach the stationary pdf"
