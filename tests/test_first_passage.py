"""Absorption (sink) term and first-passage/re-entry validation.

With zero dynamics the killing term is exactly solvable,
    p(x, t) = p0(x) exp(-sigma(x) t),
which pins down :meth:`FokkerPlanckSolver.add_sink`. The re-entry test then
validates the sponge-layer first-passage formulation of example 08 against
Monte Carlo crossing times.
"""

import numpy as np
from scipy.integrate import trapezoid

import fpe


def test_sink_constant_rate_exact_decay():
    basis = fpe.TensorBSplineBasis([(-2.0, 2.0)], n_basis=20, order=3)
    solver = fpe.FokkerPlanckSolver(basis)  # no drift, no diffusion
    solver.assemble()
    sigma = 0.3
    solver.add_sink(lambda X: sigma * np.ones(X.shape[0]))
    a0 = solver.project(fpe.GaussianPDF([0.0], [[0.2]]))
    mass0 = solver.integral(a0)  # projection truncates the ~2e-5 boundary tails
    coeffs = solver.propagate(a0, np.array([1.0, 2.5]))
    for t, a in zip([1.0, 2.5], coeffs):
        # constant sigma: M_sink = -sigma B, so the decay is exact in the
        # Galerkin space: mass(t) = mass(0) exp(-sigma t) to solver precision
        assert abs(solver.integral(a) - mass0 * np.exp(-sigma * t)) < 1e-10
        x = np.linspace(-1.5, 1.5, 50)[:, None]
        np.testing.assert_allclose(
            solver.evaluate(a, x), np.exp(-sigma * t) * solver.evaluate(a0, x), atol=1e-8
        )


def test_sink_spatially_varying_exact_decay():
    """f = 0, D = 0: p(x, t) = p0(x) exp(-sigma(x) t) pointwise."""
    basis = fpe.TensorBSplineBasis([(-2.0, 2.0)], n_basis=44, order=3)
    solver = fpe.FokkerPlanckSolver(basis)
    solver.assemble()
    sig = lambda x: 0.4 * (1.0 + np.tanh(1.5 * x))  # noqa: E731
    solver.add_sink(lambda X: sig(X[:, 0]))
    p0 = fpe.GaussianPDF([0.2], [[0.15]])
    a0 = solver.project(p0)
    t = 1.2
    a_t = solver.propagate(a0, np.array([t]))[-1]
    x = np.linspace(-1.4, 1.6, 120)
    ref = p0(x[:, None]) * np.exp(-sig(x) * t)
    np.testing.assert_allclose(solver.evaluate(a_t, x[:, None]), ref, atol=2e-3)


def test_reentry_survival_matches_monte_carlo():
    """Mini version of example 08: sponge-layer survival vs MC crossings."""
    mu, a_ref, a_std = 398600.4418, 6778.0, 4.0
    a_ctrl, h_scale = 6678.0, 60.0
    d_med, s_log = 1.2e-11, 0.15
    year = 365.25 * 86400.0
    t_final = 4.5 * year
    sink_start, sink_ramp, sink_max = a_ctrl - 15.0, 25.0, 2.5e-6

    def rate(a, dl):
        return -dl * np.exp(np.minimum((a_ref - a) / h_scale, 4.0)) * np.sqrt(mu * np.maximum(a, 1.0))

    def drift(X):
        return np.column_stack([rate(X[:, 0], X[:, 1]), np.zeros(X.shape[0])])

    def div_drift(X):
        return rate(X[:, 0], X[:, 1]) * (0.5 / X[:, 0] - 1.0 / h_scale)

    def sink(X):
        s = np.clip((sink_start - X[:, 0]) / sink_ramp, 0.0, 1.0)
        return sink_max * s * s * (3.0 - 2.0 * s)

    domain = [(a_ctrl - 60.0, a_ref + 6.5 * a_std),
              (d_med * np.exp(-5.5 * s_log), d_med * np.exp(5.5 * s_log))]
    basis = fpe.TensorBSplineBasis(domain, n_basis=[64, 10], order=3)
    dyn = fpe.dynamics.CallableDynamics(f=drift, div_f=div_drift, dim=2)
    solver = fpe.FokkerPlanckSolver(basis, dyn).assemble()
    solver.add_sink(sink)

    def p0(X):
        a, dl = X[:, 0], X[:, 1]
        pa = np.exp(-0.5 * ((a - a_ref) / a_std) ** 2) / (a_std * np.sqrt(2 * np.pi))
        z = (np.log(dl) - np.log(d_med)) / s_log
        return pa * np.exp(-0.5 * z**2) / (dl * s_log * np.sqrt(2 * np.pi))

    times = np.linspace(0.0, t_final, 10)
    coeffs = solver.propagate(solver.project(p0), times)
    ag = np.linspace(a_ctrl, domain[0][1], 600)
    S_fpe = np.array([
        trapezoid(np.maximum(solver.marginal(c, 0, ag), 0.0), ag) for c in coeffs
    ])

    # Monte Carlo first passage (RK4, interpolated crossings)
    rng = np.random.default_rng(3)
    n = 30_000
    S = np.column_stack([rng.normal(a_ref, a_std, n),
                         d_med * np.exp(s_log * rng.standard_normal(n))])
    t_cross = np.full(n, np.inf)
    n_steps = 250
    tg = np.linspace(0.0, t_final, n_steps + 1)
    for i in range(n_steps):
        dt = tg[i + 1] - tg[i]
        act = np.isinf(t_cross)
        a_prev = S[act, 0].copy()
        Sa = S[act]
        k1, k2 = drift(Sa), drift(Sa + 0.5 * dt * drift(Sa))
        k3 = drift(Sa + 0.5 * dt * k2)
        k4 = drift(Sa + dt * k3)
        Sa = Sa + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        S[act] = Sa
        crossed = Sa[:, 0] <= a_ctrl
        if np.any(crossed):
            frac = (a_prev[crossed] - a_ctrl) / np.maximum(a_prev[crossed] - Sa[crossed, 0], 1e-30)
            t_cross[np.flatnonzero(act)[crossed]] = tg[i] + np.clip(frac, 0, 1) * dt
    S_mc = np.array([np.mean(t_cross > t) for t in times])

    # survival: physical bounds, monotone decay (up to marginal-quadrature
    # ripple of a few 1e-4), and MC agreement
    assert np.all(S_fpe <= 1.0 + 1e-3) and np.all(S_fpe >= -1e-3)
    assert np.all(np.diff(S_fpe) <= 5e-4)
    assert np.abs(S_fpe - S_mc).max() < 0.04
    assert S_fpe[-1] < 0.01, "essentially everything re-enters within the horizon"
    med_fpe = np.interp(0.5, 1 - S_fpe, times)
    med_mc = np.interp(0.5, 1 - S_mc, times)
    assert abs(med_fpe - med_mc) < 0.1 * year
