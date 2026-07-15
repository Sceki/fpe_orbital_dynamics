"""Statistical validation against Monte Carlo, including non-Gaussian cases.

Two kinds of check (deterministic through fixed seeds):

1. Closed-form non-Gaussian solutions: for a linear SDE a Gaussian-mixture
   initial pdf evolves into the mixture of the evolved components -- an
   exactly known bimodal transient the solver must reproduce (pdf shape and
   the first three moments).

2. MC convergence: samples of the true process must converge to the FPE
   prediction at the Monte Carlo rate. The Kolmogorov-Smirnov distance
   between N samples and the FPE marginal CDF must shrink ~ 1/sqrt(N) and,
   at large N, fall below (DKW bound + FPE discretization floor).
"""

import numpy as np
from scipy.special import ndtr

import fpe

THETA, S = 1.0, 0.5


def _ou_solver(n_basis=48):
    basis = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: -THETA * X, div_f=lambda X: -THETA * np.ones(X.shape[0]), dim=1
    )
    return fpe.FokkerPlanckSolver(basis, dyn, sigma=[[S]]).assemble()


def _evolve(m0, v0, t):
    m = m0 * np.exp(-THETA * t)
    v = S**2 / (2 * THETA) + (v0 - S**2 / (2 * THETA)) * np.exp(-2 * THETA * t)
    return m, v


def _fpe_cdf(solver, a, lo=-4.0, hi=4.0, n=4001):
    xs = np.linspace(lo, hi, n)
    pdf = np.maximum(solver.marginal(solver.normalize(a), 0, xs), 0.0)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs))])
    c /= c[-1]
    return lambda q: np.interp(q, xs, c, left=0.0, right=1.0)


def test_ks_statistic_basics():
    # Single sample at 0.5 against U(0,1): D = 0.5 exactly.
    assert abs(fpe.metrics.ks_statistic([0.5], lambda x: x) - 0.5) < 1e-15
    # Ideal uniform grid: D = 1/(2n).
    n = 100
    grid = (np.arange(n) + 0.5) / n
    assert abs(fpe.metrics.ks_statistic(grid, lambda x: x) - 0.5 / n) < 1e-12
    # Gaussian samples vs the true CDF: DKW 99.9% bound.
    rng = np.random.default_rng(0)
    s = rng.standard_normal(50_000)
    d = fpe.metrics.ks_statistic(s, ndtr)
    assert d < np.sqrt(np.log(2 / 0.001) / (2 * 50_000))


class TestGaussianMixtureIC:
    """Exactly solvable NON-GAUSSIAN (bimodal) transient."""

    w = np.array([0.5, 0.5])
    m0s = np.array([-1.8, 1.2])
    v0s = np.array([0.15**2, 0.3**2])

    def _exact_pdf(self, x, t):
        p = np.zeros_like(x)
        for w, m0, v0 in zip(self.w, self.m0s, self.v0s):
            m, v = _evolve(m0, v0, t)
            p += w * np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
        return p

    def _exact_moments(self, t):
        ms, vs = zip(*[_evolve(m0, v0, t) for m0, v0 in zip(self.m0s, self.v0s)])
        ms, vs = np.array(ms), np.array(vs)
        mean = float(self.w @ ms)
        var = float(self.w @ (vs + ms**2) - mean**2)
        # third central moment of a Gaussian mixture
        mu3 = float(self.w @ ((ms - mean) ** 3 + 3 * (ms - mean) * vs))
        return mean, var, mu3 / var**1.5

    def test_matches_exact_mixture(self):
        # 64 basis: the sigma=0.15 component needs ~2 knot spans per sigma.
        solver = _ou_solver(64)
        comps = [fpe.GaussianPDF([m], [[v]]) for m, v in zip(self.m0s, self.v0s)]
        a0 = solver.project(lambda X: 0.5 * comps[0](X) + 0.5 * comps[1](X))
        times = np.array([0.0, 0.3, 0.6, 1.2])
        coeffs = solver.propagate(a0, times)
        xs = np.linspace(-3.5, 3.5, 600)
        for t, a in zip(times, coeffs):
            assert abs(solver.integral(a) - 1.0) < 2e-3
            p_num = solver.evaluate(solver.normalize(a), xs[:, None])
            p_ref = self._exact_pdf(xs, t)
            assert fpe.metrics.hellinger(p_num, p_ref) < 8e-3
            # moments incl. the third: the solution is genuinely skewed/bimodal
            m_ref, v_ref, skew_ref = self._exact_moments(t)
            mean, cov = solver.moments(solver.normalize(a))
            assert abs(mean[0] - m_ref) < 2e-3
            assert abs(cov[0, 0] - v_ref) < 2e-3
            # skewness from the marginal (uniform grid; self-normalizing sums)
            marg = np.maximum(solver.marginal(solver.normalize(a), 0, xs), 0.0)
            m1 = float(np.sum(xs * marg) / np.sum(marg))
            mu2 = float(np.sum((xs - m1) ** 2 * marg) / np.sum(marg))
            mu3 = float(np.sum((xs - m1) ** 3 * marg) / np.sum(marg))
            assert abs(mu3 / mu2**1.5 - skew_ref) < 0.02
        # sanity: the transient really is bimodal at t=0.6
        p_mid = solver.evaluate(solver.normalize(coeffs[2]), np.array([[0.0]]))[0]
        m1, _ = _evolve(self.m0s[0], self.v0s[0], 0.6)
        p_peak = solver.evaluate(solver.normalize(coeffs[2]), np.array([[m1]]))[0]
        assert p_mid < 0.5 * p_peak

    def test_mc_ks_converges_to_fpe(self):
        """Exact mixture sampling: KS(MC, FPE) must drop at ~1/sqrt(N)."""
        t_star = 0.6
        solver = _ou_solver()
        comps = [fpe.GaussianPDF([m], [[v]]) for m, v in zip(self.m0s, self.v0s)]
        a0 = solver.project(lambda X: 0.5 * comps[0](X) + 0.5 * comps[1](X))
        a = solver.propagate(a0, [t_star])[-1]
        cdf = _fpe_cdf(solver, a)

        rng = np.random.default_rng(1)
        n_max = 200_000
        ms, vs = zip(*[_evolve(m0, v0, t_star) for m0, v0 in zip(self.m0s, self.v0s)])
        comp = rng.random(n_max) < self.w[0]
        pool = np.where(comp, rng.normal(ms[0], np.sqrt(vs[0]), n_max),
                        rng.normal(ms[1], np.sqrt(vs[1]), n_max))

        d_small = fpe.metrics.ks_statistic(pool[:2000], cdf)
        d_large = fpe.metrics.ks_statistic(pool, cdf)
        assert d_large < d_small / 4.0, "KS must keep dropping with more samples"
        # DKW 99.9% bound at n_max plus a generous FPE-floor allowance
        dkw = np.sqrt(np.log(2 / 0.001) / (2 * n_max))
        assert d_large < dkw + 2e-3


def test_mc_ks_converges_skewed_advection():
    """Non-Gaussian by nonlinearity: accelerating decay with a lognormal
    parameter (mini version of example 06); MC via bias-free RK4."""
    mu, a_ref, h_scale = 398600.4418, 6778.0, 60.0
    a_std, d_med, s_log = 4.0, 5.0e-12, 0.12
    t_star = 1.5 * 365.25 * 86400.0

    def rate(a, dl):
        return -dl * np.exp(np.minimum((a_ref - a) / h_scale, 4.0)) * np.sqrt(mu * np.maximum(a, 1.0))

    def drift(X):
        return np.column_stack([rate(X[:, 0], X[:, 1]), np.zeros(X.shape[0])])

    def div_drift(X):
        return rate(X[:, 0], X[:, 1]) * (0.5 / X[:, 0] - 1.0 / h_scale)

    worst_a = a_ref - 6 * a_std
    worst_d = d_med * np.exp(5.5 * s_log)
    for _ in range(300):
        worst_a += (t_star / 300) * rate(worst_a, worst_d)
    domain = [(worst_a - 6.0, a_ref + 6.5 * a_std),
              (d_med * np.exp(-5.5 * s_log), d_med * np.exp(5.5 * s_log))]
    basis = fpe.TensorBSplineBasis(domain, n_basis=[48, 12], order=3)
    dyn = fpe.dynamics.CallableDynamics(f=drift, div_f=div_drift, dim=2)
    solver = fpe.FokkerPlanckSolver(basis, dyn).assemble()

    def p0(X):
        a, dl = X[:, 0], X[:, 1]
        pa = np.exp(-0.5 * ((a - a_ref) / a_std) ** 2) / (a_std * np.sqrt(2 * np.pi))
        z = (np.log(dl) - np.log(d_med)) / s_log
        return pa * np.exp(-0.5 * z**2) / (dl * s_log * np.sqrt(2 * np.pi))

    a_c = solver.propagate(solver.project(p0), [t_star])[-1]
    cdf = _fpe_cdf(solver, a_c, domain[0][0], domain[0][1])

    rng = np.random.default_rng(2)
    n_max = 150_000
    S0 = np.column_stack([rng.normal(a_ref, a_std, n_max),
                          d_med * np.exp(s_log * rng.standard_normal(n_max))])
    n_steps = 150
    dt = t_star / n_steps
    S = S0
    for _ in range(n_steps):
        k1 = drift(S)
        k2 = drift(S + 0.5 * dt * k1)
        k3 = drift(S + 0.5 * dt * k2)
        k4 = drift(S + dt * k3)
        S = S + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    samples = S[:, 0]

    # the propagated marginal is genuinely skewed
    mc_skew = float(((samples - samples.mean()) ** 3).mean() / samples.std() ** 3)
    assert mc_skew < -0.05

    d_small = fpe.metrics.ks_statistic(samples[:2000], cdf)
    d_large = fpe.metrics.ks_statistic(samples, cdf)
    assert d_large < d_small / 3.0
    dkw = np.sqrt(np.log(2 / 0.001) / (2 * n_max))
    assert d_large < dkw + 4e-3
