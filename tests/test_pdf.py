"""GaussianPDF: correctness incl. the extreme scale disparities of orbital states."""

import numpy as np
import scipy.stats

import fpe


def test_matches_scipy_correlated():
    mean = [0.3, -1.0, 2.0]
    cov = np.array([[0.5, 0.1, 0.0], [0.1, 0.4, -0.05], [0.0, -0.05, 0.8]])
    g = fpe.GaussianPDF(mean, cov)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    ref = scipy.stats.multivariate_normal(mean, cov).pdf(X)
    np.testing.assert_allclose(g(X), ref, rtol=1e-12)


def test_extreme_scale_disparity():
    """Semi-major axis in km next to a 1e-12-scale drag parameter: the
    covariance condition number (~1e25) breaks scipy's PSD check, but the
    correlation-based evaluation must stay exact."""
    mean = np.array([6778.0, 1.5e-11])
    std = np.array([4.0, 2.25e-12])
    g = fpe.GaussianPDF(mean, np.diag(std**2))
    rng = np.random.default_rng(1)
    X = mean + std * rng.normal(size=(100, 2))
    ref = (
        scipy.stats.norm(mean[0], std[0]).pdf(X[:, 0])
        * scipy.stats.norm(mean[1], std[1]).pdf(X[:, 1])
    )
    np.testing.assert_allclose(g(X), ref, rtol=1e-12)


def test_sampling_moments():
    mean = [1.0, -2.0]
    cov = np.array([[0.3, 0.12], [0.12, 0.5]])
    g = fpe.GaussianPDF(mean, cov)
    S = g.sample(200_000, seed=2)
    np.testing.assert_allclose(S.mean(axis=0), mean, atol=5e-3)
    np.testing.assert_allclose(np.cov(S.T), cov, atol=5e-3)
