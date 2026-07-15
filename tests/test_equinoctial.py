"""Averaged equinoctial drag dynamics (paper Sec. 5.2, Eqs. 24-27)."""

import numpy as np
import pytest

import fpe

MU = 398600.4418  # km^3/s^2
DELTA = 1e-6      # rho Cd A/m [1/km]


def _reference_rhs(state, n_quad_L=64):
    """Independent NumPy implementation of the orbit-averaged Gauss
    variational equations with in-plane drag, using the same midpoint rule
    in L as the C++ code."""
    a, P1, P2 = state
    B2 = 1.0 - P1**2 - P2**2
    B = np.sqrt(B2)
    p = a * B2
    h = np.sqrt(MU * p)
    L = -np.pi + (np.arange(n_quad_L) + 0.5) * (2 * np.pi / n_quad_L)
    sL, cL = np.sin(L), np.cos(L)
    Phi = 1.0 + P1 * sL + P2 * cL
    esf = P2 * sL - P1 * cL
    v2 = (MU / a) * (2.0 * Phi / B2 - 1.0)
    Dv = np.sqrt(1.0 + P1**2 + P2**2 + 2.0 * (P1 * sL + P2 * cL))
    c = -0.5 * DELTA * v2
    ar = c * esf / Dv
    at = c * Phi / Dv
    w = B**3 / Phi**2
    da = (2.0 * a**2 / h) * (esf * ar + Phi * at)
    dP1 = p / (h * Phi) * (-Phi * cL * ar + (P1 + (1.0 + Phi) * sL) * at)
    dP2 = p / (h * Phi) * (Phi * sL * ar + (P2 + (1.0 + Phi) * cL) * at)
    return np.array([np.mean(w * da), np.mean(w * dP1), np.mean(w * dP2)])


@pytest.fixture
def dyn():
    return fpe.dynamics.EquinoctialAveragedDrag(mu=MU, delta=DELTA, n_quad_L=64)


def test_drift_matches_reference(dyn):
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = np.array(
            [6600.0 + 200.0 * rng.random(), 2e-3 * rng.standard_normal(), 2e-3 * rng.standard_normal()]
        )
        f = np.asarray(dyn.eval_f(state))
        # Same rule, different summation order (C++ sequential vs NumPy
        # pairwise): agreement to a few ulps of the largest term.
        np.testing.assert_allclose(f, _reference_rhs(state), rtol=1e-10, atol=1e-18)


def test_drag_decays_semimajor_axis(dyn):
    f = np.asarray(dyn.eval_f(np.array([6665.15, 1e-4, -2e-4])))
    assert f[0] < 0.0, "drag must shrink the orbit"


def test_divergence_matches_finite_differences(dyn):
    state = np.array([6665.15, 3e-4, -1.5e-4])
    _, div_ad = dyn.eval(state)
    div_fd = 0.0
    for i in range(3):
        h = 1e-6 * max(1.0, abs(state[i]))
        sp = state.copy()
        sm = state.copy()
        sp[i] += h
        sm[i] -= h
        div_fd += (dyn.eval_f(sp)[i] - dyn.eval_f(sm)[i]) / (2 * h)
    assert abs(div_ad - div_fd) < 1e-6 * max(abs(div_ad), 1e-12) + 1e-14


def test_batch_matches_single(dyn):
    rng = np.random.default_rng(1)
    X = np.column_stack(
        [
            6650.0 + 30.0 * rng.random(50),
            1e-3 * rng.standard_normal(50),
            1e-3 * rng.standard_normal(50),
        ]
    )
    F, divF = dyn.eval_batch(X)
    for i in range(50):
        f_i, div_i = dyn.eval(X[i])
        np.testing.assert_allclose(F[i], np.asarray(f_i), rtol=1e-14)
        assert abs(divF[i] - div_i) < 1e-18 + 1e-12 * abs(div_i)


def test_quadrature_in_L_converged():
    """The midpoint rule over the periodic L integrand converges spectrally:
    64 vs 256 nodes must agree to near machine precision."""
    d64 = fpe.dynamics.EquinoctialAveragedDrag(mu=MU, delta=DELTA, n_quad_L=64)
    d256 = fpe.dynamics.EquinoctialAveragedDrag(mu=MU, delta=DELTA, n_quad_L=256)
    state = np.array([6665.15, 5e-4, -3e-4])
    np.testing.assert_allclose(
        np.asarray(d64.eval_f(state)), np.asarray(d256.eval_f(state)), rtol=1e-12
    )


def test_small_3d_propagation_conserves_mass():
    """Smoke test of the full 3D pipeline (reduced basis for speed).

    delta is sized so the ~2 km semi-major-axis decay over 45 days stays
    well inside the basis box (the method requires the pdf to remain in the
    domain over the whole horizon).
    """
    mean = np.array([6665.15, 0.0, 0.0])
    std = np.array([3.189, 1e-4, 1e-4])
    lo = mean - 6 * std
    hi = mean + 6 * std
    basis = fpe.TensorBSplineBasis(list(zip(lo, hi)), n_basis=14, order=3)
    dyn = fpe.dynamics.EquinoctialAveragedDrag(mu=MU, delta=1e-11, n_quad_L=32)
    solver = fpe.FokkerPlanckSolver(basis, dyn)
    solver.assemble(quadrature="gauss", q=4)
    a0 = solver.project(fpe.GaussianPDF(mean, np.diag(std**2)))
    day = 86400.0
    coeffs = solver.propagate(a0, np.array([0.0, 15 * day, 45 * day]), method="krylov")
    for a in coeffs:
        assert abs(solver.integral(a) - 1.0) < 2e-3
    m0, _ = solver.moments(solver.normalize(coeffs[0]))
    m1, _ = solver.moments(solver.normalize(coeffs[-1]))
    assert m1[0] - m0[0] < -1.5, "mean semi-major axis must decay ~2 km under drag"
