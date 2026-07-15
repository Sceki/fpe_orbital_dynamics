"""Shared setup for the orbit-averaged equinoctial examples (paper Sec. 5.2-5.3).

Note on parameters: the paper quotes rho*Cd*A/m = 1e-6 m^-1 and 50 years of
propagation. Here `DELTA` is sized so that the ~5 km semi-major-axis decay
over 50 years stays comfortably inside a basis domain that also resolves the
initial 3.189 km dispersion -- the method requires the pdf to remain inside
the domain over the whole horizon. Change `DELTA`/`T_FINAL` freely; they are
plain physical parameters of the built-in C++ dynamics.
"""

import numpy as np

import fpe

MU = 398600.4418          # km^3 / s^2
DELTA = 6.0e-14           # rho Cd A/m [1/km] -> ~5 km decay over 50 years
YEAR = 365.25 * 86400.0
T_FINAL = 50.0 * YEAR

MEAN0 = np.array([6665.15, 0.0, 0.0])          # [km, -, -] (paper Sec. 5.2)
STD0 = np.array([3.189, 1e-4, 1e-4])


def make_domain(drift_a_km: float = -6.0, margin_sigma: float = 6.5, p_half_width: float = 8e-4):
    """Basis box: initial +-margin_sigma dispersion plus the drag drift in a."""
    lo = [MEAN0[0] + drift_a_km - margin_sigma * STD0[0], -p_half_width, -p_half_width]
    hi = [MEAN0[0] + margin_sigma * STD0[0], p_half_width, p_half_width]
    return list(zip(lo, hi))


def averaged_rhs(states: np.ndarray, n_quad_L: int = 64, delta: float = DELTA) -> np.ndarray:
    """Vectorized NumPy implementation of the averaged Gauss equations with
    in-plane drag (paper Eqs. 24-27) -- an implementation independent of the
    C++ core, used for the Monte Carlo ground truth."""
    a, P1, P2 = states[:, 0:1], states[:, 1:2], states[:, 2:3]
    L = (-np.pi + (np.arange(n_quad_L) + 0.5) * (2 * np.pi / n_quad_L))[None, :]
    sL, cL = np.sin(L), np.cos(L)
    B2 = 1.0 - P1**2 - P2**2
    B = np.sqrt(B2)
    p = a * B2
    h = np.sqrt(MU * p)
    Phi = 1.0 + P1 * sL + P2 * cL
    esf = P2 * sL - P1 * cL
    v2 = (MU / a) * (2.0 * Phi / B2 - 1.0)
    Dv = np.sqrt(1.0 + P1**2 + P2**2 + 2.0 * (P1 * sL + P2 * cL))
    c = -0.5 * delta * v2
    ar = c * esf / Dv
    at = c * Phi / Dv
    w = B**3 / Phi**2
    da = (2.0 * a**2 / h) * (esf * ar + Phi * at)
    dP1 = p / (h * Phi) * (-Phi * cL * ar + (P1 + (1.0 + Phi) * sL) * at)
    dP2 = p / (h * Phi) * (Phi * sL * ar + (P2 + (1.0 + Phi) * cL) * at)
    return np.column_stack(
        [np.mean(w * da, axis=1), np.mean(w * dP1, axis=1), np.mean(w * dP2, axis=1)]
    )


def monte_carlo(n_samples: int, times: np.ndarray, n_steps: int, seed=0,
                noise_p1: float = 0.0) -> np.ndarray:
    """Propagate samples of the initial Gaussian through the averaged ODE
    (vectorized RK4); optional additive white noise on P1 (Euler-Maruyama)
    for the stochastic case. Returns samples at each requested time,
    shape (len(times), n_samples, 3)."""
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(MEAN0, np.diag(STD0**2), size=n_samples)
    t_grid = np.linspace(0.0, times[-1], n_steps + 1)
    out = np.empty((len(times), n_samples, 3))
    next_out = 0
    if times[0] == 0.0:
        out[0] = S
        next_out = 1
    for i in range(n_steps):
        t0, t1 = t_grid[i], t_grid[i + 1]
        dt = t1 - t0
        k1 = averaged_rhs(S)
        k2 = averaged_rhs(S + 0.5 * dt * k1)
        k3 = averaged_rhs(S + 0.5 * dt * k2)
        k4 = averaged_rhs(S + dt * k3)
        S = S + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if noise_p1 > 0.0:
            S[:, 1] += noise_p1 * np.sqrt(dt) * rng.standard_normal(n_samples)
        while next_out < len(times) and times[next_out] <= t1 + 1e-9:
            out[next_out] = S
            next_out += 1
    return out


def build_solver(n_basis: int, diffusion=None, n_quad_L: int = 64, q: int = 4,
                 delta: float = DELTA, domain=None, n_threads: int = 0):
    basis = fpe.TensorBSplineBasis(domain or make_domain(), n_basis=n_basis, order=3)
    dyn = fpe.dynamics.EquinoctialAveragedDrag(mu=MU, delta=delta, n_quad_L=n_quad_L)
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=diffusion)
    solver.assemble(quadrature="gauss", q=q, n_threads=n_threads)
    return solver
