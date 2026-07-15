"""Stochastic Duffing oscillator: a genuinely non-Gaussian FPE benchmark.

    dx = v dt
    dv = (-gamma v + alpha x - beta x^3) dt + sqrt(2 D) dW

The double-well potential U(x) = -alpha x^2/2 + beta x^4/4 makes the pdf
split from an initial Gaussian centred at the unstable equilibrium into a
*bimodal* distribution -- exactly the regime where moment-based uncertainty
propagation fails and solving the full Fokker-Planck equation pays off.
This is the classical nonlinear test case of the FPE literature the paper
builds on (e.g. Kumar & Narayanan 2006, cited in Sec. 1).

Ground truths:
- the exact stationary solution  p_inf ~ exp(-(gamma/D) (v^2/2 + U(x))),
- an Euler-Maruyama Monte Carlo simulation of the SDE.

Run:  python examples/05_duffing_oscillator.py [--quick]
"""

import argparse
import pathlib

import numpy as np
from scipy.integrate import trapezoid

import fpe

OUT = pathlib.Path(__file__).parent / "output"

GAMMA, ALPHA, BETA, D = 0.5, 1.0, 1.0, 0.05
M0 = np.array([0.0, 0.0])          # start on the potential barrier
P0 = np.diag([0.15**2, 0.15**2])
DOMAIN = [(-2.2, 2.2), (-1.8, 1.8)]
T_FINAL = 12.0
SNAPSHOTS = [0.0, 1.0, 3.0, 12.0]


def drift(X):
    return np.column_stack([X[:, 1], -GAMMA * X[:, 1] + ALPHA * X[:, 0] - BETA * X[:, 0] ** 3])


def stationary_pdf(X):
    """Exact stationary FPE solution (unnormalized; Kramers form)."""
    x, v = X[:, 0], X[:, 1]
    U = -0.5 * ALPHA * x**2 + 0.25 * BETA * x**4
    return np.exp(-(GAMMA / D) * (0.5 * v**2 + U))


def monte_carlo(times, n_samples, dt, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(M0, P0, size=n_samples)
    out = np.empty((len(times), n_samples, 2))
    t, nxt = 0.0, 0
    while nxt < len(times) and times[nxt] <= 1e-12:
        out[nxt] = S
        nxt += 1
    n_steps = int(round(times[-1] / dt))
    for _ in range(n_steps):
        S = S + dt * drift(S)
        S[:, 1] += np.sqrt(2.0 * D * dt) * rng.standard_normal(n_samples)
        t += dt
        while nxt < len(times) and times[nxt] <= t + 1e-9:
            out[nxt] = S
            nxt += 1
    return out


def main(quick: bool = False) -> None:
    n_basis = [30, 28] if quick else [44, 40]
    n_mc = 15_000 if quick else 60_000
    dt_mc = 5e-3 if quick else 2e-3
    times = np.linspace(0.0, T_FINAL, 25)

    basis = fpe.TensorBSplineBasis(DOMAIN, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=drift,
        div_f=lambda X: -GAMMA * np.ones(X.shape[0]),  # d(v)/dx + d(f_v)/dv
        dim=2,
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, D]])
    solver.assemble(quadrature="gauss")
    print(f"assembled N={solver.n} (interior {solver.active_indices.size})")

    a0 = solver.project(fpe.GaussianPDF(M0, P0))
    coeffs = solver.propagate(a0, times)

    # exact stationary density, normalized on the evaluation grid
    xg = np.linspace(*DOMAIN[0], 120)
    vg = np.linspace(*DOMAIN[1], 110)
    XX, VV = np.meshgrid(xg, vg, indexing="ij")
    grid = np.column_stack([XX.ravel(), VV.ravel()])
    cell = (xg[1] - xg[0]) * (vg[1] - vg[0])
    p_inf = stationary_pdf(grid)
    p_inf /= p_inf.sum() * cell

    hell_inf = [
        fpe.metrics.hellinger(solver.evaluate(a, grid), p_inf) for a in coeffs
    ]
    mass = [solver.integral(a) for a in coeffs]
    Ex2 = []
    for a in coeffs:
        m, cov = solver.moments(solver.normalize(a))
        Ex2.append(cov[0, 0] + m[0] ** 2)

    print("running Euler-Maruyama Monte Carlo ...")
    mc = monte_carlo(times, n_mc, dt_mc)
    mc_Ex2 = (mc[:, :, 0] ** 2).mean(axis=1)

    print(f"final E[x^2]: FPE {Ex2[-1]:.4f} | MC {mc_Ex2[-1]:.4f}")
    print(f"Hellinger to exact stationary at t={T_FINAL}: {hell_inf[-1]:.4f}")
    print(f"max |integral - 1|: {np.abs(np.array(mass) - 1).max():.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
    snap_idx = [int(np.argmin(np.abs(times - t))) for t in SNAPSHOTS]
    for col, si in enumerate(snap_idx):
        P = solver.evaluate(coeffs[si], grid).reshape(XX.shape)
        axes[0, col].contourf(XX, VV, np.maximum(P, 0.0), levels=24, cmap="magma")
        axes[0, col].set(title=f"Galerkin FPE, t = {times[si]:.0f}", xlabel="x", ylabel="v")
        axes[1, col].hist2d(
            mc[si][:, 0], mc[si][:, 1], bins=70, range=[DOMAIN[0], DOMAIN[1]], cmap="magma"
        )
        axes[1, col].set(title=f"Monte Carlo (N = {n_mc:,}), t = {times[si]:.0f}", xlabel="x", ylabel="v")

    axes[2, 0].plot(times, Ex2, "C1-", label="Galerkin FPE")
    axes[2, 0].plot(times, mc_Ex2, "o", ms=3.5, color="0.45", label=f"Monte Carlo (N = {n_mc:,})")
    axes[2, 0].set(xlabel="t", title=r"$E[x^2]$: spread across the wells")
    axes[2, 0].legend()

    axes[2, 1].plot(times, mass, "C0-")
    axes[2, 1].axhline(1.0, color="k", lw=0.8)
    axes[2, 1].set(xlabel="t", title="integral of approximated pdf")

    # marginal in x at final time vs exact stationary marginal + MC histogram
    xf = np.linspace(*DOMAIN[0], 300)
    marg_fpe = solver.marginal(solver.normalize(coeffs[-1]), 0, xf)
    Ux = -0.5 * ALPHA * xf**2 + 0.25 * BETA * xf**4
    marg_inf = np.exp(-(GAMMA / D) * Ux)
    marg_inf /= trapezoid(marg_inf, xf)
    axes[2, 2].hist(
        mc[-1][:, 0], bins=80, density=True, color="0.8", label=f"Monte Carlo (N = {n_mc:,})"
    )
    axes[2, 2].plot(xf, marg_inf, "k-", label="exact stationary")
    axes[2, 2].plot(xf, marg_fpe, "C1--", label="Galerkin FPE")
    axes[2, 2].set(xlabel="x", title=f"marginal p(x), t = {T_FINAL:.0f}")
    axes[2, 2].legend(fontsize=8)

    axes[2, 3].plot(times, hell_inf, "C2-")
    axes[2, 3].set(
        xlabel="t", title="Hellinger distance to exact\nstationary solution"
    )
    fig.tight_layout()
    fig.savefig(OUT / "duffing_oscillator.png", dpi=150)
    print(f"figure -> {OUT / 'duffing_oscillator.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
