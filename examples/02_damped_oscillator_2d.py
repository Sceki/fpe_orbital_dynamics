"""Paper Sec. 5.1: stochastic damped harmonic oscillator (2D).

    dx = v dt
    dv = (-K x - gamma v) dt + sqrt(2 sigma) dW,   K=1, gamma=2.1, sigma=0.08

Because the SDE is linear, the exact pdf is Gaussian with moments given in
closed form; an Euler-Maruyama Monte Carlo run (the paper's ground truth,
Sec. 2) is included as well. The script reproduces the content of the
paper's Fig. 1: first/second moments (exact / Monte Carlo / Galerkin),
Hellinger distance, KL divergence, and the integral of the approximated pdf
over time, for two basis sizes (36 and 40 per dimension).

Run:  python examples/02_damped_oscillator_2d.py [--quick]
"""

import argparse
import pathlib

import numpy as np
import scipy.linalg

import fpe

OUT = pathlib.Path(__file__).parent / "output"

K, GAMMA, SIGMA = 1.0, 2.1, 0.08
A = np.array([[0.0, 1.0], [-K, -GAMMA]])
GGT = np.array([[0.0, 0.0], [0.0, 2.0 * SIGMA]])
M0 = np.array([-4.0, 0.001])
P0 = np.diag([0.09, 0.09])
DOMAIN = [(-6.0, 1.5), (-2.5, 2.5)]


def exact_moments(t: float):
    """Van Loan block method: exact linear-SDE mean/covariance."""
    n = 2
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -A
    C[:n, n:] = GGT
    C[n:, n:] = A.T
    E = scipy.linalg.expm(C * t)
    Phi = E[n:, n:].T
    return Phi @ M0, Phi @ P0 @ Phi.T + Phi @ E[:n, n:]


def monte_carlo_moments(times: np.ndarray, n_samples: int, n_steps: int, seed=0):
    """Euler-Maruyama sampling of the SDE; empirical mean/std at each epoch."""
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(M0, P0, size=n_samples)
    t_grid = np.linspace(0.0, times[-1], n_steps + 1)
    mean, std = np.empty((len(times), 2)), np.empty((len(times), 2))
    nxt = 0
    while nxt < len(times) and times[nxt] <= 1e-12:
        mean[nxt], std[nxt] = S.mean(axis=0), S.std(axis=0)
        nxt += 1
    for i in range(n_steps):
        dt = t_grid[i + 1] - t_grid[i]
        drift = np.column_stack([S[:, 1], -K * S[:, 0] - GAMMA * S[:, 1]])
        S = S + dt * drift
        S[:, 1] += np.sqrt(2.0 * SIGMA * dt) * rng.standard_normal(n_samples)
        while nxt < len(times) and times[nxt] <= t_grid[i + 1] + 1e-9:
            mean[nxt], std[nxt] = S.mean(axis=0), S.std(axis=0)
            nxt += 1
    return mean, std


def run_case(n_basis: int, times: np.ndarray):
    basis = fpe.TensorBSplineBasis(DOMAIN, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.DampedOscillator(k=K, gamma=GAMMA)
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, SIGMA]])
    solver.assemble(quadrature="gauss")
    a0 = solver.project(fpe.GaussianPDF(M0, P0))
    coeffs = solver.propagate(a0, times)

    res = {"mean": [], "std": [], "hell": [], "kl": [], "mass": []}
    for t, a in zip(times, coeffs):
        m_ref, P_ref = exact_moments(t)
        mean, cov = solver.moments(solver.normalize(a))
        res["mean"].append(mean)
        res["std"].append(np.sqrt(np.clip(np.diag(cov), 0.0, None)))
        res["mass"].append(solver.integral(a))
        # pdf-shape metrics on a +-5 sigma grid around the exact mean
        sx, sv = np.sqrt(np.diag(P_ref))
        xg = np.linspace(m_ref[0] - 5 * sx, m_ref[0] + 5 * sx, 70)
        vg = np.linspace(m_ref[1] - 5 * sv, m_ref[1] + 5 * sv, 70)
        XX, VV = np.meshgrid(xg, vg, indexing="ij")
        grid = np.column_stack([XX.ravel(), VV.ravel()])
        p_num = solver.evaluate(a, grid)
        p_ref = fpe.GaussianPDF(m_ref, P_ref)(grid)
        res["hell"].append(fpe.metrics.hellinger(p_num, p_ref))
        res["kl"].append(fpe.metrics.kl_divergence(p_ref, np.maximum(p_num, 1e-30)))
    return {k: np.array(v) for k, v in res.items()}, solver, coeffs


def main(quick: bool = False) -> None:
    times = np.linspace(0.0, 2.05, 12 if quick else 42)  # t0=0.95 -> tf=3 in the paper
    cases = {32: None, 36: None} if quick else {36: None, 40: None}
    for nb in cases:
        cases[nb], solver, coeffs = run_case(nb, times)
        r = cases[nb]
        print(
            f"n_basis={nb}: final Hellinger={r['hell'][-1]:.4f} "
            f"KL={r['kl'][-1]:.2e} |mass-1|={abs(r['mass'][-1] - 1):.2e}"
        )

    m_ex = np.array([exact_moments(t)[0] for t in times])
    s_ex = np.array([np.sqrt(np.diag(exact_moments(t)[1])) for t in times])
    print("running Euler-Maruyama Monte Carlo ...")
    n_mc = 10_000 if quick else 100_000
    mc_mean, mc_std = monte_carlo_moments(times, n_samples=n_mc, n_steps=500 if quick else 2050)
    print(f"MC vs exact: max mean dev {np.abs(mc_mean - m_ex).max():.2e} (sampling noise)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    labels = list(cases)
    mark = dict(marker="o", ms=3.5, ls="none", color="0.45", zorder=1)
    for j, (name, ex) in enumerate([("position x", 0), ("velocity v", 1)]):
        axes[0, j].plot(times, m_ex[:, ex], "k-", label="exact")
        axes[0, j].plot(times, mc_mean[:, ex], label=f"Monte Carlo (N = {n_mc:,})", **mark)
        axes[1, j].plot(times, s_ex[:, ex], "k-", label="exact")
        axes[1, j].plot(times, mc_std[:, ex], label=f"Monte Carlo (N = {n_mc:,})", **mark)
        for nb, color in zip(labels, ["C1", "C2"]):
            axes[0, j].plot(times, cases[nb]["mean"][:, ex], color + "--", label=f"{nb} basis")
            axes[1, j].plot(times, cases[nb]["std"][:, ex], color + "--", label=f"{nb} basis")
        axes[0, j].set(xlabel="t", title=f"mean, {name}")
        axes[1, j].set(xlabel="t", title=f"std, {name}")
        axes[0, j].legend()
    for nb, color in zip(labels, ["C1", "C2"]):
        axes[2, 0].plot(times, cases[nb]["hell"], color + "-", label=f"{nb} basis")
        axes[2, 1].semilogy(times, np.maximum(cases[nb]["kl"], 1e-12), color + "-", label=f"{nb} basis")
        axes[3, 0].plot(times, cases[nb]["mass"], color + "-", label=f"{nb} basis")
    axes[2, 0].set(xlabel="t", title="Hellinger distance vs exact pdf")
    axes[2, 1].set(xlabel="t", title="KL divergence vs exact pdf")
    axes[3, 0].axhline(1.0, color="k", lw=0.8)
    axes[3, 0].set(xlabel="t", title="integral of approximated pdf")
    for ax in (axes[2, 0], axes[2, 1], axes[3, 0]):
        ax.legend()

    # final pdf surface (largest basis case)
    xg = np.linspace(*DOMAIN[0], 90)
    vg = np.linspace(*DOMAIN[1], 90)
    XX, VV = np.meshgrid(xg, vg, indexing="ij")
    grid = np.column_stack([XX.ravel(), VV.ravel()])
    P_i = solver.evaluate(coeffs[0], grid).reshape(XX.shape)
    P_f = solver.evaluate(coeffs[-1], grid).reshape(XX.shape)
    levels = lambda P: np.linspace(0.08, 0.9, 6) * P.max()  # noqa: E731
    axes[3, 1].contour(XX, VV, P_i, levels=levels(P_i), colors="C3")
    axes[3, 1].contour(XX, VV, P_f, levels=levels(P_f), colors="C0")
    axes[3, 1].set(
        xlabel="x", ylabel="v", title="initial (red) and final (blue) pdf"
    )
    fig.tight_layout()
    fig.savefig(OUT / "damped_oscillator_2d.png", dpi=150)
    print(f"figure -> {OUT / 'damped_oscillator_2d.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
