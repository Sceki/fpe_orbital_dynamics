"""Monte Carlo convergence validation of the Galerkin FPE solution.

If the propagated pdf is correct, it is the distribution that Monte Carlo
converges to: as the sample count N grows, the discrepancy between the
empirical samples and the FPE prediction must decay at the theoretical
Monte Carlo rate ~ 1/sqrt(N) (DKW inequality for the Kolmogorov-Smirnov
statistic; CLT for moments), until it hits the small floor set by the FPE
discretization itself. This script runs that experiment on four cases of
increasing "non-Gaussianity", with several independent MC replicates per
sample count:

  A. Ornstein-Uhlenbeck, Gaussian IC        (Gaussian; exact solution +
                                             exact sampling)
  B. Ornstein-Uhlenbeck, two-Gaussian-mixture IC
                                            (bimodal/NON-GAUSSIAN; exact
                                             solution + exact sampling)
  C. Duffing double-well oscillator         (bimodal/NON-GAUSSIAN,
                                             nonlinear; Euler-Maruyama)
  D. Debris decay, lognormal ballistic coeff.
                                            (skewed/NON-GAUSSIAN, parameter
                                             uncertainty; RK4 - no
                                             stochastic-integrator bias)

For cases A and B the exact solution is known in closed form, so the FPE
error floor (KS distance between the FPE and exact CDFs) is drawn as well:
the MC-vs-FPE curve must ride the 1/sqrt(N) guide down to that floor.

Run:  python examples/07_mc_convergence.py [--quick]
"""

import argparse
import importlib.util
import pathlib

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorized

import fpe

HERE = pathlib.Path(__file__).parent
OUT = HERE / "output"


def load_sibling(fname):
    spec = importlib.util.spec_from_file_location(fname.replace(".py", "").lstrip("0123456789_"), HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def marginal_cdf(solver, a, dim, lo, hi, n=4001):
    """CDF of the FPE marginal (clipped/normalized), as a callable."""
    xs = np.linspace(lo, hi, n)
    pdf = np.maximum(solver.marginal(solver.normalize(a), dim, xs), 0.0)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs))])
    c /= c[-1]
    return lambda q, xs=xs, c=c: np.interp(q, xs, c, left=0.0, right=1.0)


def sup_cdf_distance(cdf1, cdf2, lo, hi, n=20001):
    xs = np.linspace(lo, hi, n)
    return float(np.abs(cdf1(xs) - cdf2(xs)).max())


def ladder_stats(pool, ladder, fpe_cdf, fpe_mean, fpe_std):
    """KS / moment errors of MC prefixes of increasing size vs the FPE."""
    R = pool.shape[0]
    ks = np.empty((len(ladder), R))
    dm = np.empty((len(ladder), R))
    ds = np.empty((len(ladder), R))
    for i, N in enumerate(ladder):
        for r in range(R):
            s = pool[r, :N]
            ks[i, r] = fpe.metrics.ks_statistic(s, fpe_cdf)
            dm[i, r] = abs(s.mean() - fpe_mean)
            ds[i, r] = abs(s.std() - fpe_std)
    return ks, dm, ds


# ---------------------------------------------------------------------- #
# Case builders: each returns a dict with the FPE prediction, the MC
# sample pools of the observed coordinate, and optional exact references.
# ---------------------------------------------------------------------- #
THETA, S_OU = 1.0, 0.5


def _ou_solver(n_basis):
    basis = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: -THETA * X, div_f=lambda X: -THETA * np.ones(X.shape[0]), dim=1
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, sigma=[[S_OU]])
    solver.assemble(quadrature="gauss")
    return solver


def _ou_component(m0, v0, t):
    m = m0 * np.exp(-THETA * t)
    v = S_OU**2 / (2 * THETA) + (v0 - S_OU**2 / (2 * THETA)) * np.exp(-2 * THETA * t)
    return m, v


def case_ou_gaussian(R, n_max, n_basis, seed0=100):
    t_star, m0, v0 = 1.0, 1.5, 0.2**2
    solver = _ou_solver(n_basis)
    a = solver.propagate(solver.project(fpe.GaussianPDF([m0], [[v0]])), [t_star])[-1]
    m_ex, v_ex = _ou_component(m0, v0, t_star)
    pool = np.stack(
        [np.random.default_rng(seed0 + r).normal(m_ex, np.sqrt(v_ex), n_max) for r in range(R)]
    )
    fpe_cdf = marginal_cdf(solver, a, 0, -4.0, 4.0)
    exact_cdf = lambda q: ndtr((q - m_ex) / np.sqrt(v_ex))  # noqa: E731
    mean, cov = solver.moments(solver.normalize(a))
    return {
        "name": "A: OU, Gaussian",
        "pool": pool,
        "fpe_cdf": fpe_cdf,
        "fpe_mean": mean[0],
        "fpe_std": np.sqrt(cov[0, 0]),
        "ks_floor": sup_cdf_distance(fpe_cdf, exact_cdf, -4.0, 4.0),
        "mean_floor": abs(mean[0] - m_ex),
        "std_floor": abs(np.sqrt(cov[0, 0]) - np.sqrt(v_ex)),
        "note": "exact sampling",
    }


def case_ou_mixture(R, n_max, n_basis, seed0=200):
    """Non-Gaussian (bimodal) with a CLOSED-FORM solution: for a linear SDE,
    a Gaussian-mixture IC evolves into the mixture of the evolved
    components."""
    t_star = 0.6
    w = np.array([0.5, 0.5])
    m0s, v0s = np.array([-1.8, 1.2]), np.array([0.15**2, 0.3**2])
    solver = _ou_solver(n_basis)

    comps0 = [fpe.GaussianPDF([m], [[v]]) for m, v in zip(m0s, v0s)]
    p0 = lambda X: w[0] * comps0[0](X) + w[1] * comps0[1](X)  # noqa: E731
    a = solver.propagate(solver.project(p0), [t_star])[-1]

    ms, vs = zip(*[_ou_component(m, v, t_star) for m, v in zip(m0s, v0s)])
    ms, vs = np.array(ms), np.array(vs)
    pool = []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        comp = rng.random(n_max) < w[0]
        pool.append(np.where(comp, rng.normal(ms[0], np.sqrt(vs[0]), n_max),
                             rng.normal(ms[1], np.sqrt(vs[1]), n_max)))
    pool = np.stack(pool)
    fpe_cdf = marginal_cdf(solver, a, 0, -4.0, 4.0)
    exact_cdf = lambda q: w[0] * ndtr((q - ms[0]) / np.sqrt(vs[0])) + w[1] * ndtr(  # noqa: E731
        (q - ms[1]) / np.sqrt(vs[1]))
    m_ex = float(w @ ms)
    v_ex = float(w @ (vs + ms**2) - m_ex**2)
    mean, cov = solver.moments(solver.normalize(a))
    return {
        "name": "B: OU, mixture IC (bimodal)",
        "pool": pool,
        "fpe_cdf": fpe_cdf,
        "fpe_mean": mean[0],
        "fpe_std": np.sqrt(cov[0, 0]),
        "ks_floor": sup_cdf_distance(fpe_cdf, exact_cdf, -4.0, 4.0),
        "mean_floor": abs(mean[0] - m_ex),
        "std_floor": abs(np.sqrt(cov[0, 0]) - np.sqrt(v_ex)),
        "note": "exact sampling",
    }


def case_duffing(R, n_max, n_basis, dt, seed0=300):
    ex05 = load_sibling("05_duffing_oscillator.py")
    t_star = 6.0
    basis = fpe.TensorBSplineBasis(ex05.DOMAIN, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=ex05.drift, div_f=lambda X: -ex05.GAMMA * np.ones(X.shape[0]), dim=2
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, ex05.D]])
    solver.assemble(quadrature="gauss")
    a = solver.propagate(solver.project(fpe.GaussianPDF(ex05.M0, ex05.P0)), [t_star])[-1]

    pool = np.stack(
        [ex05.monte_carlo(np.array([t_star]), n_max, dt, seed=seed0 + r)[-1][:, 0] for r in range(R)]
    )
    mean, cov = solver.moments(solver.normalize(a))
    return {
        "name": "C: Duffing (bimodal)",
        "pool": pool,
        "fpe_cdf": marginal_cdf(solver, a, 0, *ex05.DOMAIN[0]),
        "fpe_mean": mean[0],
        "fpe_std": np.sqrt(cov[0, 0]),
        "note": f"Euler-Maruyama, dt={dt:g}",
    }


def case_debris(R, n_max, n_basis, mc_steps, seed0=400):
    ex06 = load_sibling("06_debris_decay_ballistic.py")
    t_star = ex06.T_FINAL
    worst = np.array([[ex06.A0_MEAN - 6.0 * ex06.A0_STD,
                       ex06.DELTA_MEDIAN * np.exp(5.5 * ex06.SIGMA_LOG)]])
    for _ in range(400):
        worst[0, 0] += (t_star / 400) * ex06.rate(worst[0, 0], worst[0, 1])
    domain = [
        (worst[0, 0] - 6.0, ex06.A0_MEAN + 6.5 * ex06.A0_STD),
        (ex06.DELTA_MEDIAN * np.exp(-5.5 * ex06.SIGMA_LOG),
         ex06.DELTA_MEDIAN * np.exp(5.5 * ex06.SIGMA_LOG)),
    ]
    basis = fpe.TensorBSplineBasis(domain, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(f=ex06.drift, div_f=ex06.div_drift, dim=2)
    solver = fpe.FokkerPlanckSolver(basis, dyn)
    solver.assemble(quadrature="gauss")
    a = solver.propagate(solver.project(ex06.InitialPDF()), [t_star])[-1]

    pool = np.stack(
        [ex06.monte_carlo(np.array([t_star]), n_max, mc_steps, seed=seed0 + r)[-1][:, 0]
         for r in range(R)]
    )
    mean, cov = solver.moments(solver.normalize(a))
    return {
        "name": "D: debris decay (skewed)",
        "pool": pool,
        "fpe_cdf": marginal_cdf(solver, a, 0, *domain[0]),
        "fpe_mean": mean[0],
        "fpe_std": np.sqrt(cov[0, 0]),
        "note": "RK4 (bias-free)",
    }


# ---------------------------------------------------------------------- #
def main(quick: bool = False) -> None:
    if quick:
        ladders = {
            "A": [1000, 4000, 16000], "B": [1000, 4000, 16000],
            "C": [1000, 4000, 16000], "D": [1000, 4000, 16000],
        }
        R, nb1, nb_duf, nb_deb = 2, 36, [30, 28], [40, 12]
        dt_duffing, deb_steps = 5e-3, 120
    else:
        ladders = {
            "A": [2000, 8000, 32000, 128000, 512000],
            "B": [2000, 8000, 32000, 128000, 512000],
            "C": [2000, 8000, 32000, 128000],
            "D": [2000, 8000, 32000, 128000, 512000],
        }
        R, nb1, nb_duf, nb_deb = 4, 48, [40, 36], [64, 16]
        dt_duffing, deb_steps = 1e-3, 400

    print("building cases (FPE solves + MC sample pools) ...")
    cases = [
        (case_ou_gaussian(R, max(ladders["A"]), nb1), ladders["A"]),
        (case_ou_mixture(R, max(ladders["B"]), nb1), ladders["B"]),
        (case_duffing(R, max(ladders["C"]), nb_duf, dt_duffing), ladders["C"]),
        (case_debris(R, max(ladders["D"]), nb_deb, deb_steps), ladders["D"]),
    ]

    results = []
    for case, ladder in cases:
        ks, dm, ds = ladder_stats(case["pool"], ladder, case["fpe_cdf"],
                                  case["fpe_mean"], case["fpe_std"])
        results.append((case, ladder, ks, dm, ds))
        print(f"\n{case['name']}   [{case['note']}]")
        header = f"  {'N':>8} {'KS(MC,FPE)':>12} {'KS*sqrt(N)':>11} {'|d mean|':>10} {'|d std|':>10}"
        print(header)
        for i, N in enumerate(ladder):
            print(f"  {N:>8d} {ks[i].mean():>12.5f} {ks[i].mean()*np.sqrt(N):>11.3f} "
                  f"{dm[i].mean():>10.2e} {ds[i].mean():>10.2e}")
        if "ks_floor" in case:
            print(f"  FPE-vs-exact floors: KS {case['ks_floor']:.2e}, "
                  f"mean {case['mean_floor']:.2e}, std {case['std_floor']:.2e}")
        # theoretical DKW constant for KS*sqrt(N) is ~0.5-1.36 (95%): check order
        rate = np.log(ks[0].mean() / ks[-1].mean()) / np.log(np.sqrt(ladder[-1] / ladder[0]))
        print(f"  observed KS decay ~ N^(-{rate/2:.2f})   (Monte Carlo rate: N^-0.50)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex="col")
    for col, (case, ladder, ks, dm, ds) in enumerate(results):
        N = np.array(ladder, dtype=float)
        ax = axes[0, col]
        for r in range(ks.shape[1]):
            ax.loglog(N, ks[:, r], "o", ms=3, color="C0", alpha=0.4)
        ax.loglog(N, ks.mean(axis=1), "C0-", label="KS(MC, FPE)")
        guide = ks[0].mean() * np.sqrt(N[0] / N)
        ax.loglog(N, guide, "k--", lw=1, label=r"$\propto 1/\sqrt{N}$")
        if "ks_floor" in case:
            ax.axhline(case["ks_floor"], color="C3", ls=":", lw=1.2,
                       label="FPE error floor (vs exact)")
        ax.set(title=f"{case['name']}\n[{case['note']}]")
        if col == 0:
            ax.set_ylabel("KS distance")
        ax.legend(fontsize=7)

        ax = axes[1, col]
        ax.loglog(N, dm.mean(axis=1), "C1-o", ms=3, label="|mean$_{MC}$ - mean$_{FPE}$|")
        ax.loglog(N, ds.mean(axis=1), "C2-s", ms=3, label="|std$_{MC}$ - std$_{FPE}$|")
        ax.loglog(N, dm[0].mean() * np.sqrt(N[0] / N), "k--", lw=1)
        if "mean_floor" in case:
            ax.axhline(max(case["mean_floor"], 1e-16), color="C1", ls=":", lw=1.2)
            ax.axhline(max(case["std_floor"], 1e-16), color="C2", ls=":", lw=1.2)
        ax.set_xlabel("Monte Carlo samples N")
        if col == 0:
            ax.set_ylabel("moment error")
        ax.legend(fontsize=7)
    fig.suptitle("Monte Carlo converges to the Galerkin-FPE solution at the theoretical "
                 r"$1/\sqrt{N}$ rate (dots: independent replicates)")
    fig.tight_layout()
    fig.savefig(OUT / "mc_convergence.png", dpi=150)
    print(f"\nfigure -> {OUT / 'mc_convergence.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
