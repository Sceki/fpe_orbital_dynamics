"""Space-debris orbital decay: exponential atmosphere, uncertain ballistic
coefficient (2D, non-Gaussian).

For LEO debris the dominant error source in decay prediction is the drag
(ballistic) parameter delta = rho Cd A/m: attitude, shape, and atmospheric
density are all poorly known, and the uncertainty is conventionally modelled
as *lognormal*. The FPE framework handles this parameter uncertainty by
augmenting the state (paper Sec. 1: "uncertain initial conditions and/or
parameters"). With an exponentially increasing air density,

    da/dt     = -delta exp(-(a - a_ref)/H) sqrt(mu a),
    ddelta/dt = 0,

decay *accelerates* as the object descends, so fast decayers race ahead:
even from a Gaussian semi-major axis the marginal develops a heavy left
tail. Mean/covariance methods cannot represent this; the propagated pdf
yields it directly -- e.g. as the probability of having decayed below a
threshold altitude vs. time (the quantity that matters for re-entry and
collision-risk screening).

Also showcased here: a *non-Gaussian initial pdf* (Gaussian x lognormal),
projected like any other callable density -- the propagation cost is
unchanged, since the assembled operator is independent of the initial
condition (the paper's key selling point).

Ground truth: Monte Carlo with vectorized RK4 on the same ODE.

Run:  python examples/06_debris_decay_ballistic.py [--quick]
"""

import argparse
import pathlib

import numpy as np
from scipy.integrate import trapezoid

import fpe

OUT = pathlib.Path(__file__).parent / "output"

MU = 398600.4418                 # km^3/s^2
A0_MEAN, A0_STD = 6778.0, 4.0    # ~400 km altitude
H_SCALE = 60.0                   # atmospheric density scale height [km]
DELTA_MEDIAN = 5.0e-12           # delta = rho Cd A/m at a_ref [1/km]
SIGMA_LOG = 0.12                 # lognormal spread of the ballistic parameter
YEAR = 365.25 * 86400.0
T_FINAL = 2.0 * YEAR
A_THRESHOLD = A0_MEAN - 20.0     # "decayed below ~380 km" corridor


def rate(a, dl):
    # exp argument capped for numerical safety far below the domain
    return -dl * np.exp(np.minimum((A0_MEAN - a) / H_SCALE, 4.0)) * np.sqrt(MU * np.maximum(a, 1.0))


def drift(X):
    a, dl = X[:, 0], X[:, 1]
    return np.column_stack([rate(a, dl), np.zeros_like(a)])


def div_drift(X):
    a, dl = X[:, 0], X[:, 1]
    # d/da of rate = rate * (1/(2a) - 1/H)
    return rate(a, dl) * (0.5 / a - 1.0 / H_SCALE)


class InitialPDF:
    """Gaussian in a, lognormal in delta (independent)."""

    def __call__(self, X):
        a, dl = X[:, 0], X[:, 1]
        pa = np.exp(-0.5 * ((a - A0_MEAN) / A0_STD) ** 2) / (A0_STD * np.sqrt(2 * np.pi))
        z = (np.log(dl) - np.log(DELTA_MEDIAN)) / SIGMA_LOG
        pd = np.exp(-0.5 * z**2) / (dl * SIGMA_LOG * np.sqrt(2 * np.pi))
        return pa * pd


def sample_initial(n, seed=0):
    rng = np.random.default_rng(seed)
    a0 = rng.normal(A0_MEAN, A0_STD, n)
    dl = DELTA_MEDIAN * np.exp(SIGMA_LOG * rng.standard_normal(n))
    return np.column_stack([a0, dl])


def monte_carlo(times, n_samples, n_steps, seed=0):
    S = sample_initial(n_samples, seed)
    t_grid = np.linspace(0.0, times[-1], n_steps + 1)
    out = np.empty((len(times), n_samples, 2))
    nxt = 0
    while nxt < len(times) and times[nxt] <= 1e-12:
        out[nxt] = S
        nxt += 1
    for i in range(n_steps):
        dt = t_grid[i + 1] - t_grid[i]
        k1 = drift(S)
        k2 = drift(S + 0.5 * dt * k1)
        k3 = drift(S + 0.5 * dt * k2)
        k4 = drift(S + dt * k3)
        S = S + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        while nxt < len(times) and times[nxt] <= t_grid[i + 1] + 1e-9:
            out[nxt] = S
            nxt += 1
    return out


def main(quick: bool = False) -> None:
    n_basis = [40, 12] if quick else [72, 18]
    n_mc = 50_000 if quick else 300_000
    mc_steps = 150 if quick else 500
    times = np.linspace(0.0, T_FINAL, 5 if quick else 21)

    # Automatic domain sizing: integrate the worst-case corner (low a0,
    # high delta) so the box provably contains the pdf over the horizon.
    worst = np.array([[A0_MEAN - 6.0 * A0_STD, DELTA_MEDIAN * np.exp(5.5 * SIGMA_LOG)]])
    for _ in range(400):
        worst[0, 0] += (T_FINAL / 400) * rate(worst[0, 0], worst[0, 1])
    a_lo = worst[0, 0] - 6.0
    domain = [
        (a_lo, A0_MEAN + 6.5 * A0_STD),
        (DELTA_MEDIAN * np.exp(-5.5 * SIGMA_LOG), DELTA_MEDIAN * np.exp(5.5 * SIGMA_LOG)),
    ]
    print(f"domain: a in [{domain[0][0]:.1f}, {domain[0][1]:.1f}] km, "
          f"delta in [{domain[1][0]:.2e}, {domain[1][1]:.2e}] 1/km")

    basis = fpe.TensorBSplineBasis(domain, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(f=drift, div_f=div_drift, dim=2)
    solver = fpe.FokkerPlanckSolver(basis, dyn)  # no diffusion: pure advection
    solver.assemble(quadrature="gauss")
    print(f"assembled N={solver.n} (interior {solver.active_indices.size})")

    a0 = solver.project(InitialPDF())
    coeffs = solver.propagate(a0, times)

    print("running Monte Carlo (vectorized RK4) ...")
    mc = monte_carlo(times, n_mc, mc_steps)

    ag = np.linspace(domain[0][0], domain[0][1], 600)
    ag_cdf = np.linspace(domain[0][0], A_THRESHOLD, 500)  # endpoint ON the threshold
    p_below, mc_below, means, stds, skews, mass = [], [], [], [], [], []
    for a_c, samp in zip(coeffs, mc):
        an = solver.normalize(a_c)
        marg = np.maximum(solver.marginal(an, 0, ag), 0.0)
        norm = trapezoid(marg, ag)
        marg /= norm
        below = np.maximum(solver.marginal(an, 0, ag_cdf), 0.0) / norm
        p_below.append(trapezoid(below, ag_cdf))
        mc_below.append(np.mean(samp[:, 0] <= A_THRESHOLD))
        m1 = trapezoid(ag * marg, ag)
        m2 = trapezoid((ag - m1) ** 2 * marg, ag)
        m3 = trapezoid((ag - m1) ** 3 * marg, ag)
        means.append(m1)
        stds.append(np.sqrt(m2))
        skews.append(m3 / m2**1.5)
        mass.append(solver.integral(a_c))

    mc_mean = mc[:, :, 0].mean(axis=1)
    mc_std = mc[:, :, 0].std(axis=1)
    mc_cent = mc[:, :, 0] - mc_mean[:, None]
    mc_skew = (mc_cent**3).mean(axis=1) / mc_std**3

    print(f"final mean a  : FPE {means[-1]:.3f} km | MC {mc_mean[-1]:.3f} km "
          f"(err {abs(means[-1]-mc_mean[-1])*1e3:.0f} m)")
    print(f"final std a   : FPE {stds[-1]:.3f} km | MC {mc_std[-1]:.3f} km")
    print(f"final skewness: FPE {skews[-1]:+.3f} | MC {mc_skew[-1]:+.3f}  (Gaussian: 0)")
    print(f"final P(a < {A_THRESHOLD:.0f} km): FPE {p_below[-1]:.4f} | MC {mc_below[-1]:.4f}")
    print(f"max |integral - 1|: {np.abs(np.array(mass)-1).max():.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)
    yrs = times / YEAR

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    # 1) marginal pdf of a at several epochs, vs MC histograms
    idx = np.linspace(0, len(times) - 1, 4).astype(int)
    for i, c in zip(idx, ["C3", "C4", "C2", "C0"]):
        an = solver.normalize(coeffs[i])
        axes[0, 0].hist(mc[i][:, 0], bins=140, density=True, alpha=0.25, color=c)
        axes[0, 0].plot(ag, solver.marginal(an, 0, ag), c, label=f"t = {yrs[i]:.1f} y")
    axes[0, 0].axvline(A_THRESHOLD, color="k", ls=":", lw=1)
    axes[0, 0].set(xlabel="a [km]", ylabel="pdf",
                   title="semi-major-axis marginal: heavy left tail develops\n"
                         f"(lines: Galerkin FPE, shaded: Monte Carlo, N = {n_mc:,})")
    axes[0, 0].legend()

    # 2) joint pdf at final time (windowed to +-4.5 sigma of the final pdf)
    m_f, c_f = solver.moments(solver.normalize(coeffs[-1]))
    s_f = np.sqrt(np.diag(c_f))
    aw = np.linspace(max(m_f[0] - 4.5 * s_f[0], domain[0][0]),
                     min(m_f[0] + 4.5 * s_f[0], domain[0][1]), 150)
    dg = np.linspace(max(m_f[1] - 4.5 * s_f[1], domain[1][0]),
                     min(m_f[1] + 4.5 * s_f[1], domain[1][1]), 150)
    AA, DD = np.meshgrid(aw, dg, indexing="ij")
    grid = np.column_stack([AA.ravel(), DD.ravel()])
    P = solver.evaluate(solver.normalize(coeffs[-1]), grid).reshape(AA.shape)
    axes[0, 1].contourf(AA, DD * 1e12, np.maximum(P, 0), levels=20, cmap="magma")
    axes[0, 1].set(xlabel="a [km]", ylabel=r"$\delta$ [$10^{-12}$ km$^{-1}$]",
                   title=f"joint pdf at t = {yrs[-1]:.0f} y:\n"
                         "a-delta correlation from differential decay")

    # 3) probability of having decayed below the threshold
    axes[1, 0].plot(yrs, p_below, "C1-", label="Galerkin FPE")
    axes[1, 0].plot(yrs, mc_below, "o", ms=4, color="0.45", label=f"Monte Carlo (N = {n_mc:,})")
    axes[1, 0].set(xlabel="t [years]", ylabel="probability",
                   title=f"P(a < {A_THRESHOLD:.0f} km): decay-corridor probability")
    axes[1, 0].legend()

    # 4) decay envelope and non-Gaussianity
    means, stds, skews = map(np.array, (means, stds, skews))
    axes[1, 1].fill_between(yrs, means - 3 * stds, means + 3 * stds, color="C1", alpha=0.2,
                            label=r"FPE mean $\pm 3\sigma$")
    axes[1, 1].plot(yrs, means, "C1-")
    axes[1, 1].plot(yrs, mc_mean, "o", ms=4, color="0.45", label=f"MC mean (N = {n_mc:,})")
    ax2 = axes[1, 1].twinx()
    ax2.plot(yrs, skews, "C2--", label="FPE skewness")
    ax2.plot(yrs, mc_skew, "s", ms=3.5, color="C2", alpha=0.6, label="MC skewness")
    ax2.set_ylabel("skewness of a", color="C2")
    axes[1, 1].set(xlabel="t [years]", ylabel="a [km]",
                   title="decay envelope and non-Gaussianity")
    h1, l1 = axes[1, 1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "debris_decay_ballistic.png", dpi=150)
    print(f"figure -> {OUT / 'debris_decay_ballistic.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
