"""Re-entry as a first-passage problem: absorbing sponge layer + pdf mass.

Example 06 keeps the whole population inside the domain and shows the
non-Gaussian shape of the decaying pdf. Here the drag is strong enough that
objects actually *leave* through a control altitude (300 km, below which the
residual lifetime is negligible), and the quantities of interest are those
of first-passage theory (Risken 1996; Gardiner 2009):

    survival probability   S(t)  = probability mass above the control altitude
    re-entry probability   P(t)  = 1 - S(t)
    re-entry-time density  f(t)  = -dS/dt

Numerically, a hard p = 0 cut at the crossing altitude would make the
non-dissipative Galerkin advection ring as the pdf exits. The robust,
standard alternative is a *sponge layer*: the domain extends one buffer
below the control altitude, where a smooth killing term -sigma(a) p
(:meth:`FokkerPlanckSolver.add_sink`) removes the pdf inside the domain
before it can interact with the boundary. Survival is measured as the mass
*above* the control altitude, so the sink introduces no bias with respect
to the Monte Carlo first-passage reference (RK4 trajectories, crossing
instants interpolated within a step).

Dynamics as in example 06 (exponential atmosphere, lognormal ballistic
parameter), with stronger drag so the median object re-enters in ~2.5
years: the re-entry-time distribution is itself skewed.

Run:  python examples/08_reentry_first_passage.py [--quick]
"""

import argparse
import pathlib

import numpy as np
from scipy.integrate import trapezoid

import fpe

OUT = pathlib.Path(__file__).parent / "output"

MU = 398600.4418                 # km^3/s^2
A_REF, A0_STD = 6778.0, 4.0      # initial mean (~400 km altitude), spread
A_CTRL = 6678.0                  # control altitude (~300 km): the event
H_SCALE = 60.0                   # atmospheric density scale height [km]
DELTA_MEDIAN = 1.2e-11           # delta = rho Cd A/m at a_ref [1/km]
SIGMA_LOG = 0.15                 # lognormal spread of delta
YEAR = 365.25 * 86400.0
T_FINAL = 4.5 * YEAR             # median crossing ~2.5 y -> full S-curve

# Sponge layer below the control altitude: smooth killing ramp, fully
# absorbing well before the domain edge (decay factor ~ e^-20 over the
# buffer at the local advection speed).
L_BUFFER = 60.0                  # [km] domain extension below A_CTRL
SINK_START = A_CTRL - 15.0       # sink inactive above this
SINK_RAMP = 25.0                 # [km] smoothstep ramp width
SINK_MAX = 2.5e-6                # [1/s] maximum killing rate


def rate(a, dl):
    return -dl * np.exp(np.minimum((A_REF - a) / H_SCALE, 4.0)) * np.sqrt(MU * np.maximum(a, 1.0))


def drift(X):
    return np.column_stack([rate(X[:, 0], X[:, 1]), np.zeros(X.shape[0])])


def div_drift(X):
    return rate(X[:, 0], X[:, 1]) * (0.5 / X[:, 0] - 1.0 / H_SCALE)


def sink(X):
    s = np.clip((SINK_START - X[:, 0]) / SINK_RAMP, 0.0, 1.0)
    return SINK_MAX * s * s * (3.0 - 2.0 * s)  # smoothstep


class InitialPDF:
    def __call__(self, X):
        a, dl = X[:, 0], X[:, 1]
        pa = np.exp(-0.5 * ((a - A_REF) / A0_STD) ** 2) / (A0_STD * np.sqrt(2 * np.pi))
        z = (np.log(dl) - np.log(DELTA_MEDIAN)) / SIGMA_LOG
        pd = np.exp(-0.5 * z**2) / (dl * SIGMA_LOG * np.sqrt(2 * np.pi))
        return pa * pd


def monte_carlo_first_passage(n_samples, times, n_steps, seed=0):
    """RK4 trajectories; returns first-passage times through A_CTRL
    (np.inf if still above at t_final) and surviving-sample snapshots."""
    rng = np.random.default_rng(seed)
    S = np.column_stack([
        rng.normal(A_REF, A0_STD, n_samples),
        DELTA_MEDIAN * np.exp(SIGMA_LOG * rng.standard_normal(n_samples)),
    ])
    t_cross = np.full(n_samples, np.inf)
    t_grid = np.linspace(0.0, times[-1], n_steps + 1)
    snapshots = []
    nxt = 0
    while nxt < len(times) and times[nxt] <= 1e-12:
        snapshots.append(S[:, 0].copy())
        nxt += 1
    for i in range(n_steps):
        dt = t_grid[i + 1] - t_grid[i]
        active = np.isinf(t_cross)
        a_prev = S[active, 0].copy()
        Sa = S[active]
        k1 = drift(Sa)
        k2 = drift(Sa + 0.5 * dt * k1)
        k3 = drift(Sa + 0.5 * dt * k2)
        k4 = drift(Sa + dt * k3)
        Sa = Sa + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        S[active] = Sa
        crossed = Sa[:, 0] <= A_CTRL
        if np.any(crossed):
            frac = (a_prev[crossed] - A_CTRL) / np.maximum(a_prev[crossed] - Sa[crossed, 0], 1e-30)
            idx = np.flatnonzero(active)[crossed]
            t_cross[idx] = t_grid[i] + np.clip(frac, 0.0, 1.0) * dt
        while nxt < len(times) and times[nxt] <= t_grid[i + 1] + 1e-9:
            snapshots.append(np.where(np.isinf(t_cross), S[:, 0], np.nan).copy())
            nxt += 1
    return t_cross, snapshots


def main(quick: bool = False) -> None:
    n_basis = [64, 10] if quick else [120, 18]
    n_mc = 40_000 if quick else 200_000
    mc_steps = 300 if quick else 900
    times = np.linspace(0.0, T_FINAL, 10 if quick else 37)

    domain = [
        (A_CTRL - L_BUFFER, A_REF + 6.5 * A0_STD),
        (DELTA_MEDIAN * np.exp(-5.5 * SIGMA_LOG), DELTA_MEDIAN * np.exp(5.5 * SIGMA_LOG)),
    ]
    basis = fpe.TensorBSplineBasis(domain, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(f=drift, div_f=div_drift, dim=2)
    solver = fpe.FokkerPlanckSolver(basis, dyn)
    solver.assemble(quadrature="gauss")
    solver.add_sink(sink)
    print(f"assembled N={solver.n} (interior {solver.active_indices.size}), sponge sink added")

    a0 = solver.project(InitialPDF())
    coeffs = solver.propagate(a0, times)

    # survival = mass ABOVE the control altitude (grid endpoint exactly on it)
    ag_surv = np.linspace(A_CTRL, domain[0][1], 900)
    S_fpe = np.array([
        trapezoid(np.maximum(solver.marginal(c, 0, ag_surv), 0.0), ag_surv) for c in coeffs
    ])

    print("running Monte Carlo first-passage (vectorized RK4) ...")
    t_cross, snapshots = monte_carlo_first_passage(n_mc, times, mc_steps)
    S_mc = np.array([np.mean(t_cross > t) for t in times])

    f_fpe = -np.gradient(S_fpe, times)          # re-entry-time density [1/s]
    med_fpe = np.interp(0.5, 1.0 - S_fpe, times)
    med_mc = np.interp(0.5, 1.0 - S_mc, times)

    print(f"max |S_fpe - S_mc| over the horizon : {np.abs(S_fpe - S_mc).max():.4f}")
    print(f"re-entered by t = {T_FINAL/YEAR:.1f} y : FPE {1-S_fpe[-1]:.4f} | MC {1-S_mc[-1]:.4f}")
    print(f"median re-entry time                : FPE {med_fpe/YEAR:.3f} y | MC {med_mc/YEAR:.3f} y")
    print(f"total mass at t_final (incl. sponge): {solver.integral(coeffs[-1]):.4f} "
          "(absorbed mass has left the pdf)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)
    yrs = times / YEAR

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    # 1) surviving pdf draining towards/through the control altitude
    ag = np.linspace(domain[0][0], domain[0][1], 600)
    idx = np.linspace(0, len(times) - 1, 5).astype(int)
    n_bins, bin_w = 90, (domain[0][1] - domain[0][0]) / 90
    for i, c in zip(idx, ["C3", "C1", "C4", "C2", "C0"]):
        surv = snapshots[i]
        surv = surv[np.isfinite(surv)]
        if surv.size > 100:
            axes[0, 0].hist(surv, bins=n_bins, range=domain[0],
                            weights=np.full(surv.size, 1.0 / (n_mc * bin_w)),
                            alpha=0.25, color=c)
        marg = solver.marginal(coeffs[i], 0, ag)  # unnormalized: sub-density
        axes[0, 0].plot(ag, marg, c, label=f"t = {yrs[i]:.1f} y  (S = {S_fpe[i]:.2f})")
    axes[0, 0].axvline(A_CTRL, color="k", ls=":", lw=1)
    axes[0, 0].axvspan(domain[0][0], SINK_START, color="0.85", zorder=0)
    axes[0, 0].set(xlabel="a [km]", ylabel="sub-probability density",
                   title=f"pdf drains through the control altitude (dotted); shaded:\n"
                         f"MC survivors (N = {n_mc:,}); "
                         "grey: sponge layer where the sink absorbs it")
    axes[0, 0].legend(fontsize=8)

    # 2) survival probability
    axes[0, 1].plot(yrs, S_fpe, "C1-", label="FPE: mass above control altitude")
    axes[0, 1].plot(yrs, S_mc, "o", ms=4, color="0.45", label="MC: fraction not yet crossed")
    axes[0, 1].set(xlabel="t [years]", ylabel="survival probability",
                   title="survival probability from the propagated pdf")
    axes[0, 1].legend()

    # 3) re-entry-time density
    fin = t_cross[np.isfinite(t_cross)]
    axes[1, 0].hist(fin / YEAR, bins=60, range=(0, T_FINAL / YEAR),
                    weights=np.full(fin.size, 1.0 / n_mc / (T_FINAL / YEAR / 60)),
                    color="0.8", label=f"MC first-passage times (N = {n_mc:,})")
    axes[1, 0].plot(yrs, f_fpe * YEAR, "C1-", lw=2, label="FPE: -dS/dt")
    axes[1, 0].set(xlabel="re-entry time [years]", ylabel="density [1/year]",
                   title="re-entry-time distribution (skewed: lognormal drag)")
    axes[1, 0].legend()

    # 4) re-entry probability with FPE percentiles
    axes[1, 1].plot(yrs, 1.0 - S_fpe, "C1-", label="FPE")
    axes[1, 1].plot(yrs, 1.0 - S_mc, "o", ms=4, color="0.45", label=f"Monte Carlo (N = {n_mc:,})")
    for qv, ls in [(0.05, ":"), (0.5, "--"), (0.95, ":")]:
        tq = np.interp(qv, 1.0 - S_fpe, times) / YEAR
        axes[1, 1].axvline(tq, color="C2", ls=ls, lw=1)
        axes[1, 1].annotate(f"{int(qv*100)}%", (tq + 0.03, 0.03), color="C2", fontsize=8)
    axes[1, 1].set(xlabel="t [years]", ylabel="re-entry probability",
                   title="P(re-entered by t) with FPE percentiles")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "reentry_first_passage.png", dpi=150)
    print(f"figure -> {OUT / 'reentry_first_passage.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
