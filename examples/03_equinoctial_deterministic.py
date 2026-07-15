"""Paper Sec. 5.2: LEO decay under drag, uncertain initial conditions (3D).

Orbit-averaged equinoctial dynamics (a, P1, P2) with in-plane atmospheric
drag; deterministic dynamics, Gaussian initial uncertainty
    mean = [6665.15 km, 0, 0],  std = [3.189 km, 1e-4, 1e-4].

The Galerkin FPE solution (22 basis functions per dimension, as in the
paper) is compared against a Monte Carlo run through an independent NumPy
implementation of the same averaged dynamics.

Run:  python examples/03_equinoctial_deterministic.py [--quick]
"""

import argparse
import pathlib
import time

import numpy as np

import fpe
from equinoctial_common import MEAN0, STD0, T_FINAL, YEAR, build_solver, monte_carlo

OUT = pathlib.Path(__file__).parent / "output"


def main(quick: bool = False) -> None:
    n_basis = 12 if quick else 22           # paper: 22 per dimension
    n_quad_L = 32 if quick else 64
    q = 3 if quick else 4
    n_mc = 2000 if quick else 20000
    mc_steps = 60 if quick else 150
    times = np.linspace(0.0, T_FINAL, 6 if quick else 26)

    t0 = time.perf_counter()
    solver = build_solver(n_basis, n_quad_L=n_quad_L, q=q)
    t_asm = time.perf_counter() - t0
    print(f"assembled N={solver.n} (interior {solver.active_indices.size}) in {t_asm:.1f} s")

    a0 = solver.project(fpe.GaussianPDF(MEAN0, np.diag(STD0**2)))
    t0 = time.perf_counter()
    coeffs = solver.propagate(a0, times, method="krylov")
    print(f"propagated {len(times)} epochs over 50 years in {time.perf_counter() - t0:.1f} s")

    means, stds, mass = [], [], []
    for a in coeffs:
        m, c = solver.moments(solver.normalize(a))
        means.append(m)
        stds.append(np.sqrt(np.clip(np.diag(c), 0, None)))
        mass.append(solver.integral(a))
    means, stds, mass = np.array(means), np.array(stds), np.array(mass)

    print("running Monte Carlo ground truth ...")
    mc = monte_carlo(n_mc, times, mc_steps)
    mc_mean = mc.mean(axis=1)
    mc_std = mc.std(axis=1)

    da_fpe = means[-1, 0] - means[0, 0]
    print(f"mean a drift over horizon: FPE {da_fpe:+.3f} km | MC {mc_mean[-1,0]-mc_mean[0,0]:+.3f} km")
    print(f"final |mean_a error|: {abs(means[-1,0]-mc_mean[-1,0])*1e3:.1f} m")
    print(f"final |std_a error| : {abs(stds[-1,0]-mc_std[-1,0])*1e3:.1f} m")
    print(f"max |integral - 1|  : {np.abs(mass-1).max():.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)
    yrs = times / YEAR
    labels = ["a [km]", "P1", "P2"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    for i in range(3):
        r, c = divmod(i, 2)
        axes[r, c].plot(yrs, mc_mean[:, i], "k-", label=f"Monte Carlo (N = {n_mc:,})")
        axes[r, c].plot(yrs, means[:, i], "C1--", label="Galerkin FPE")
        axes[r, c].set(xlabel="t [years]", title=f"mean {labels[i]}")
        axes[r, c].legend()
        r, c = divmod(i + 3, 2)
        axes[r, c].plot(yrs, mc_std[:, i], "k-", label=f"Monte Carlo (N = {n_mc:,})")
        axes[r, c].plot(yrs, stds[:, i], "C1--", label="Galerkin FPE")
        axes[r, c].set(xlabel="t [years]", title=f"std {labels[i]}")
        axes[r, c].legend()
    axes[3, 0].plot(yrs, mass, "C0-")
    axes[3, 0].axhline(1.0, color="k", lw=0.8)
    axes[3, 0].set(xlabel="t [years]", title="integral of approximated pdf")
    # marginal pdf of a at t0 and tf
    ag = np.linspace(solver.basis.domain[0][0], solver.basis.domain[0][1], 200)
    for a_c, color, lab in [(coeffs[0], "C3", "t = 0"), (coeffs[-1], "C0", "t = 50 y")]:
        A3 = solver.normalize(a_c).reshape(solver.basis.shape)
        tables = solver.basis.integral_tables()
        a_marg = np.tensordot(np.tensordot(A3, tables[2][0], axes=(2, 0)), tables[1][0], axes=(1, 0))
        marg = solver.basis.spline(0).basis_matrix(ag, 0) @ a_marg
        axes[3, 1].plot(ag, marg, color, label=lab)
    axes[3, 1].set(xlabel="a [km]", title="marginal pdf of semi-major axis")
    axes[3, 1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "equinoctial_deterministic.png", dpi=150)
    print(f"figure -> {OUT / 'equinoctial_deterministic.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
