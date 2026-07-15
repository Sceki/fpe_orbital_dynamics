"""Paper Sec. 5.3: equinoctial dynamics with a diffusive term on P1 (3D).

Same setup as example 03, plus additive white noise on dP1/dt:
    dP1 = f_P1(a, P1, P2) dt + sqrt(2 sigma) dW.

The diffusion magnitude is sized so the P1 dispersion roughly doubles over
the 50-year horizon (qualitatively matching the paper's Fig. 3, where the
diffusive term visibly inflates the P1 covariance while the deterministic
case would shrink it). The Galerkin solution is compared against an
Euler-Maruyama Monte Carlo run of the same SDE.

Run:  python examples/04_equinoctial_stochastic.py [--quick]
"""

import argparse
import pathlib

import numpy as np

import fpe
from equinoctial_common import MEAN0, STD0, T_FINAL, YEAR, build_solver, make_domain, monte_carlo

OUT = pathlib.Path(__file__).parent / "output"

# sqrt(2 sigma): white-noise strength on P1 [1/sqrt(s)]. D_P1P1 = sigma.
NOISE_P1 = 4.4e-9
SIGMA_P1 = 0.5 * NOISE_P1**2


def main(quick: bool = False) -> None:
    n_basis = 12 if quick else 22
    n_quad_L = 32 if quick else 64
    q = 3 if quick else 4
    n_mc = 2000 if quick else 20000
    mc_steps = 120 if quick else 300
    times = np.linspace(0.0, T_FINAL, 6 if quick else 26)

    D = np.zeros((3, 3))
    D[1, 1] = SIGMA_P1
    # Widen the P1 axis: the diffusive spread must stay inside the box.
    domain = make_domain(p_half_width=1.2e-3)
    solver = build_solver(n_basis, diffusion=D, n_quad_L=n_quad_L, q=q, domain=domain)
    print(f"assembled N={solver.n} (interior {solver.active_indices.size})")

    a0 = solver.project(fpe.GaussianPDF(MEAN0, np.diag(STD0**2)))
    coeffs = solver.propagate(a0, times, method="krylov")

    stds, mass = [], []
    for a in coeffs:
        _, c = solver.moments(solver.normalize(a))
        stds.append(np.sqrt(np.clip(np.diag(c), 0, None)))
        mass.append(solver.integral(a))
    stds, mass = np.array(stds), np.array(mass)

    print("running Euler-Maruyama Monte Carlo ...")
    mc = monte_carlo(n_mc, times, mc_steps, noise_p1=NOISE_P1)
    mc_std = mc.std(axis=1)

    print(f"P1 std: initial {stds[0,1]:.2e} -> final FPE {stds[-1,1]:.2e} | MC {mc_std[-1,1]:.2e}")
    print(f"max |integral - 1|: {np.abs(mass-1).max():.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)
    yrs = times / YEAR
    labels = ["a [km]", "P1", "P2"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i in range(3):
        r, c = divmod(i, 2)
        axes[r, c].plot(yrs, mc_std[:, i], "k-", label=f"Monte Carlo (N = {n_mc:,})")
        axes[r, c].plot(yrs, stds[:, i], "C1--", label="Galerkin FPE")
        axes[r, c].set(xlabel="t [years]", title=f"std {labels[i]}")
        axes[r, c].legend()
    axes[1, 1].plot(yrs, mass, "C0-")
    axes[1, 1].axhline(1.0, color="k", lw=0.8)
    axes[1, 1].set(xlabel="t [years]", title="integral of approximated pdf")
    fig.tight_layout()
    fig.savefig(OUT / "equinoctial_stochastic.png", dpi=150)
    print(f"figure -> {OUT / 'equinoctial_stochastic.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
