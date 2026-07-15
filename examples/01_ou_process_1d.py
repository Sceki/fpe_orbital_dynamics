"""Ornstein-Uhlenbeck process: Galerkin FPE vs. the closed-form solution.

    dX = -theta X dt + s dW

The FPE solution stays Gaussian with
    mean(t) = m0 exp(-theta t)
    var(t)  = s^2/(2 theta) + (v0 - s^2/(2 theta)) exp(-2 theta t),
making this the cleanest end-to-end check of the whole pipeline.

Run:  python examples/01_ou_process_1d.py [--quick]
"""

import argparse
import pathlib

import numpy as np

import fpe

OUT = pathlib.Path(__file__).parent / "output"


def main(quick: bool = False) -> None:
    theta, s = 1.0, 0.5
    m0, v0 = 1.5, 0.2**2
    n_basis = 36 if quick else 48

    basis = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: -theta * X,
        div_f=lambda X: -theta * np.ones(X.shape[0]),
        dim=1,
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, sigma=[[s]])
    solver.assemble(quadrature="gauss")

    a0 = solver.project(fpe.GaussianPDF([m0], [[v0]]))
    times = np.linspace(0.0, 3.0, 61)
    coeffs = solver.propagate(a0, times)

    mean_num, var_num, mass = [], [], []
    for a in coeffs:
        m, c = solver.moments(solver.normalize(a))
        mean_num.append(m[0])
        var_num.append(c[0, 0])
        mass.append(solver.integral(a))
    mean_ex = m0 * np.exp(-theta * times)
    var_ex = s**2 / (2 * theta) + (v0 - s**2 / (2 * theta)) * np.exp(-2 * theta * times)

    print(f"N = {basis.n_total} basis functions")
    print(f"max |mean - exact|     : {np.abs(np.array(mean_num) - mean_ex).max():.2e}")
    print(f"max |var - exact|      : {np.abs(np.array(var_num) - var_ex).max():.2e}")
    print(f"max |integral(p) - 1|  : {np.abs(np.array(mass) - 1.0).max():.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(times, mean_ex, "k-", label="exact")
    axes[0].plot(times, mean_num, "C1--", label="Galerkin FPE")
    axes[0].set(xlabel="t", ylabel="mean", title="Mean")
    axes[0].legend()
    axes[1].plot(times, var_ex, "k-", label="exact")
    axes[1].plot(times, var_num, "C1--", label="Galerkin FPE")
    axes[1].set(xlabel="t", ylabel="variance", title="Variance")
    axes[1].legend()
    x = np.linspace(-2.5, 3.5, 400)[:, None]
    for i, c in zip([0, 20, 60], ["C0", "C2", "C3"]):
        t = times[i]
        axes[2].plot(x, solver.evaluate(coeffs[i], x), c + "-", label=f"t={t:.1f}")
        m_e = m0 * np.exp(-theta * t)
        v_e = s**2 / (2 * theta) + (v0 - s**2 / (2 * theta)) * np.exp(-2 * theta * t)
        axes[2].plot(x, fpe.GaussianPDF([m_e], [[v_e]])(x), "k:", lw=1)
    axes[2].set(xlabel="x", ylabel="p(x, t)", title="pdf evolution (dots: exact)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "ou_process_1d.png", dpi=150)
    print(f"figure -> {OUT / 'ou_process_1d.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
