"""4D truth-calibrated comparison: Galerkin FPE vs Monte Carlo vs EXACT.

The question this example answers: when the FPE result and a Monte Carlo
run disagree, who is wrong? Here the ground truth is known in closed form,
so every error is attributable.

System: planar Clohessy-Wiltshire relative motion under PD station-keeping
feedback with stochastic disturbance accelerations -- a fully coupled,
Hurwitz, 4D linear SDE for the state (x, y, vx, vy):

    dX = A X dt + G dW,
    A  = CW terms + (-kp r - kd v) feedback,     (kp = 4 n^2, kd = 2 n)

Being linear-Gaussian, the exact pdf is available via Van Loan's method,
and Monte Carlo samples can be drawn *exactly* (no time-discretization
bias): the MC error below is purely statistical. The initial covariance is
the stationary one (so the exact spreads stay constant while the mean
decays -- the box resolves the pdf uniformly in time).

What is measured, per epoch and against the exact solution:
  - FPE mean/std errors (deterministic, set by basis resolution), and
  - MC mean/std/KS errors for growing sample counts N (several replicates),
yielding the crossover N* where plain Monte Carlo starts to beat the
tensor-basis FPE. In 4D at affordable resolution (14 basis/dim, h ~ sigma)
that crossover is at N* ~ 1e3-1e4 for moments: an honest statement of
where the tensor-product construction stands in higher dimensions (the
paper's Sec. 4.1 caveat), in exchange for the full pdf and its instant
reuse for any initial condition.

Numerics notes (see docs/theory.md):
  - dim >= 4 uses the Kronecker-structured Gram solve (a sparse LDLT of B
    would suffer severe fill-in);
  - a weak absorbing "sponge" on the outer shell of the box damps the
    slowly-growing spurious outflow modes of the truncated-domain
    transport operator (max Re(eig): +7e-4 -> +4e-4 at 6-sigma boxes; the
    remaining growth is negligible over the ~2-damping-time horizon).

Run:  python examples/09_cw_station_keeping_4d.py [--quick]
"""

import argparse
import pathlib
import time

import numpy as np
import scipy.linalg
from scipy.special import ndtr

import fpe

OUT = pathlib.Path(__file__).parent / "output"

N_ORBIT = 1.0586e-3            # mean motion [1/s] (~700 km altitude)
KP = 4.0 * N_ORBIT**2          # position feedback
KD = 2.0 * N_ORBIT             # velocity feedback
SIGMA_POS = 0.1                # stationary position sigma [km] (sets noise)
BOX_SIGMA = 6.0
SPONGE_START = 4.6             # sponge ramp start [sigma]
SPONGE_MAX = 2e-2              # [1/s]
T_FINAL = 1600.0               # [s], ~1.7 damping times
LABELS = ["x [km]", "y [km]", "vx [km/s]", "vy [km/s]"]


def build_system():
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [3 * N_ORBIT**2 - KP, 0.0, -KD, 2 * N_ORBIT],
        [0.0, -KP, -2 * N_ORBIT, -KD],
    ])
    assert np.linalg.eigvals(A).real.max() < 0
    GGt = np.diag([0.0, 0.0, 1.0, 1.0])
    P = scipy.linalg.solve_continuous_lyapunov(A, -GGt)
    scale2 = SIGMA_POS**2 / P[0, 0]
    return A, GGt * scale2, P * scale2


A, GGT, P_INF = build_system()
SIG = np.sqrt(np.diag(P_INF))
M0 = np.array([1.0 * SIG[0], -1.0 * SIG[1], 0.5 * SIG[2], 0.0])


def exact_moments(t):
    n = 4
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -A
    C[:n, n:] = GGT
    C[n:, n:] = A.T
    E = scipy.linalg.expm(C * t)
    Phi = E[n:, n:].T
    return Phi @ M0, Phi @ P_INF @ Phi.T + Phi @ E[:n, n:]


def sponge(X):
    r = np.zeros(X.shape[0])
    for d in range(4):
        s = np.clip((np.abs(X[:, d]) / SIG[d] - SPONGE_START) / (BOX_SIGMA - SPONGE_START),
                    0.0, 1.0)
        r = np.maximum(r, s * s * (3.0 - 2.0 * s))
    return SPONGE_MAX * r


def fpe_solve(n_basis, times, q=3):
    domain = [(-BOX_SIGMA * SIG[d], BOX_SIGMA * SIG[d]) for d in range(4)]
    basis = fpe.TensorBSplineBasis(domain, n_basis=n_basis, order=3)
    dyn = fpe.dynamics.CallableDynamics(
        f=lambda X: X @ A.T, div_f=lambda X: np.trace(A) * np.ones(X.shape[0]), dim=4
    )
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=0.5 * GGT)
    t0 = time.perf_counter()
    solver.assemble(quadrature="gauss", q=q)  # q=3: exact for the linear drift
    solver.add_sink(sponge)
    t_asm = time.perf_counter() - t0
    a0 = solver.project(fpe.GaussianPDF(M0, P_INF))
    t0 = time.perf_counter()
    coeffs = solver.propagate(a0, times, method="krylov")
    t_prop = time.perf_counter() - t0
    print(f"FPE: N={solver.n} (interior {solver.active_indices.size}), "
          f"assemble {t_asm:.0f} s, propagate {t_prop:.0f} s")
    return solver, coeffs


def marginal_cdf(solver, a, dim):
    lo, hi = solver.basis.domain[dim]
    xs = np.linspace(lo, hi, 3001)
    pdf = np.maximum(solver.marginal(solver.normalize(a), dim, xs), 0.0)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs))])
    c /= c[-1]
    return lambda q, xs=xs, c=c: np.interp(q, xs, c, left=0.0, right=1.0)


def main(quick: bool = False) -> None:
    n_basis = 12 if quick else 14
    ladder = [1_000, 10_000, 100_000] if quick else [1_000, 10_000, 100_000, 1_000_000]
    replicates = 3 if quick else 5
    times = np.linspace(0.0, T_FINAL, 5)

    solver, coeffs = fpe_solve(n_basis, times)

    # ---- FPE vs exact ---------------------------------------------------
    fpe_mean_err = np.empty((len(times), 4))
    fpe_std_err = np.empty((len(times), 4))
    for i, (t, a) in enumerate(zip(times, coeffs)):
        m_ref, P_ref = exact_moments(t)
        mean, cov = solver.moments(solver.normalize(a))
        fpe_mean_err[i] = np.abs(mean - m_ref) / SIG
        fpe_std_err[i] = np.abs(np.sqrt(np.clip(np.diag(cov), 0, None)) / np.sqrt(np.diag(P_ref)) - 1)

    # KS floor of the FPE x/y marginals at the final epoch
    m_f, P_f = exact_moments(times[-1])
    ks_floor = []
    for d in (0, 1):
        cdf_fpe = marginal_cdf(solver, coeffs[-1], d)
        xs = np.linspace(*solver.basis.domain[d], 20001)
        cdf_ex = ndtr((xs - m_f[d]) / np.sqrt(P_f[d, d]))
        ks_floor.append(float(np.abs(cdf_fpe(xs) - cdf_ex).max()))

    # ---- MC vs exact (exact sampling: zero bias, pure statistics) -------
    Lf = np.linalg.cholesky(P_f)
    rng = np.random.default_rng(0)
    mc_mean_err = np.empty((len(ladder), replicates))
    mc_std_err = np.empty((len(ladder), replicates))
    mc_ks = np.empty((len(ladder), replicates))
    for r in range(replicates):
        pool = m_f + rng.standard_normal((max(ladder), 4)) @ Lf.T
        for i, N in enumerate(ladder):
            S = pool[:N]
            mc_mean_err[i, r] = (np.abs(S.mean(axis=0) - m_f) / SIG).max()
            mc_std_err[i, r] = np.abs(S.std(axis=0) / np.sqrt(np.diag(P_f)) - 1).max()
            mc_ks[i, r] = fpe.metrics.ks_statistic(  # MC vs EXACT, x-marginal
                S[:, 0], lambda q: ndtr((q - m_f[0]) / np.sqrt(P_f[0, 0]))
            )

    # ---- report ----------------------------------------------------------
    print(f"\nexact ground truth (Van Loan); MC uses exact sampling (bias-free)")
    print(f"FPE errors at t={T_FINAL:.0f} s: max |mean|/sigma = {fpe_mean_err[-1].max():.4f}, "
          f"max std rel = {fpe_std_err[-1].max():.4f}, KS(x) = {ks_floor[0]:.4f}")
    print(f"{'N':>9} {'|mean|/sig (MC)':>16} {'std rel (MC)':>13} {'KS x (MC)':>10}")
    for i, N in enumerate(ladder):
        print(f"{N:>9d} {mc_mean_err[i].mean():>16.4f} {mc_std_err[i].mean():>13.4f} "
              f"{mc_ks[i].mean():>10.4f}")

    def crossover(mc_err, floor):
        """Sample count where the mean MC error crosses the FPE floor."""
        m = mc_err.mean(axis=1)
        if m[-1] > floor:
            return None
        i = int(np.searchsorted(-m, -floor))  # first index with m[i] <= floor
        if i == 0:
            return float(ladder[0])
        return float(np.exp(np.interp(np.log(floor), np.log(m[[i, i - 1]]),
                                      np.log([ladder[i], ladder[i - 1]]))))

    for name, arr, floor in [
        ("mean", mc_mean_err, fpe_mean_err[-1].max()),
        ("std", mc_std_err, fpe_std_err[-1].max()),
        ("KS(x)", mc_ks, ks_floor[0]),
    ]:
        n_star = crossover(arr, floor)
        msg = f"~{n_star:,.0f}" if n_star else f"> {ladder[-1]:,}"
        print(f"crossover N* ({name}): MC beats the {n_basis}^4-basis FPE beyond {msg} samples")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    OUT.mkdir(exist_ok=True)
    N_arr = np.array(ladder, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    panels = [
        (axes[0, 0], mc_mean_err, fpe_mean_err[-1].max(),
         r"max |mean error| / $\sigma$", "mean error"),
        (axes[0, 1], mc_std_err, fpe_std_err[-1].max(),
         "max relative std error", "std error"),
        (axes[1, 0], mc_ks, ks_floor[0], "KS distance, x-marginal", "KS(x) vs exact"),
    ]
    for ax, arr, floor, ylab, title in panels:
        for r in range(arr.shape[1]):
            ax.loglog(N_arr, arr[:, r], "o", ms=3, color="C0", alpha=0.4)
        ax.loglog(N_arr, arr.mean(axis=1), "C0-",
                  label=f"Monte Carlo (exact sampling, {arr.shape[1]} replicates)")
        ax.loglog(N_arr, arr.mean(axis=1)[0] * np.sqrt(N_arr[0] / N_arr), "k--", lw=1,
                  label=r"$\propto 1/\sqrt{N}$")
        ax.axhline(floor, color="C3", ls=":", lw=1.5,
                   label=f"Galerkin FPE ({n_basis}$^4$ basis) vs exact")
        ax.set(xlabel="Monte Carlo samples N", ylabel=ylab,
               title=f"{title} at t = {T_FINAL:.0f} s")
        ax.legend(fontsize=8)

    # 2D (x, y) marginal: FPE vs exact vs MC histogram
    ax = axes[1, 1]
    gx = np.linspace(m_f[0] - 4.5 * SIG[0], m_f[0] + 4.5 * SIG[0], 80)
    gy = np.linspace(m_f[1] - 4.5 * SIG[1], m_f[1] + 4.5 * SIG[1], 80)
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    P_fpe = solver.marginal(solver.normalize(coeffs[-1]), (0, 1), pts).reshape(XX.shape)
    P_ex = fpe.GaussianPDF(m_f[:2], P_f[:2, :2])(pts).reshape(XX.shape)
    n_show = ladder[-2]
    S = m_f + np.random.default_rng(1).standard_normal((n_show, 4)) @ Lf.T
    ax.hist2d(S[:, 0], S[:, 1], bins=60, range=[(gx[0], gx[-1]), (gy[0], gy[-1])],
              cmap="Greys")
    levels = np.linspace(0.08, 0.9, 5) * P_ex.max()  # avoid tracing tail ripples
    ax.contour(XX, YY, P_fpe, levels=levels, colors="C1")
    ax.contour(XX, YY, P_ex, levels=levels, colors="C2", linestyles="--")
    ax.set(xlabel="x [km]", ylabel="y [km]",
           title=f"(x, y) marginal at t = {T_FINAL:.0f} s\n"
                 f"FPE (solid), exact (dashed), MC histogram (N = {n_show:,})")
    fig.suptitle("4D station-keeping: exact truth attributes the errors -- FPE floor vs MC statistics")
    fig.tight_layout()
    fig.savefig(OUT / "cw_station_keeping_4d.png", dpi=150)
    print(f"figure -> {OUT / 'cw_station_keeping_4d.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced-size run")
    main(**vars(ap.parse_args()))
