"""Timing of the performance-critical stages.

Run:  python benchmarks/bench_assembly.py
"""

import time

import numpy as np

import fpe
from fpe import _core


def bench(label, fn, repeat=3):
    best = min(_timed(fn) for _ in range(repeat))
    print(f"{label:55s} {best*1e3:10.1f} ms")
    return best


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    print(f"fpe {fpe.__version__}\n")

    # --- 2D oscillator, paper Sec. 5.1 size (40 x 40 basis) --------------
    basis2 = fpe.TensorBSplineBasis([(-6.0, 1.5), (-2.5, 2.5)], n_basis=40, order=3)
    dyn2 = fpe.dynamics.DampedOscillator(k=1.0, gamma=2.1)
    s2 = fpe.FokkerPlanckSolver(basis2, dyn2, diffusion=[[0.0, 0.0], [0.0, 0.08]])
    X2, W2 = basis2.element_quadrature(4)
    print(f"2D: N={basis2.n_total}, quadrature points={len(W2)}")
    bench("  dynamics eval (C++, threaded)", lambda: dyn2.eval_batch(X2))
    F2, divF2 = dyn2.eval_batch(X2)
    bench("  assemble M (gauss)", lambda: _core.assemble_M(
        basis2._cb, X2, W2, F2, divF2, s2.D))
    s2.assemble()
    a0 = s2.project(fpe.GaussianPDF([-4.0, 0.001], np.diag([0.09, 0.09])))
    ts = np.linspace(0.0, 2.05, 42)
    bench("  propagate 42 epochs (dense expm, cached)", lambda: s2.propagate(a0, ts))
    grid = np.random.default_rng(0).uniform([-6, -2.5], [1.5, 2.5], size=(200_000, 2))
    bench("  evaluate pdf at 200k points", lambda: s2.evaluate(a0, grid))

    # --- 3D equinoctial, paper Sec. 5.2 size (22^3 basis) ----------------
    from math import sqrt  # noqa: F401  (kept minimal)

    mean = np.array([6665.15, 0.0, 0.0])
    std = np.array([3.189, 1e-4, 1e-4])
    dom = list(zip(mean - 7 * std - [6, 0, 0], mean + 7 * std))
    basis3 = fpe.TensorBSplineBasis(dom, n_basis=22, order=3)
    dyn3 = fpe.dynamics.EquinoctialAveragedDrag(mu=398600.4418, delta=6e-14, n_quad_L=64)
    s3 = fpe.FokkerPlanckSolver(basis3, dyn3)
    X3, W3 = basis3.element_quadrature(4)
    print(f"\n3D: N={basis3.n_total}, quadrature points={len(W3)}")
    bench("  dynamics eval (C++ dual-number AD, threaded)", lambda: dyn3.eval_batch(X3), repeat=1)
    F3, divF3 = dyn3.eval_batch(X3)
    bench("  assemble M (gauss)", lambda: _core.assemble_M(
        basis3._cb, X3, W3, F3, divF3, s3.D), repeat=1)
    s3.assemble()
    a0 = s3.project(fpe.GaussianPDF(mean, np.diag(std**2)))
    year = 365.25 * 86400.0
    bench("  Krylov propagate 26 epochs over 50 years",
          lambda: s3.propagate(a0, np.linspace(0, 50 * year, 26), method="krylov"), repeat=1)


if __name__ == "__main__":
    main()
