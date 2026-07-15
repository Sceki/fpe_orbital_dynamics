"""Galerkin assembly and L2 projection correctness."""

import numpy as np
import scipy.sparse as sp

import fpe
from fpe import _core


def _basis_2d():
    return fpe.TensorBSplineBasis([(-2.0, 2.0), (-1.0, 3.0)], n_basis=[9, 8], order=3)


def test_gram_kron_is_spd_and_banded():
    basis = _basis_2d()
    B = basis.gram_kron()
    assert B.shape == (72, 72)
    dense = B.toarray()
    np.testing.assert_allclose(dense, dense.T, atol=1e-14)
    eigvals = np.linalg.eigvalsh(dense)
    assert eigvals.min() > 0, "Gram matrix must be SPD"
    # Kronecker band sparsity from the paper's Sec. 4.1 condition.
    for k in range(72):
        for j in range(72):
            k0, k1 = divmod(k, 8)
            j0, j1 = divmod(j, 8)
            if abs(k0 - j0) >= 3 or abs(k1 - j1) >= 3:
                assert dense[k, j] == 0.0


def test_projection_recovers_spline_function():
    """Projecting a function already in the span must return its coefficients."""
    basis = _basis_2d()
    rng = np.random.default_rng(0)
    a_true = rng.random(basis.n_total)
    solver = fpe.FokkerPlanckSolver(basis, boundary="free")

    def p0(X):
        return basis.evaluate(a_true, X)

    a_proj = solver.project(p0)
    np.testing.assert_allclose(a_proj, a_true, atol=1e-10)


def test_dirichlet_projection_zeroes_boundary_coefficients():
    basis = _basis_2d()
    solver = fpe.FokkerPlanckSolver(basis)  # dirichlet by default
    a0 = solver.project(fpe.GaussianPDF([0.0, 1.0], np.diag([0.15, 0.15])))
    A = a0.reshape(basis.shape)
    assert np.all(A[0, :] == 0) and np.all(A[-1, :] == 0)
    assert np.all(A[:, 0] == 0) and np.all(A[:, -1] == 0)
    # ... and the pdf is exactly zero on the boundary of the box.
    edge = np.array([[-2.0, 1.3], [2.0, 0.2], [0.4, -1.0], [1.1, 3.0]])
    np.testing.assert_allclose(solver.evaluate(a0, edge), 0.0, atol=1e-14)


def test_projection_of_gaussian_evaluates_accurately():
    basis = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=30, order=3)
    solver = fpe.FokkerPlanckSolver(basis)
    p0 = fpe.GaussianPDF([0.3], [[0.4]])
    a0 = solver.project(p0)
    x = np.linspace(-3.0, 3.0, 200)[:, None]
    # L2 best-approximation error of 30 quadratic splines on this Gaussian.
    np.testing.assert_allclose(solver.evaluate(a0, x), p0(x), atol=1.5e-3)
    assert abs(solver.integral(a0) - 1.0) < 1e-5
    # Refining the basis must shrink the error (h^order convergence).
    fine = fpe.TensorBSplineBasis([(-4.0, 4.0)], n_basis=60, order=3)
    a_fine = fpe.FokkerPlanckSolver(fine).project(p0)
    err_coarse = np.abs(solver.evaluate(a0, x) - p0(x)).max()
    err_fine = np.abs(fine.evaluate(a_fine, x) - p0(x)).max()
    assert err_fine < err_coarse / 4.0


def test_assemble_M_sparsity_pattern():
    basis = _basis_2d()
    dyn = fpe.dynamics.DampedOscillator(k=1.0, gamma=0.5)
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, 0.1]])
    solver.assemble(quadrature="gauss")
    M = solver.M.toarray()
    for k in range(basis.n_total):
        for j in range(basis.n_total):
            k0, k1 = divmod(k, 8)
            j0, j1 = divmod(j, 8)
            if abs(k0 - j0) >= 3 or abs(k1 - j1) >= 3:
                assert M[k, j] == 0.0, "paper Sec. 4.1: entries beyond the band must vanish"


def test_assemble_M_matches_direct_quadrature():
    """Cross-check the element-grouped assembly against a brute-force
    computation of a few entries with the same quadrature points."""
    basis = fpe.TensorBSplineBasis([(-1.0, 1.0), (-1.0, 1.0)], n_basis=[5, 5], order=3)
    dyn = fpe.dynamics.DampedOscillator(k=2.0, gamma=1.3)
    D = np.array([[0.0, 0.0], [0.0, 0.07]])
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=D)
    solver.assemble(quadrature="gauss", q=4)
    X, W = basis.element_quadrature(4)
    F, divF = dyn.eval_batch(X)

    B0 = [basis.spline(d).basis_matrix(X[:, d], 0) for d in range(2)]
    B1 = [basis.spline(d).basis_matrix(X[:, d], 1) for d in range(2)]
    B2 = [basis.spline(d).basis_matrix(X[:, d], 2) for d in range(2)]

    rng = np.random.default_rng(1)
    for _ in range(30):
        k = rng.integers(0, 25)
        j = rng.integers(0, 25)
        k0, k1 = divmod(int(k), 5)
        j0, j1 = divmod(int(j), 5)
        phik = B0[0][:, k0] * B0[1][:, k1]
        phij = B0[0][:, j0] * B0[1][:, j1]
        dxj = B1[0][:, j0] * B0[1][:, j1]
        dvj = B0[0][:, j0] * B1[1][:, j1]
        dvvj = B0[0][:, j0] * B2[1][:, j1]
        integrand = phik * (
            -F[:, 0] * dxj - F[:, 1] * dvj - divF * phij + D[1, 1] * dvvj
        )
        expected = float(np.sum(W * integrand))
        assert abs(solver.M[int(k), int(j)] - expected) < 1e-11


def test_halton_assembly_converges_to_gauss():
    basis = fpe.TensorBSplineBasis([(-2.0, 2.0), (-2.0, 2.0)], n_basis=[6, 6], order=3)
    dyn = fpe.dynamics.DampedOscillator(k=1.0, gamma=2.1)
    D = [[0.0, 0.0], [0.0, 0.08]]
    s_gauss = fpe.FokkerPlanckSolver(basis, dyn, diffusion=D).assemble(quadrature="gauss")
    s_halton = fpe.FokkerPlanckSolver(basis, dyn, diffusion=D).assemble(
        quadrature="halton", n_points=400_000
    )
    ref = s_gauss.M.toarray()
    approx = s_halton.M.toarray()
    scale = np.abs(ref).max()
    assert np.abs(ref - approx).max() < 5e-3 * scale


def test_moments_are_exact_for_the_spline_pdf():
    """moments() must integrate the *spline approximation* exactly: compare
    against dense numerical integration of the same evaluated pdf."""
    basis = fpe.TensorBSplineBasis([(-3.0, 3.0), (-2.0, 4.0)], n_basis=[12, 11], order=3)
    solver = fpe.FokkerPlanckSolver(basis)
    p0 = fpe.GaussianPDF([0.5, 1.2], np.array([[0.5, 0.1], [0.1, 0.3]]))
    a0 = solver.project(p0)
    mean, cov = solver.moments(a0)

    x = np.linspace(-3.0, 3.0, 801)
    y = np.linspace(-2.0, 4.0, 801)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    P = solver.evaluate(a0, grid).reshape(XX.shape)

    from scipy.integrate import trapezoid

    def integ2d(f):
        return trapezoid(trapezoid(f, y, axis=1), x, axis=0)

    mass = integ2d(P)
    m_ref = np.array([integ2d(P * XX), integ2d(P * YY)]) / mass
    raw = np.array(
        [
            [integ2d(P * XX * XX), integ2d(P * XX * YY)],
            [integ2d(P * XX * YY), integ2d(P * YY * YY)],
        ]
    ) / mass
    cov_ref = raw - np.outer(m_ref, m_ref)
    np.testing.assert_allclose(mean, m_ref, atol=2e-6)
    np.testing.assert_allclose(cov, cov_ref, atol=2e-5)
    # And the projection itself reproduces the Gaussian's moments closely.
    np.testing.assert_allclose(mean, [0.5, 1.2], atol=2e-3)
    np.testing.assert_allclose(cov, [[0.5, 0.1], [0.1, 0.3]], atol=5e-3)


def test_marginal_matches_gaussian_marginal():
    basis = fpe.TensorBSplineBasis([(-4.0, 4.0), (-3.0, 5.0)], n_basis=[24, 22], order=3)
    solver = fpe.FokkerPlanckSolver(basis)
    cov = np.array([[0.5, 0.15], [0.15, 0.4]])
    a0 = solver.project(fpe.GaussianPDF([0.2, 1.0], cov))
    x = np.linspace(-2.5, 2.9, 200)
    # Marginal of a joint Gaussian is the 1D Gaussian of that component.
    # atol reflects the projection resolution (22-24 quadratic splines),
    # not the marginalization itself, which is an exact contraction.
    m0 = solver.marginal(a0, 0, x)
    ref0 = fpe.GaussianPDF([0.2], [[0.5]])(x[:, None])
    np.testing.assert_allclose(m0, ref0, atol=4e-3)
    y = np.linspace(-1.0, 3.0, 200)
    m1 = solver.marginal(a0, 1, y)
    ref1 = fpe.GaussianPDF([1.0], [[0.4]])(y[:, None])
    np.testing.assert_allclose(m1, ref1, atol=4e-3)


def test_save_load_roundtrip(tmp_path):
    basis = _basis_2d()
    dyn = fpe.dynamics.DampedOscillator(k=1.0, gamma=2.1)
    solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, 0.08]])
    solver.assemble()
    path = str(tmp_path / "solver.npz")
    solver.save(path)
    loaded = fpe.FokkerPlanckSolver.load(path)
    assert (loaded.B != solver.B).nnz == 0
    assert (loaded.M != solver.M).nnz == 0
    # The loaded solver must propagate identically without a dynamics object.
    p0 = fpe.GaussianPDF([0.0, 1.0], np.diag([0.2, 0.2]))
    a0 = solver.project(p0)
    t = np.array([0.0, 0.4])
    np.testing.assert_allclose(
        loaded.propagate(a0, t), solver.propagate(a0, t), atol=1e-12
    )
