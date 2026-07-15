"""B-spline basis correctness: partition of unity, derivatives, exact integrals."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

import fpe
from fpe import _core


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("n_basis", [6, 13])
def test_partition_of_unity(order, n_basis):
    s = _core.BSpline1D(-2.0, 3.0, n_basis, order)
    x = np.linspace(-2.0, 3.0, 257)
    Bmat = s.basis_matrix(x, der=0)
    np.testing.assert_allclose(Bmat.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(Bmat >= -1e-14), "B-splines are non-negative"


@pytest.mark.parametrize("order", [3, 4])
def test_derivatives_match_finite_differences(order):
    s = _core.BSpline1D(0.0, 1.0, 11, order)
    # Stay away from knots, where higher derivatives are discontinuous.
    x = np.array([0.03, 0.171, 0.333, 0.61, 0.777, 0.949])
    h = 1e-6
    B0p = s.basis_matrix(x + h, der=0)
    B0m = s.basis_matrix(x - h, der=0)
    B1 = s.basis_matrix(x, der=1)
    np.testing.assert_allclose(B1, (B0p - B0m) / (2 * h), atol=1e-5)
    B1p = s.basis_matrix(x + h, der=1)
    B1m = s.basis_matrix(x - h, der=1)
    B2 = s.basis_matrix(x, der=2)
    np.testing.assert_allclose(B2, (B1p - B1m) / (2 * h), atol=1e-4)


def test_against_scipy_bspline():
    scipy_interp = pytest.importorskip("scipy.interpolate")
    if not hasattr(scipy_interp.BSpline, "design_matrix"):
        pytest.skip("scipy too old for BSpline.design_matrix")
    order, n_basis = 3, 9
    s = _core.BSpline1D(-1.0, 2.0, n_basis, order)
    x = np.linspace(-1.0, 2.0, 101, endpoint=False)
    ours = s.basis_matrix(x, der=0)
    theirs = scipy_interp.BSpline.design_matrix(x, np.asarray(s.knots), order - 1).toarray()
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_gram_matches_quadrature():
    s = _core.BSpline1D(0.0, 2.0, 8, 3)
    G = s.gram()
    # Band structure: <phi_i, phi_j> = 0 for |i-j| >= order (paper Sec. 4.1).
    for i in range(8):
        for j in range(8):
            if abs(i - j) >= 3:
                assert G[i, j] == 0.0
    # Compare against a dense trapezoid integration.
    x = np.linspace(0.0, 2.0, 20001)
    Bmat = s.basis_matrix(x, der=0)
    G_ref = trapezoid(Bmat[:, :, None] * Bmat[:, None, :], x, axis=0)
    np.testing.assert_allclose(G, G_ref, atol=1e-6)


def test_integrals_exact():
    s = _core.BSpline1D(-1.0, 1.5, 10, 3)
    I0, I1, I2 = s.integrals()
    # Sum of all basis integrals = length of interval (partition of unity).
    assert abs(I0.sum() - 2.5) < 1e-12
    x = np.linspace(-1.0, 1.5, 40001)
    Bmat = s.basis_matrix(x, der=0)
    np.testing.assert_allclose(I0, trapezoid(Bmat, x, axis=0), atol=1e-7)
    np.testing.assert_allclose(I1, trapezoid(Bmat * x[:, None], x, axis=0), atol=1e-7)
    np.testing.assert_allclose(I2, trapezoid(Bmat * (x**2)[:, None], x, axis=0), atol=1e-7)


def test_tensor_basis_layout():
    basis = fpe.TensorBSplineBasis([(-1, 1), (0, 2)], n_basis=[5, 4], order=3)
    assert basis.dim == 2
    assert basis.shape == (5, 4)
    assert basis.n_total == 20
    # Row-major flattening: evaluating basis function (i, j) at x equals the
    # product of the 1D functions.
    a = np.zeros(20)
    i, j = 2, 3
    a[i * 4 + j] = 1.0
    X = np.array([[0.3, 0.7], [-0.5, 1.9]])
    vals = basis.evaluate(a, X)
    b0 = basis.spline(0).basis_matrix(X[:, 0], 0)[:, i]
    b1 = basis.spline(1).basis_matrix(X[:, 1], 0)[:, j]
    np.testing.assert_allclose(vals, b0 * b1, atol=1e-13)
