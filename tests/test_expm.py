"""Matrix exponential engines vs. SciPy references."""

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from fpe import _core


def test_dense_expm_various_norms():
    rng = np.random.default_rng(0)
    for scale in [1e-4, 0.1, 1.0, 12.0, 80.0]:
        A = scale * rng.standard_normal((30, 30)) / np.sqrt(30)
        ours = np.asarray(_core.expm(A))
        ref = scipy.linalg.expm(A)
        np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_dense_expm_known_solution():
    # Rotation generator: expm([[0, -w], [w, 0]] t) is a rotation matrix.
    w, t = 0.7, 2.3
    A = np.array([[0.0, -w], [w, 0.0]])
    E = np.asarray(_core.expm(A * t))
    c, s = np.cos(w * t), np.sin(w * t)
    np.testing.assert_allclose(E, [[c, -s], [s, c]], atol=1e-13)


def _random_banded_system(n, rng, stiffness=1.0):
    """SPD mass-like B and a banded non-symmetric M, mimicking the Galerkin pair."""
    diags = [0.2 * rng.standard_normal(n - abs(k)) for k in (-2, -1, 1, 2)]
    B = sp.diags(
        [diags[0], diags[1], np.ones(n) * 1.5, diags[2], diags[3]], [-2, -1, 0, 1, 2]
    )
    B = (B @ B.T + sp.identity(n)).tocsc()  # SPD, banded
    M = sp.diags(
        [
            stiffness * rng.standard_normal(n - 2),
            stiffness * rng.standard_normal(n - 1),
            -stiffness * (0.5 + rng.random(n)),
            stiffness * rng.standard_normal(n - 1),
            stiffness * rng.standard_normal(n - 2),
        ],
        [-2, -1, 0, 1, 2],
    ).tocsc()
    return B, M


def test_krylov_matches_dense():
    rng = np.random.default_rng(1)
    n = 300
    B, M = _random_banded_system(n, rng)
    A = np.linalg.solve(B.toarray(), M.toarray())
    v = rng.standard_normal(n)
    prop = _core.KrylovPropagator(B, M, m=30, tol=1e-12)
    for t in [0.05, 1.0, 6.0]:
        ref = scipy.linalg.expm(A * t) @ v
        ours = np.asarray(prop.apply(v, t))
        np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10 * np.linalg.norm(ref))


def test_krylov_small_system_dense_fallback():
    rng = np.random.default_rng(2)
    n = 12
    B, M = _random_banded_system(n, rng)
    A = np.linalg.solve(B.toarray(), M.toarray())
    v = rng.standard_normal(n)
    prop = _core.KrylovPropagator(B, M, m=40, tol=1e-12)
    ref = scipy.linalg.expm(A * 2.0) @ v
    np.testing.assert_allclose(np.asarray(prop.apply(v, 2.0)), ref, rtol=1e-9)


def test_krylov_identity_time_zero():
    rng = np.random.default_rng(3)
    B, M = _random_banded_system(50, rng)
    v = rng.standard_normal(50)
    prop = _core.KrylovPropagator(B, M)
    np.testing.assert_allclose(np.asarray(prop.apply(v, 0.0)), v, atol=0)
