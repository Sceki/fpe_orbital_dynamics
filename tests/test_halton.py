"""Halton sequence correctness."""

import numpy as np

from fpe import _core


def test_first_values_base2_base3():
    # Van der Corput in base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8 ...
    # base 3: 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9 ...
    pts = _core.halton(7, 2, skip=1)
    expected_b2 = [1 / 2, 1 / 4, 3 / 4, 1 / 8, 5 / 8, 3 / 8, 7 / 8]
    expected_b3 = [1 / 3, 2 / 3, 1 / 9, 4 / 9, 7 / 9, 2 / 9, 5 / 9]
    np.testing.assert_allclose(pts[:, 0], expected_b2, atol=1e-15)
    np.testing.assert_allclose(pts[:, 1], expected_b3, atol=1e-15)


def test_qmc_integrates_polynomial():
    # QMC equal-weight rule (paper Eq. 19) on int_[0,1]^2 x*y^2 = 1/6.
    pts = np.asarray(_core.halton(20000, 2))
    est = np.mean(pts[:, 0] * pts[:, 1] ** 2)
    assert abs(est - 1 / 6) < 5e-4
