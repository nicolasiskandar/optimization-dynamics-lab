"""Tests for numerical Hessian approximation."""

import numpy as np
import pytest

from core.hessian import Hessian


def quadratic(x):
    """Smooth quadratic with diagonal Hessian."""
    return x[0] ** 2 + 3.0 * x[1] ** 2


def coupled_quadratic(x):
    """Quadratic with off-diagonal curvature."""
    return x[0] ** 2 + x[0] * x[1] + 2.0 * x[1] ** 2


@pytest.mark.parametrize(
    ("f", "x", "expected", "tolerance"),
    [
        (quadratic, np.array([3.0, -4.0]),
         np.array([[2.0, 0.0], [0.0, 6.0]]), 5e-3),
        (coupled_quadratic, np.array([-1.0, 2.0]),
         np.array([[2.0, 1.0], [1.0, 4.0]]), 5e-3),
    ],
)
def test_hessian_matches_analytic_quadratics(f, x, expected, tolerance):
    """Finite-difference Hessians should match analytic second derivatives."""
    numeric = Hessian.get_hessian(f, x)

    assert np.allclose(numeric, expected, atol=tolerance, rtol=0.0)


def test_hessian_is_symmetric_for_smooth_function():
    """Numerical Hessian of a smooth scalar objective should be symmetric."""
    x = np.array([1.25, -0.75])

    numeric = Hessian.get_hessian(coupled_quadratic, x)

    assert np.allclose(numeric, numeric.T, atol=1e-8, rtol=0.0)
