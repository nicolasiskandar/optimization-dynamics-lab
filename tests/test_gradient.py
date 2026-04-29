"""Tests for numerical gradient approximation."""

import numpy as np
import pytest

from core.gradients import Gradient


def quadratic(x):
    """Simple convex quadratic with known analytic gradient."""
    return x[0] ** 2 + 3.0 * x[1] ** 2


def quadratic_grad(x):
    """Analytic gradient for ``quadratic``."""
    return np.array([2.0 * x[0], 6.0 * x[1]])


@pytest.mark.parametrize(
    ("x", "tolerance"),
    [
        (np.array([3.0, -4.0]), 1e-6),
        (np.array([0.5, 2.25]), 1e-6),
        (np.array([-1.2, 0.0]), 1e-6),
    ],
)
def test_gradient_matches_analytic_quadratic(x, tolerance):
    """Central differences should recover gradients for smooth quadratics."""
    numeric = Gradient.get_grad(quadratic, x)
    analytic = quadratic_grad(x)

    assert np.allclose(numeric, analytic, atol=tolerance, rtol=0.0)


def test_gradient_returns_same_shape_as_input_point():
    """Gradient output should preserve the dimensionality of the input."""
    x = [1.0, -2.0]

    grad = Gradient.get_grad(quadratic, x)

    assert grad.shape == (2,)


def test_gradient_is_zero_at_quadratic_minimum():
    """The numerical gradient should vanish near the minimizer."""
    grad = Gradient.get_grad(quadratic, np.zeros(2))

    assert np.allclose(grad, np.zeros(2), atol=1e-9, rtol=0.0)
