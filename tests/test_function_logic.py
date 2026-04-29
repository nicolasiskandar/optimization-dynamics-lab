"""Tests for objective-function definitions and derivative contracts."""

import numpy as np
import pytest

from core.gradients import Gradient
from core.hessian import Hessian
from functions.ill_conditioned import IllConditioned
from functions.nonconvex import NonConvex
from functions.plateau import Plateau
from functions.quadratic import Quadratic
from functions.saddle import Saddle


@pytest.mark.parametrize(
    ("function", "x"),
    [
        (Quadratic(), np.array([1.5, -2.0])),
        (IllConditioned(a=2.0, b=5.0), np.array([-1.0, 0.75])),
        (NonConvex(), np.array([0.3, -1.25])),
        (Plateau(), np.array([2.0, -3.0])),
        (Saddle(), np.array([1.2, -0.8])),
    ],
)
def test_analytic_gradients_match_numerical_gradients(function, x):
    """Each function's analytic gradient should match finite differences."""
    numeric = Gradient.get_grad(function.f, x)
    analytic = function.grad(x)

    assert np.allclose(numeric, analytic, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    ("function", "x", "tol"),
    [
        (Quadratic(), np.array([1.5, -2.0]), 5e-3),
        (IllConditioned(a=2.0, b=5.0), np.array([-1.0, 0.75]), 5e-3),
        (NonConvex(), np.array([0.3, -1.25]), 5e-3),
        (Plateau(), np.array([2.0, -3.0]), 5e-3),
        (Saddle(), np.array([1.2, -0.8]), 5e-3),
    ],
)
def test_analytic_hessians_match_numerical_hessians(function, x, tol):
    """Each function's Hessian should match the finite-difference estimate."""
    numeric = Hessian.get_hessian(function.f, x)
    analytic = function.hessian(x)

    assert np.allclose(numeric, analytic, atol=tol, rtol=0.0)


def test_plateau_function_saturates_on_large_inputs():
    """The plateau objective should approach tanh saturation limits."""
    function = Plateau()
    x = np.array([100.0, -100.0])

    value = function.f(x)
    grad = function.grad(x)

    assert abs(value) < 1e-8
    assert np.all(grad < 1e-7)


def test_nonconvex_hessian_changes_sign_with_position():
    """The non-convex function should expose changing curvature in x0."""
    function = NonConvex()

    hessian_at_zero = function.hessian(np.array([0.0, 0.0]))
    hessian_at_pi_over_two = function.hessian(np.array([np.pi / 2, 0.0]))

    assert hessian_at_zero[0, 0] > 0
    assert hessian_at_pi_over_two[0, 0] < 0
