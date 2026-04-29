"""Tests for fixed-step gradient descent."""

import numpy as np

from functions.ill_conditioned import IllConditioned
from functions.nonconvex import NonConvex
from functions.quadratic import Quadratic
from optimizers.gradient_descent import GradientDescent


def test_gradient_descent_converges_on_quadratic():
    """A stable fixed step should drive the quadratic close to its minimizer."""
    function = Quadratic()
    optimizer = GradientDescent(step_size=0.1)

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)
    values = np.array([function.f(point) for point in path])

    assert path.shape[1] == 2
    assert len(path) < 101
    assert np.all(np.diff(values) <= 1e-10)
    assert values[-1] < 1e-10
    assert np.linalg.norm(path[-1]) < 1e-3


def test_gradient_descent_reduces_nonconvex_objective():
    """The current default setup should still improve the non-convex example."""
    function = NonConvex()
    optimizer = GradientDescent(step_size=0.1)

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)

    assert function.f(path[-1]) < function.f(path[0])
    assert abs(path[-1][1]) < 1e-6


def test_gradient_descent_diverges_on_ill_conditioned_problem_with_large_step():
    """This documents the known instability of a large fixed step."""
    function = IllConditioned()
    optimizer = GradientDescent(step_size=0.1)

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=8)

    assert function.f(path[-1]) > function.f(path[0])
    assert abs(path[-1][1]) > abs(path[0][1])
