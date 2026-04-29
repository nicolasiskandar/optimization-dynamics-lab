"""Tests for Newton's method."""

import numpy as np

from functions.ill_conditioned import IllConditioned
from functions.nonconvex import NonConvex
from functions.quadratic import Quadratic
from functions.saddle import Saddle
from optimizers.newton import Newton


def singular_quadratic(x):
    """Quadratic with a rank-deficient Hessian."""
    return x[0] ** 2


def test_newton_solves_quadratic_in_few_steps():
    """Newton should reach the quadratic minimizer almost immediately."""
    function = Quadratic()
    optimizer = Newton()

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=10)

    assert len(path) == 3
    assert np.linalg.norm(path[-1]) < 1e-9
    assert function.f(path[-1]) < 1e-20


def test_newton_handles_ill_conditioned_quadratic():
    """Second-order updates should still solve the anisotropic quadratic quickly."""
    function = IllConditioned()
    optimizer = Newton()

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=10)

    assert len(path) == 3
    assert function.f(path[-1]) < 1e-15


def test_newton_finds_stationary_point_on_saddle():
    """The method should move to the saddle stationary point for this quadratic."""
    function = Saddle()
    optimizer = Newton()

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=10)

    assert len(path) == 3
    assert np.linalg.norm(path[-1]) < 1e-9


def test_newton_reduces_nonconvex_objective():
    """Newton should improve the provided smooth non-convex example."""
    function = NonConvex()
    optimizer = Newton()

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=25)

    assert function.f(path[-1]) < function.f(path[0])
    assert abs(path[-1][1]) < 1e-6


def test_newton_stops_when_hessian_is_singular():
    """A singular Hessian should terminate without appending a new iterate."""
    optimizer = Newton()

    path = optimizer.optimize(singular_quadratic, x0=[1.0, -1.0], steps=5)

    assert len(path) == 1
    assert np.allclose(path[0], np.array([1.0, -1.0]))
