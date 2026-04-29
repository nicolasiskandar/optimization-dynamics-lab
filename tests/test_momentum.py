"""Tests for momentum gradient descent."""

import numpy as np

from functions.nonconvex import NonConvex
from functions.quadratic import Quadratic
from functions.saddle import Saddle
from optimizers.momentum import MomentumGD


def test_momentum_reduces_quadratic_objective_substantially():
    """Momentum should make clear progress on the quadratic benchmark."""
    function = Quadratic()
    optimizer = MomentumGD(step_size=0.1, beta=0.9)

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)

    assert path.shape[1] == 2
    assert function.f(path[-1]) < 1e-3
    assert function.f(path[-1]) < function.f(path[0])


def test_momentum_improves_nonconvex_objective():
    """The implementation should reduce the provided non-convex test loss."""
    function = NonConvex()
    optimizer = MomentumGD(step_size=0.1, beta=0.9)

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)

    assert function.f(path[-1]) < function.f(path[0])
    assert abs(path[-1][1]) < 5e-2


def test_momentum_exhibits_saddle_instability():
    """The fixed-step momentum setup should diverge along the saddle descent axis."""
    function = Saddle()
    optimizer = MomentumGD(step_size=0.1, beta=0.9)

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=60)

    assert function.f(path[-1]) < function.f(path[0])
    assert abs(path[-1][1]) > abs(path[0][1])
    assert np.isfinite(path).all()
