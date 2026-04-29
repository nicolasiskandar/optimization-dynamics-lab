"""Tests for line-search behavior."""

import numpy as np
import pytest

from functions.ill_conditioned import IllConditioned
from functions.quadratic import Quadratic
from optimizers import GradientDescentWithLineSearch, MomentumWithLineSearch
from optimizers.line_search import LineSearch


@pytest.mark.parametrize(
    ("method_name", "line_search"),
    [
        (
            "backtracking",
            lambda f, x, d: LineSearch.backtracking_line_search(
                f, x, d, initial_step=1.0
            ),
        ),
        (
            "golden_section",
            lambda f, x, d: LineSearch.golden_section_search(
                f, x, d, initial_step=1.0
            ),
        ),
    ],
)
def test_line_search_methods_find_near_optimal_quadratic_step(method_name, line_search):
    """Both line search methods should choose about 0.5 on the quadratic bowl."""
    function = Quadratic()
    x = np.array([3.0, -4.0])
    direction = -np.array([6.0, -8.0])

    step = line_search(function.f, x, direction)

    assert 0.0 < step <= 1.0
    assert step == pytest.approx(0.5, abs=2e-2), method_name
    assert function.f(x + step * direction) < function.f(x)


@pytest.mark.parametrize("line_search_method", ["backtracking", "golden_section"])
def test_gradient_descent_with_line_search_converges_on_quadratic(line_search_method):
    """Adaptive step size should solve the quadratic quickly for both modes."""
    function = Quadratic()
    optimizer = GradientDescentWithLineSearch(
        line_search_method=line_search_method,
        initial_step=1.0,
    )

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)

    assert function.f(path[-1]) < 1e-8
    assert len(path) <= 3 if line_search_method == "golden_section" else len(
        path) <= 4


@pytest.mark.parametrize("line_search_method", ["backtracking", "golden_section"])
def test_gradient_descent_with_line_search_handles_ill_conditioned_problem(line_search_method):
    """Line search should stabilize a case where fixed-step GD diverges."""
    function = IllConditioned()
    optimizer = GradientDescentWithLineSearch(
        line_search_method=line_search_method,
        initial_step=1.0,
    )

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)

    assert function.f(path[-1]) < 0.1
    assert function.f(path[-1]) < function.f(path[0])


@pytest.mark.parametrize(
    ("line_search_method", "upper_bound"),
    [("backtracking", 1e-4), ("golden_section", 1e-6)],
)
def test_momentum_with_line_search_converges_on_quadratic(line_search_method, upper_bound):
    """Momentum plus line search should significantly reduce the quadratic."""
    function = Quadratic()
    optimizer = MomentumWithLineSearch(
        beta=0.9,
        line_search_method=line_search_method,
        initial_step=1.0,
    )

    path = optimizer.optimize(function.f, x0=[3.0, -4.0], steps=100)

    assert function.f(path[-1]) < upper_bound
    assert function.f(path[-1]) < function.f(path[0])
