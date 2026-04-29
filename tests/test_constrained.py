"""Tests for constrained optimization methods."""

import numpy as np

from dynamics.constrained import (
    ConstrainedOptimizer,
    LagrangeMultiplierMethod,
    PenaltyMethod,
    create_circle_constraint_problem,
    create_ellipse_constraint_problem,
)


def test_lagrange_method_improves_objective_and_feasibility():
    """The Lagrange multiplier solver should move toward the constrained optimum."""
    problem = create_circle_constraint_problem()
    optimizer = LagrangeMultiplierMethod(step_size=0.01)
    x0 = np.array([1.5, 1.5])

    x_opt, lambda_opt, history = optimizer.optimize(
        problem.f,
        problem.g,
        x0,
        lambda0=1.0,
        steps=300,
    )

    x_history, lambda_history = history

    assert isinstance(optimizer, ConstrainedOptimizer)
    assert problem.f(x_opt) < problem.f(x0)
    assert abs(problem.g(x_opt)) < 1e-2
    assert x_history.shape == (301, 2)
    assert lambda_history.shape == (301,)
    assert lambda_opt == lambda_history[-1]


def test_penalty_method_reduces_constraint_violation_and_increases_rho():
    """The penalty method should tighten feasibility by raising the penalty."""
    problem = create_ellipse_constraint_problem()
    optimizer = PenaltyMethod(step_size=0.01)
    x0 = np.array([1.5, 0.5])

    x_opt, rho_final, history = optimizer.optimize(
        problem.f,
        problem.g,
        x0,
        rho_init=1.0,
        steps=200,
        rho_increase_rate=5.0,
    )

    x_history, rho_history = history

    assert isinstance(optimizer, ConstrainedOptimizer)
    assert abs(problem.g(x_opt)) < abs(problem.g(x0))
    assert abs(problem.g(x_opt)) < 1e-4
    assert rho_final > 1.0
    assert x_history.shape == (201, 2)
    assert rho_history.shape == (201,)
