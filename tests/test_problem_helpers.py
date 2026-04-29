"""Tests for constrained-problem helper objects."""

import numpy as np

from dynamics.constrained.problems import (
    ConstrainedOptimizationProblem,
    create_circle_constraint_problem,
    create_ellipse_constraint_problem,
)


def test_constrained_problem_feasibility_and_violation_helpers():
    """Helper methods should report feasibility against the stored constraint."""
    problem = ConstrainedOptimizationProblem(
        f=lambda x: x[0] ** 2 + x[1] ** 2,
        g=lambda x: x[0] + x[1] - 1.0,
        name="toy",
    )

    feasible = np.array([0.25, 0.75])
    infeasible = np.array([0.25, 0.5])

    assert problem.is_feasible(feasible)
    assert not problem.is_feasible(infeasible)
    assert problem.constraint_violation(infeasible) == 0.25


def test_circle_constraint_problem_matches_documented_solution():
    """The circle helper should encode x + y = 2 and objective x^2 + y^2."""
    problem = create_circle_constraint_problem()
    optimum = np.array([1.0, 1.0])

    assert problem.name == "Circle with linear constraint"
    assert problem.is_feasible(optimum)
    assert problem.f(optimum) == 2.0


def test_ellipse_constraint_problem_matches_documented_geometry():
    """The ellipse helper should treat (2, 0) as a feasible zero-loss point."""
    problem = create_ellipse_constraint_problem()
    optimum = np.array([2.0, 0.0])

    assert problem.name == "Ellipse constraint"
    assert problem.is_feasible(optimum)
    assert problem.f(optimum) == 0.0
