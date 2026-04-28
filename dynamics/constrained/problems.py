"""Canonical constrained optimization problem definitions used in demos/tests."""


class ConstrainedOptimizationProblem:
    """Container for an equality-constrained optimization problem."""

    def __init__(self, f, g, name=""):
        """Initialize."""
        self.f = f
        self.g = g
        self.name = name

    def is_feasible(self, x, tol=1e-4):
        """Check if x satisfies g(x)=0 within tolerance."""
        return abs(self.g(x)) < tol

    def constraint_violation(self, x):
        """Return absolute constraint violation |g(x)|."""
        return abs(self.g(x))


def create_circle_constraint_problem():
    """Minimize x^2 + y^2 subject to x + y = 2."""

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    def g(x):
        return x[0] + x[1] - 2.0

    return ConstrainedOptimizationProblem(f, g, "Circle with linear constraint")


def create_ellipse_constraint_problem():
    """Minimize (x-2)^2 + y^2 subject to x^2/4 + y^2 = 1."""

    def f(x):
        return (x[0] - 2.0) ** 2 + x[1] ** 2

    def g(x):
        return x[0] ** 2 / 4.0 + x[1] ** 2 - 1.0

    return ConstrainedOptimizationProblem(f, g, "Ellipse constraint")
