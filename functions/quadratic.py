"""Well-conditioned convex quadratic objective."""

import numpy as np
from .base import Function2D


class Quadratic(Function2D):
    """Standard isotropic quadratic bowl in two dimensions."""

    def f(self, x):
        """Compute f(x, y) = x^2 + y^2."""
        return x[0]**2 + x[1]**2

    def grad(self, x):
        """Return the gradient of the quadratic at x."""
        return np.array([2*x[0], 2*x[1]])

    def hessian(self, x):
        """Return the constant Hessian of the quadratic."""
        return np.array([[2, 0],
                         [0, 2]])
