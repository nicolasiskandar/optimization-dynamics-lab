"""Ill-conditioned quadratic objective used to study anisotropic curvature."""

import numpy as np
from .base import Function2D


class IllConditioned(Function2D):
    """Quadratic function with configurable axis-wise curvature scales."""

    def __init__(self, a=1, b=100):
        """Initialize anisotropic quadratic coefficients."""
        self.a = a
        self.b = b

    def f(self, x):
        """Compute f(x, y) = a*x^2 + b*y^2."""
        return self.a * x[0]**2 + self.b * x[1]**2

    def grad(self, x):
        """Return the gradient of the ill-conditioned quadratic at x."""
        return np.array([2*self.a*x[0], 2*self.b*x[1]])

    def hessian(self, x):
        """Return the constant Hessian for the configured coefficients."""
        return np.array([[2*self.a, 0],
                         [0, 2*self.b]])
