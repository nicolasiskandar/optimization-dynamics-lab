"""Saddle-point objective with one ascent and one descent direction."""

import numpy as np
from .base import Function2D


class Saddle(Function2D):
    """Indefinite quadratic surface used to expose saddle behavior."""

    def f(self, x):
        """Compute the saddle objective f(x, y) = x^2 - y^2."""
        return x[0]**2 - x[1]**2

    def grad(self, x):
        """Return the gradient of the saddle objective at x."""
        return np.array([2*x[0], -2*x[1]])

    def hessian(self, x):
        """Return the constant Hessian of the saddle objective."""
        return np.array([[2, 0],
                         [0, -2]])
