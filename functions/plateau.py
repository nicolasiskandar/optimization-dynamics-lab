"""Plateau-like 2D objective based on hyperbolic tangents."""

import numpy as np
from .base import Function2D


class Plateau(Function2D):
    """Plateau objective with saturating tanh terms."""

    def f(self, x):
        """Compute tanh(0.1*x) + tanh(0.1*y)."""
        return np.tanh(0.1 * x[0]) + np.tanh(0.1 * x[1])

    def grad(self, x):
        """Return gradient of plateau objective at x."""
        return np.array([
            0.1 * (1 - np.tanh(0.1 * x[0]) ** 2),
            0.1 * (1 - np.tanh(0.1 * x[1]) ** 2)
        ])

    def hessian(self, x):
        """Return Hessian matrix at x."""
        t0 = np.tanh(0.1 * x[0])
        t1 = np.tanh(0.1 * x[1])

        return np.array([
            [-0.02 * t0 * (1 - t0 ** 2), 0.0],
            [0.0, -0.02 * t1 * (1 - t1 ** 2)]
        ])
