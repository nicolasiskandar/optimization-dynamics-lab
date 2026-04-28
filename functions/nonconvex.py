"""Non-convex 2D objective combining sinusoidal and quadratic terms."""

import numpy as np
from .base import Function2D


class NonConvex(Function2D):
    """Smooth non-convex landscape with local and global minima."""

    def f(self, x):
        """Compute a non-convex objective with sinusoidal curvature."""
        return np.sin(x[0]) + x[1]**2 + 0.1*x[0]**2

    def grad(self, x):
        """Return the gradient of the non-convex objective at x."""
        return np.array([
            np.cos(x[0]) + 0.2*x[0],
            2*x[1]
        ])

    def hessian(self, x):
        """Return the Hessian matrix at x."""
        return np.array([
            [-np.sin(x[0]) + 0.2, 0],
            [0, 2]
        ])
