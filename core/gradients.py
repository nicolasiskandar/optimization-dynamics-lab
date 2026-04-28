"""Numerical gradient approximations for scalar objective functions."""

import numpy as np


class Gradient:
    """Finite-difference gradient utilities."""

    @staticmethod
    def get_grad(f, x, eps=1e-6):
        """Approximate the gradient using central finite differences."""
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_forward = x.copy()
            x_backward = x.copy()

            x_forward[i] += eps
            x_backward[i] -= eps

            grad[i] = (f(x_forward) - f(x_backward)) / (2 * eps)

        return grad
