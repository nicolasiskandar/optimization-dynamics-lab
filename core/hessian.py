"""Numerical Hessian approximations for scalar objective functions."""

import numpy as np
from .gradients import Gradient


class Hessian:
    """Finite-difference Hessian utilities."""

    @staticmethod
    def get_hessian(f, x, eps=1e-6):
        """Approximate the Hessian using finite differences of gradients."""
        x = np.array(x, dtype=float)
        n = len(x)
        H = np.zeros((n, n))

        for i in range(n):
            x_forward = x.copy()
            x_backward = x.copy()

            x_forward[i] += eps
            x_backward[i] -= eps

            grad_forward = Gradient.get_grad(f, x_forward)
            grad_backward = Gradient.get_grad(f, x_backward)

            H[:, i] = (grad_forward - grad_backward) / (2 * eps)

        return H
