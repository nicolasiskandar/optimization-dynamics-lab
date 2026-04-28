"""Gradient descent optimizer with fixed step size."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer


class GradientDescent(Optimizer):
    """Gradient descent with fixed step size."""

    def __init__(self, step_size=0.01):
        """Set the fixed step size used by gradient descent."""
        self.step_size = step_size

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Run gradient descent and return the optimization trajectory."""
        x = np.array(x0, float)
        history = [x.copy()]

        for _ in range(steps):
            grad = Gradient.get_grad(f, x)

            if np.linalg.norm(grad) < tol:
                break

            x = x - self.step_size * grad
            history.append(x.copy())

        return np.array(history)
