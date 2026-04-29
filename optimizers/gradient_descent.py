"""Gradient descent optimizer with fixed step size."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer


class GradientDescent(Optimizer):
    """Gradient descent with a fixed learning rate."""

    def __init__(self, step_size=0.01):
        """Initialize gradient descent with a fixed step size.

        Args:
            step_size: Learning rate controlling the magnitude of each update.
                       Smaller values converge more slowly but are more stable.
                       Defaults to 0.01.
        """
        self.step_size = step_size

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Run gradient descent and return the optimization trajectory.

        Args:
            f: Scalar objective function f(x) -> R.
            x0: Initial point as a list or array of coordinates.
            steps: Maximum number of optimization iterations. Defaults to 100.
            tol: Convergence tolerance on the gradient norm. The algorithm
                 stops early if ||grad|| < tol. Defaults to 1e-6.

        Returns:
            numpy.ndarray of shape (n_steps, n_dims) containing the sequence
            of iterates, starting from x0.
        """
        x = np.array(x0, float)
        history = [x.copy()]

        for _ in range(steps):
            grad = Gradient.get_grad(f, x)

            if np.linalg.norm(grad) < tol:
                break

            x = x - self.step_size * grad
            history.append(x.copy())

        return np.array(history)
