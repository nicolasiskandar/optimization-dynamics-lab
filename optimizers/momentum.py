"""Momentum-accelerated gradient descent optimizer."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer


class MomentumGD(Optimizer):
    """Gradient descent with a velocity accumulator."""

    def __init__(self, step_size=0.01, beta=0.9):
        """Initialize momentum gradient descent.

        Args:
            step_size: Learning rate controlling the magnitude of each update.
                       Defaults to 0.01.
            beta: Momentum decay factor in [0, 1). Higher values retain more
                  historical gradient information. Defaults to 0.9.
        """
        self.step_size = step_size
        self.beta = beta

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Run momentum gradient descent and return the trajectory.

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
        v = np.zeros_like(x)

        history = [x.copy()]

        for _ in range(steps):
            grad = Gradient.get_grad(f, x)

            if np.linalg.norm(grad) < tol:
                break

            # multiplying grad by (1-beta) turns momentum into a normalized moving avg
            v = self.beta * v + grad
            x = x - self.step_size * v

            history.append(x.copy())

        return np.array(history)
