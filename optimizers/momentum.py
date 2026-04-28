"""Momentum-accelerated gradient descent optimizer."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer


class MomentumGD(Optimizer):
    """Gradient descent with a velocity accumulator."""

    def __init__(self, step_size=0.01, beta=0.9):
        """Set hyperparameters for momentum gradient descent."""
        self.step_size = step_size
        self.beta = beta

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Run momentum gradient descent and return the trajectory."""
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
