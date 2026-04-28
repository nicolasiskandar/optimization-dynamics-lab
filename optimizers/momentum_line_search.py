"""Momentum optimizer with adaptive line-search step selection."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer
from optimizers.line_search import LineSearch


class MomentumWithLineSearch(Optimizer):
    """Momentum GD with adaptive step size."""

    def __init__(self, beta=0.9, line_search_method="backtracking", initial_step=1.0):
        """Initialize."""
        self.beta = beta
        self.line_search_method = line_search_method
        self.initial_step = initial_step

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Optimize with momentum and line search."""
        x = np.array(x0, float)
        v = np.zeros_like(x)
        history = [x.copy()]

        for _ in range(steps):
            grad = Gradient.get_grad(f, x)

            if np.linalg.norm(grad) < tol:
                break

            v = self.beta * v + grad
            direction = -v / (np.linalg.norm(v) + 1e-10)

            if self.line_search_method == "backtracking":
                step_size = LineSearch.backtracking_line_search(
                    f, x, direction, self.initial_step
                )
            else:
                step_size = LineSearch.golden_section_search(
                    f, x, direction, self.initial_step
                )

            x = x + step_size * direction
            history.append(x.copy())

        return np.array(history)
