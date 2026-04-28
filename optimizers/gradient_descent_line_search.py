"""Gradient descent optimizer with adaptive line-search step selection."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer
from optimizers.line_search import LineSearch


class GradientDescentWithLineSearch(Optimizer):
    """Gradient Descent with adaptive step size selection."""

    def __init__(self, line_search_method="backtracking", initial_step=1.0):
        """Initialize."""
        self.line_search_method = line_search_method
        self.initial_step = initial_step

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Optimize with line search."""
        x = np.array(x0, float)
        history = [x.copy()]

        for _ in range(steps):
            grad = Gradient.get_grad(f, x)

            if np.linalg.norm(grad) < tol:
                break

            direction = -grad

            if self.line_search_method == "backtracking":
                step_size = LineSearch.backtracking_line_search(
                    f, x, direction, self.initial_step
                )
            elif self.line_search_method == "golden_section":
                step_size = LineSearch.golden_section_search(
                    f, x, direction, self.initial_step
                )
            else:
                step_size = self.initial_step

            x = x + step_size * direction
            history.append(x.copy())

        return np.array(history)
