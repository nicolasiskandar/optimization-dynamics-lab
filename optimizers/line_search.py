"""Line-search routines for selecting step sizes during optimization."""

import numpy as np
from core.gradients import Gradient


class LineSearch:
    """Line search methods for step size selection."""

    @staticmethod
    def backtracking_line_search(
        f, x, direction, initial_step=1.0, c=0.5, rho=0.5, max_iters=50
    ):
        """Backtracking line search (Armijo condition)."""
        x = np.array(x)
        direction = np.array(direction)

        f_x = f(x)
        grad = Gradient.get_grad(f, x)
        directional_deriv = np.dot(grad, direction)

        step_size = initial_step

        for _ in range(max_iters):
            f_new = f(x + step_size * direction)

            # Armijo condition
            if f_new <= f_x + c * step_size * directional_deriv:
                return step_size

            step_size *= rho

        return step_size


@staticmethod
def golden_section_search(
    f, x, direction, initial_step=1.0, tol=1e-4, max_iters=50
):
    """Golden section line search."""
    x = np.array(x)
    direction = np.array(direction)

    golden_ratio = (3 - np.sqrt(5)) / 2  # ~= 0.381966

    a = 0.0
    b = initial_step

    x1 = a + golden_ratio * (b - a)
    x2 = b - golden_ratio * (b - a)

    f1 = f(x + x1 * direction)
    f2 = f(x + x2 * direction)

    for _ in range(max_iters):
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + golden_ratio * (b - a)
            f1 = f(x + x1 * direction)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - golden_ratio * (b - a)
            f2 = f(x + x2 * direction)

    return (a + b) / 2
