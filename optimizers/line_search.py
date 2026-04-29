"""Line-search routines for selecting step sizes during optimization."""

import numpy as np
from core.gradients import Gradient


class LineSearch:
    """Line search methods for adaptive step size selection.

    Provides static methods that, given a current point and a search direction,
    determine an appropriate step size along that direction.
    """

    @staticmethod
    def backtracking_line_search(
        f, x, direction, initial_step=1.0, c=0.5, rho=0.5, max_iters=50
    ):
        """Backtracking line search satisfying the Armijo condition.

        Starting from initial_step, the step size is repeatedly reduced by
        a factor of rho until the Armijo sufficient decrease condition holds:
            f(x + step * direction) <= f(x) + c * step * dot(grad, direction)

        Args:
            f: Scalar objective function f(x) -> R.
            x: Current point as a list or array of coordinates.
            direction: Search direction vector (typically -grad or -velocity).
            initial_step: Starting step size to backtrack from. Defaults to 1.0.
            c: Armijo condition parameter controlling the required decrease.
               Smaller values accept larger steps. Defaults to 0.5.
            rho: Backtracking shrinkage factor in (0, 1). Each failed step
                 is multiplied by rho. Defaults to 0.5.
            max_iters: Maximum number of backtracking reductions. Defaults to 50.

        Returns:
            float: Step size that satisfies (or comes closest to satisfying)
                   the Armijo condition.
        """
        x = np.array(x)
        direction = np.array(direction)

        f_x = f(x)
        grad = Gradient.get_grad(f, x)
        directional_deriv = np.dot(grad, direction)

        step_size = initial_step

        for _ in range(max_iters):
            f_new = f(x + step_size * direction)

            if f_new <= f_x + c * step_size * directional_deriv:
                return step_size

            step_size *= rho

        return step_size

    @staticmethod
    def golden_section_search(
        f, x, direction, initial_step=1.0, tol=1e-4, max_iters=50
    ):
        """Golden section line search for univariate function minimization.

        Treats phi(alpha) = f(x + alpha * direction) as a 1D function and
        finds its minimum within [0, initial_step] using the golden section
        method, which progressively narrows the bracketing interval.

        Args:
            f: Scalar objective function f(x) -> R.
            x: Current point as a list or array of coordinates.
            direction: Search direction vector.
            initial_step: Upper bound of the search interval [0, initial_step].
                          Defaults to 1.0.
            tol: Convergence tolerance on the interval width. The search stops
                 when b - a < tol. Defaults to 1e-4.
            max_iters: Maximum number of iterations. Defaults to 50.

        Returns:
            float: Step size alpha that approximately minimizes f along the
                   given direction within the specified tolerance.
        """
        x = np.array(x)
        direction = np.array(direction)

        golden_ratio = (3 - np.sqrt(5)) / 2

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
