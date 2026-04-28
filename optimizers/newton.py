"""Newton optimizer based on gradient and Hessian information."""

import numpy as np
from core.gradients import Gradient
from core.hessian import Hessian
from optimizers.base import Optimizer


class Newton(Optimizer):
    """Second-order optimizer using local quadratic approximation."""

    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Run Newton's method and return the optimization trajectory."""
        x = np.array(x0, float)
        history = [x.copy()]

        for _ in range(steps):
            grad = Gradient.get_grad(f, x)

            if np.linalg.norm(grad) < tol:
                break

            H = Hessian.get_hessian(f, x)

            try:
                step = np.linalg.solve(H, grad)

                # step = np.linalg.inv(H) @ grad
                # This took approx twice the nb of steps in non-convex function when testing (38 compared to 20)
                # since inverting a matrix introduces more floating point rouding errors that accumulate
                # whereas solve "normalizes" the space, so solve() is numerically more stable
                # than inv() @ grad, especially for ill-conditioned Hessians (ex ones shaped like canyon)
            except np.linalg.LinAlgError:
                break

            x = x - step
            history.append(x.copy())

        return np.array(history)
