"""Quadratic-penalty solver for equality-constrained optimization."""

import numpy as np

from core.gradients import Gradient
from dynamics.constrained.base import ConstrainedOptimizer


class PenaltyMethod(ConstrainedOptimizer):
    """Solve constrained optimization using a quadratic penalty."""

    VIOLATION_THRESHOLD = 0.1

    def __init__(self, step_size=0.01):
        """Initialize the method with a primal step size."""
        super().__init__(step_size=step_size)

    def optimize(
        self,
        f,
        g,
        x0,
        rho_init=1.0,
        steps=100,
        tol=1e-6,
        rho_increase_rate=10.0,
    ):
        """Solve constrained problem using quadratic penalty."""
        x = np.array(x0, float)
        rho = rho_init

        x_history = [x.copy()]
        rho_history = [rho]

        for _ in range(steps):
            grad_f = Gradient.get_grad(f, x)
            grad_g = Gradient.get_grad(g, x)
            constraint_val = g(x)

            # Define: f(x) + ρ(g(x))^2
            # Then:   ∇f(x) + 2 ρ g(x) ∇g(x)
            grad_penalized = grad_f + 2 * rho * constraint_val * grad_g

            if np.linalg.norm(grad_penalized) < tol and abs(constraint_val) < tol:
                break

            x = x - self.step_size * grad_penalized

            if abs(g(x)) > self.VIOLATION_THRESHOLD:
                rho = rho * rho_increase_rate

            x_history.append(x.copy())
            rho_history.append(rho)

        return x, rho, (np.array(x_history), np.array(rho_history))
