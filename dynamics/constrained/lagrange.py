"""Lagrange-multiplier solver for equality-constrained optimization."""

import numpy as np

from core.gradients import Gradient
from dynamics.constrained.base import ConstrainedOptimizer


class LagrangeMultiplierMethod(ConstrainedOptimizer):
    """Solve constrained optimization using Lagrange multipliers."""

    DEFAULT_DUAL_STEP_SIZE = 0.1
    FEASIBILITY_STEP_SIZE = 0.01
    FEASIBILITY_TOL = 1e-4

    def __init__(self, step_size=0.01):
        """Initialize the method with a primal step size."""
        super().__init__(step_size=step_size)

    def lagrangian_gradient(self, f, g, x, lam):
        """Compute ∇x L(x, λ) = ∇f(x) + λ∇g(x)."""
        grad_f = Gradient.get_grad(f, x)
        grad_g = Gradient.get_grad(g, x)

        # Standard Lagrangian: L(x, λ) = f(x) + λg(x)
        # Lagrangian Gradient: ∇L = ∇f + λ ∇g
        return grad_f + lam * grad_g

    def optimize(self, f, g, x0, lambda0=1.0, steps=100, tol=1e-6):
        """Optimize with Lagrange multipliers."""
        x = np.array(x0, float)
        lam = lambda0

        x_history = [x.copy()]
        lambda_history = [lam]

        for _ in range(steps):
            constraint_val = g(x)

            if abs(constraint_val) > self.FEASIBILITY_TOL:
                grad_g = Gradient.get_grad(g, x)
                norm_grad_g = np.linalg.norm(grad_g) + 1e-10
                direction = grad_g / norm_grad_g
                # direction of the steepest increase, when g(x) > 0 we should move in the opp direction
                # to make g(x) = 0, simillarly when g(x) < 0
                # so constraint_val is used for the sign, and the more it's magnitude increases the harder the push towards g(x)=0
                # multiply by FEASIBILITY_STEP_SIZE tp prevents overshooting
                # In Short: nove x toward the constraint surface using a small, damped step in the direction normal
                # to g(x)=0, with the sign and magnitude of g(x) controlling how far and which side correction is applied.
                x = x - self.FEASIBILITY_STEP_SIZE * constraint_val * direction

            grad_lag = self.lagrangian_gradient(f, g, x, lam)

            # Stop if The Lagrangian gradient is almost zero (∇f ≈ λ ∇g), checked the norm as it's the same condition (easier to compute)
            # and constraint violation is small (g(x) ≈ 0)
            if np.linalg.norm(grad_lag) < tol and abs(g(x)) < tol:
                break

            # Use the Lagrangian to combine objective minimization with constraint enforcement via gradient balance,
            # so optimality occurs when both objective and constraint forces cancel
            x = x - self.step_size * grad_lag

            # Dual ascent update for λ
            # drive the system back to the constraint surface from either side, goal: g(x) = 0
            # λ determines how strongly you push along ∇g(x), it's the sensitivity (lagrange multiplier) not the penalty
            # Push in the opposite direction
            # λ updates in the direction of constraint violation so it corrects errors from both sides of g(x)=0, not just penalizing magnitude
            lam = lam + self.DEFAULT_DUAL_STEP_SIZE * g(x)

            x_history.append(x.copy())
            lambda_history.append(lam)

        return x, lam, (np.array(x_history), np.array(lambda_history))
