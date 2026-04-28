"""Diagnostics computed along optimization trajectories."""

import numpy as np
from core.gradients import Gradient
from core.hessian import Hessian


class Diagnostics:
    """Compute diagnostic metrics along an optimization trajectory."""

    @staticmethod
    def gradient_norm(f, x):
        """Compute ||∇f(x)||."""
        grad = Gradient.get_grad(f, x)
        return np.linalg.norm(grad)

    @staticmethod
    def hessian_eigenvalues(f, x):
        """Compute eigenvalues of Hessian at x."""
        H = Hessian.get_hessian(f, x)
        eigenvalues = np.linalg.eigvalsh(H)
        return eigenvalues

    @staticmethod
    def condition_number(f, x):
        """Compute condition number of Hessian: λ_max / λ_min."""
        eigenvalues = Diagnostics.hessian_eigenvalues(f, x)

        abs_eigenvalues = np.abs(eigenvalues)
        abs_eigenvalues = abs_eigenvalues[abs_eigenvalues > 1e-10]

        if len(abs_eigenvalues) == 0:
            return np.inf

        lambda_max = np.max(abs_eigenvalues)
        lambda_min = np.min(abs_eigenvalues)

        if lambda_min < 1e-10:
            return np.inf

        return lambda_max / lambda_min

    @staticmethod
    def compute_trajectory_diagnostics(f, trajectory):
        """Compute diagnostics for each point in a trajectory.

        Args:
            f: Scalar function f(x) → R
            trajectory: array of shape (n_steps, n_dims)

        Returns:
            dict with keys:
                - 'gradient_norms': array of ||∇f(x_t)||
                - 'eigenvalues': list of eigenvalue arrays
                - 'condition_numbers': array of condition numbers
                - 'function_values': array of f(x_t)
        """
        n_steps = len(trajectory)
        grad_norms = np.zeros(n_steps)
        condition_numbers = np.zeros(n_steps)
        function_values = np.zeros(n_steps)
        eigenvalues_list = []

        for i, x in enumerate(trajectory):
            grad_norms[i] = Diagnostics.gradient_norm(f, x)
            condition_numbers[i] = Diagnostics.condition_number(f, x)
            function_values[i] = f(x)
            eigenvalues_list.append(Diagnostics.hessian_eigenvalues(f, x))

        return {
            'gradient_norms': grad_norms,
            'eigenvalues': eigenvalues_list,
            'condition_numbers': condition_numbers,
            'function_values': function_values
        }

    @staticmethod
    def final_diagnostics(f, x_final):
        """Print a summary of diagnostics at final point."""
        grad_norm = Diagnostics.gradient_norm(f, x_final)
        eigenvalues = Diagnostics.hessian_eigenvalues(f, x_final)
        cond_num = Diagnostics.condition_number(f, x_final)
        f_val = f(x_final)

        print(f"Final diagnostics:")
        print(f"  f(x) = {f_val:.6f}")
        print(f"  ||∇f(x)|| = {grad_norm:.6e}")
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Condition number: {cond_num:.2e}")
