"""Trajectory containers and execution helpers for optimization runs."""

import numpy as np
from dynamics.diagnostics import Diagnostics


class Trajectory:
    """Container for optimization trajectory with diagnostics."""

    def __init__(self, function, optimizer_name, x0, trajectory_array, diagnostics_data):
        """Initialize trajectory.

        Args:
            function: Function2D object
            optimizer_name: str, name of optimizer
            x0: initial point
            trajectory_array: np.array of shape (n_steps, n_dims)
            diagnostics_data: dict from Diagnostics.compute_trajectory_diagnostics
        """
        self.function = function
        self.optimizer_name = optimizer_name
        self.x0 = np.array(x0)
        self.trajectory = trajectory_array
        self.diagnostics = diagnostics_data

    @property
    def n_steps(self):
        """Return the number of stored iterates."""
        return len(self.trajectory)

    @property
    def x_final(self):
        """Return the last point in the trajectory."""
        return self.trajectory[-1]

    @property
    def f_initial(self):
        """Return the function value at the starting point."""
        return self.diagnostics['function_values'][0]

    @property
    def f_final(self):
        """Return the function value at the final point."""
        return self.diagnostics['function_values'][-1]

    @property
    def grad_norm_final(self):
        """Return the gradient norm at the final point."""
        return self.diagnostics['gradient_norms'][-1]

    def summary(self):
        """Return string summary of trajectory."""
        return (
            f"Optimizer: {self.optimizer_name}\n"
            f"  Starting point: {self.x0}\n"
            f"  Final point: {self.x_final}\n"
            f"  Steps: {self.n_steps}\n"
            f"  f(x0) → f(xf): {self.f_initial:.6f} → {self.f_final:.6f}\n"
            f"  ||∇f(xf)||: {self.grad_norm_final:.6e}\n"
        )


class TrajectoryRunner:
    """Orchestrate optimization runs and collect trajectories."""

    @staticmethod
    def run(optimizer, function, x0, steps=100, tol=1e-6):
        """Run optimizer and collect trajectory with diagnostics.

        Args:
            optimizer: Optimizer object (GradientDescent, MomentumGD, Newton, etc.)
            function: Function2D object
            x0: initial point
            steps: max number of steps
            tol: convergence tolerance on gradient norm

        Returns:
            Trajectory object
        """
        trajectory_array = optimizer.optimize(
            function.f, x0, steps=steps, tol=tol)
        optimizer_name = optimizer.__class__.__name__

        diagnostics_data = Diagnostics.compute_trajectory_diagnostics(
            function.f, trajectory_array
        )

        trajectory = Trajectory(
            function=function,
            optimizer_name=optimizer_name,
            x0=x0,
            trajectory_array=trajectory_array,
            diagnostics_data=diagnostics_data
        )

        return trajectory

    @staticmethod
    def run_comparison(optimizers, function, x0, steps=100, tol=1e-6):
        """Run multiple optimizers from same starting point.

        Args:
            optimizers: list of optimizer objects
            function: Function2D object
            x0: initial point
            steps: max number of steps
            tol: convergence tolerance

        Returns:
            list of Trajectory objects
        """
        trajectories = []
        for opt in optimizers:
            traj = TrajectoryRunner.run(
                opt, function, x0, steps=steps, tol=tol)
            trajectories.append(traj)

        return trajectories

    @staticmethod
    def run_multistart(optimizer, function, x0_list, steps=100, tol=1e-6):
        """Run optimizer from multiple starting points.

        Args:
            optimizer: single optimizer object
            function: Function2D object
            x0_list: list of starting points
            steps: max number of steps
            tol: convergence tolerance

        Returns:
            list of Trajectory objects
        """
        trajectories = []
        for x0 in x0_list:
            traj = TrajectoryRunner.run(
                optimizer, function, x0, steps=steps, tol=tol)
            trajectories.append(traj)

        return trajectories
