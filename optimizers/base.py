"""Shared optimizer interface."""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Abstract base class implemented by all unconstrained optimizers.

    All subclasses must implement `optimize()` which runs the optimization
    algorithm and returns the full trajectory of iterates.
    """

    @abstractmethod
    def optimize(self, objective, x0, steps=100, tol=1e-6):
        """Run optimization and return the optimization trajectory.

        Args:
            objective: Objective payload consumed by the optimizer. For most
                       optimizers this is a scalar callable f(x) -> R. Some
                       implementations may accept a richer optimizer-specific
                       structure.
            x0: Initial point as a list or array of coordinates.
            steps: Maximum number of optimization iterations. Defaults to 100.
            tol: Convergence tolerance on the gradient norm. The algorithm
                 stops early if ||grad|| < tol. Defaults to 1e-6.

        Returns:
            numpy.ndarray of shape (n_steps, n_dims) containing the sequence
            of iterates, starting from x0.
        """
