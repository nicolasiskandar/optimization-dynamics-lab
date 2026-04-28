"""Shared optimizer interface."""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class implemented by all unconstrained optimizers."""

    @abstractmethod
    def optimize(self, f, x0, steps=100, tol=1e-6):
        """Run optimization and return the optimization trajectory."""
