"""Base abstractions for two-dimensional objective functions."""

from abc import ABC, abstractmethod


class Function2D(ABC):
    """Interface for scalar functions f: R^2 -> R with first/second derivatives."""

    @abstractmethod
    def f(self, x):
        """Evaluate the scalar function at point x."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x):
        """Return the gradient vector at point x."""
        raise NotImplementedError

    @abstractmethod
    def hessian(self, x):
        """Return the Hessian matrix at point x."""
        raise NotImplementedError
