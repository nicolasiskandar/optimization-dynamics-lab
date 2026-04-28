"""Base abstractions for constrained optimization methods."""


class ConstrainedOptimizer:
    """Shared base class for constrained optimization methods."""

    def __init__(self, step_size=0.01):
        """Store the primal update step size."""
        self.step_size = step_size
