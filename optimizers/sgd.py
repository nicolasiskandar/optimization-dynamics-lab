"""Mini-batch SGD optimizer with additive Gaussian noise."""

import numpy as np
from core.gradients import Gradient
from optimizers.base import Optimizer


class SGD(Optimizer):
    """
    Mini-batch stochastic gradient descent with additive Gaussian noise.

    Uses:
        1. Mini-batch sampling
        2. Additive isotropic Gaussian noise on gradient estimates

    Update rule:

        x_{t+1} = x_t - step_size * (batch_grad + noise)

    where:
        batch_grad = mean(∇f_i(x)) over sampled indices
        noise ~ N(0, noise_std^2 I)
    """

    def __init__(
        self,
        step_size=0.01,
        batch_size=10,
        noise_std=0.01,
        seed=None
    ):
        """
        Initialize optimizer hyperparameters.

        Args:
            step_size: learning rate
            batch_size: number of component functions per iteration
            noise_std: standard deviation of Gaussian noise added to gradient
            seed: random seed for reproducibility
        """
        self.step_size = step_size
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.seed = seed

    def optimize(self, functions, x0, steps=100, tol=1e-6):
        """
        Run mini-batch SGD with additive Gaussian noise.

        Args:
            functions: list of component functions [f1, f2, ..., fN]
            x0: initial parameter vector
            steps: maximum number of iterations
            tol: stopping threshold on batch gradient norm

        Returns:
            np.ndarray: optimization trajectory
        """
        x = np.array(x0, dtype=float)
        history = [x.copy()]

        rng = np.random.default_rng(self.seed)
        n_functions = len(functions)

        if self.batch_size > n_functions:
            raise ValueError(
                "batch_size cannot be larger than number of functions")

        for _ in range(steps):
            batch_indices = rng.choice(
                n_functions,
                size=self.batch_size,
                replace=False
            )

            batch_grads = []
            for idx in batch_indices:
                fi = functions[idx]
                grad = Gradient.get_grad(fi, x)
                batch_grads.append(grad)

            batch_grad = np.mean(batch_grads, axis=0)

            noise = rng.normal(loc=0.0, scale=self.noise_std, size=x.shape)

            if np.linalg.norm(batch_grad) < tol:
                break

            x = x - self.step_size * (batch_grad + noise)

            history.append(x.copy())

        return np.array(history)
