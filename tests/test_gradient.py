"""Sanity checks for numerical gradient approximation."""

from core.gradients import Gradient
import numpy as np

def f(x):
    """Quadratic test function used for gradient validation."""
    return x[0]**2 + x[1]**2

def true_grad(x):
    """Analytic gradient for the quadratic test function."""
    return np.array([2*x[0], 2*x[1]])

x = np.array([3.0, -4.0])

num_grad = Gradient.get_grad(f, x)
ana_grad = true_grad(x)

print("numeric:", num_grad)
print("analytic:", ana_grad)
print("error:", np.linalg.norm(num_grad - ana_grad))

assert np.linalg.norm(num_grad - ana_grad) < 1e-6
