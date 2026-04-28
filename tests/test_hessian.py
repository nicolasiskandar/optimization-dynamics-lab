"""Sanity checks for numerical Hessian approximation."""

from core.hessian import Hessian
import numpy as np


def f(x):
    """Quadratic test function used for Hessian validation."""
    return x[0]**2 + x[1]**2


def true_hessian(_x):
    """Analytic Hessian for the quadratic test function."""
    return np.array([[2.0, 0.0], [0.0, 2.0]])


x = np.array([3.0, -4.0])

num_hessian = Hessian.get_hessian(f, x)
ana_hessian = true_hessian(x)

print("numeric hessian:\n", num_hessian)
print("analytic hessian:\n", ana_hessian)
print("error:", np.linalg.norm(num_hessian - ana_hessian))

assert np.linalg.norm(num_hessian - ana_hessian) < 5e-3
