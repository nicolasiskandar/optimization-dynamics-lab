"""Smoke tests for fixed-step gradient descent across sample functions."""

from functions.quadratic import Quadratic
from functions.ill_conditioned import IllConditioned
from functions.nonconvex import NonConvex
from functions.saddle import Saddle
from optimizers.gradient_descent import GradientDescent

optimizer = GradientDescent(step_size=0.1)

test_functions = [Quadratic, IllConditioned, NonConvex, Saddle]

for func in test_functions:
    path = optimizer.optimize(func().f, x0=[3, -4])
    print(f'{func.__name__}: ({len(path)} steps)\n{path}')
