"""Integration checks for constrained optimization methods."""

import numpy as np

from dynamics.constrained import (
    ConstrainedOptimizer,
    LagrangeMultiplierMethod,
    PenaltyMethod,
    create_circle_constraint_problem,
    create_ellipse_constraint_problem,
)


lagrange = LagrangeMultiplierMethod(step_size=0.01)
penalty = PenaltyMethod(step_size=0.01)

assert isinstance(lagrange, ConstrainedOptimizer)
assert isinstance(penalty, ConstrainedOptimizer)

circle_problem = create_circle_constraint_problem()
x0_circle = np.array([1.5, 1.5])
initial_circle_value = circle_problem.f(x0_circle)

x_opt_lagrange, lambda_opt, history_lagrange = lagrange.optimize(
    circle_problem.f,
    circle_problem.g,
    x0_circle,
    lambda0=1.0,
    steps=300,
)

final_circle_value = circle_problem.f(x_opt_lagrange)
final_circle_violation = abs(circle_problem.g(x_opt_lagrange))

print("Lagrange method:")
print(f"  f(x0): {initial_circle_value:.6f}")
print(f"  f(x*): {final_circle_value:.6f}")
print(f"  |g(x*)|: {final_circle_violation:.2e}")
print(f"  lambda*: {lambda_opt:.6f}")

assert final_circle_value < initial_circle_value
assert final_circle_violation < 1e-2
assert history_lagrange[0].shape[0] == history_lagrange[1].shape[0]

ellipse_problem = create_ellipse_constraint_problem()
x0_ellipse = np.array([1.5, 0.5])
initial_ellipse_violation = abs(ellipse_problem.g(x0_ellipse))

x_opt_penalty, rho_final, history_penalty = penalty.optimize(
    ellipse_problem.f,
    ellipse_problem.g,
    x0_ellipse,
    rho_init=1.0,
    steps=200,
    rho_increase_rate=5.0,
)

final_ellipse_violation = abs(ellipse_problem.g(x_opt_penalty))

print("Penalty method:")
print(f"  |g(x0)|: {initial_ellipse_violation:.6f}")
print(f"  |g(x*)|: {final_ellipse_violation:.6f}")
print(f"  rho_final: {rho_final:.6f}")

assert final_ellipse_violation < initial_ellipse_violation
assert rho_final >= 1.0
assert history_penalty[0].shape[0] == history_penalty[1].shape[0]
