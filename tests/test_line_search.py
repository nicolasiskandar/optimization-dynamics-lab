"""Behavior checks for line-search-enabled optimizers."""

from functions.quadratic import Quadratic
from optimizers import GradientDescentWithLineSearch, MomentumWithLineSearch


quadratic = Quadratic()
x0 = [3, -4]

optimizers = [
    GradientDescentWithLineSearch(line_search_method="backtracking", initial_step=1.0),
    GradientDescentWithLineSearch(line_search_method="golden_section", initial_step=1.0),
    MomentumWithLineSearch(beta=0.9, line_search_method="backtracking", initial_step=1.0),
    MomentumWithLineSearch(beta=0.9, line_search_method="golden_section", initial_step=1.0),
]

for optimizer in optimizers:
    path = optimizer.optimize(quadratic.f, x0=x0, steps=100)
    initial_value = quadratic.f(path[0])
    final_value = quadratic.f(path[-1])

    print(f"{optimizer.__class__.__name__} ({optimizer.line_search_method}):")
    print(f"  steps: {len(path)}")
    print(f"  f(x0): {initial_value:.6f}")
    print(f"  f(xf): {final_value:.6f}")

    assert final_value <= initial_value

    if optimizer.__class__.__name__ == "GradientDescentWithLineSearch":
        assert final_value < 1e-4
    else:
        # Momentum with golden section can converge slower with this setup.
        assert final_value < 0.1 * initial_value
