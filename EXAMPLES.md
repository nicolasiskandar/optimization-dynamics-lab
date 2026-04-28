# Usage Examples

## Basic optimization run

```python
from functions import Quadratic
from optimizers import GradientDescent
from dynamics import TrajectoryRunner

function = Quadratic()
optimizer = GradientDescent(step_size=0.1)

traj = TrajectoryRunner.run(
    optimizer,
    function,
    x0=[3.0, -4.0],
    steps=100,
    tol=1e-6,
)

print(traj.summary())
```

## Compare multiple optimizers

```python
from functions import NonConvex
from optimizers import GradientDescent, MomentumGD, Newton
from dynamics import TrajectoryRunner

function = NonConvex()
optimizers = [
    GradientDescent(step_size=0.1),
    MomentumGD(step_size=0.1, beta=0.9),
    Newton(),
]

trajectories = TrajectoryRunner.run_comparison(
    optimizers,
    function,
    x0=[1.0, 2.0],
    steps=200,
    tol=1e-6,
)

for traj in trajectories:
    print(traj.optimizer_name, traj.n_steps, traj.f_final, traj.grad_norm_final)
```

## Fixed step vs line search

```python
from functions import IllConditioned
from optimizers import GradientDescent, GradientDescentWithLineSearch
from dynamics import TrajectoryRunner

function = IllConditioned(a=1, b=100)
x0 = [4.0, -3.0]

gd_fixed = GradientDescent(step_size=0.05)
gd_backtracking = GradientDescentWithLineSearch(
    line_search_method="backtracking",
    initial_step=1.0,
)
gd_golden = GradientDescentWithLineSearch(
    line_search_method="golden_section",
    initial_step=1.0,
)

runs = TrajectoryRunner.run_comparison(
    [gd_fixed, gd_backtracking, gd_golden],
    function,
    x0,
    steps=300,
    tol=1e-8,
)

for traj in runs:
    print(traj.optimizer_name, traj.n_steps, traj.f_final)
```

## Momentum with line search

```python
from functions import Quadratic
from optimizers import MomentumWithLineSearch
from dynamics import TrajectoryRunner

function = Quadratic()
optimizer = MomentumWithLineSearch(
    beta=0.9,
    line_search_method="golden_section",
    initial_step=1.0,
)

traj = TrajectoryRunner.run(optimizer, function, x0=[3.0, -4.0], steps=100)
print(traj.n_steps, traj.f_final)
```

## Multi-start analysis

```python
from functions import NonConvex
from optimizers import GradientDescent
from experiments.compare_optimizers import MultiStartAnalyzer

function = NonConvex()
optimizer = GradientDescent(step_size=0.1)

starts = MultiStartAnalyzer.generate_grid_starts(
    x_range=(-3, 3),
    y_range=(-2, 4),
    n_per_axis=5,
)

trajectories = MultiStartAnalyzer.run_multistart(
    optimizer,
    function,
    starts,
    steps=200,
    tol=1e-6,
)

clusters = MultiStartAnalyzer.cluster_minima(trajectories, distance_threshold=0.5)
print(f"clusters: {len(clusters)}")
```

## Basin of attraction plot

```python
from functions import NonConvex
from optimizers import GradientDescent
from experiments.compare_optimizers import MultiStartAnalyzer

function = NonConvex()
optimizer = GradientDescent(step_size=0.1)

fig, ax = MultiStartAnalyzer.plot_basin_of_attraction(
    function.f,
    optimizer=optimizer,
    function=function,
    x_range=(-3, 3),
    y_range=(-2, 4),
    resolution=30,
    n_steps=200,
    title="Basin of Attraction",
)
```

## Diagnostics-only workflow

```python
import numpy as np
from functions import IllConditioned
from dynamics import Diagnostics

function = IllConditioned(a=1, b=100)
x = np.array([2.0, -1.0])

print("grad norm:", Diagnostics.gradient_norm(function.f, x))
print("eigvals:", Diagnostics.hessian_eigenvalues(function.f, x))
print("cond:", Diagnostics.condition_number(function.f, x))
```

## Visualization orchestration

```python
from functions import IllConditioned
from optimizers import GradientDescent, MomentumGD
from dynamics import TrajectoryRunner
from visualization.trajectories import VisualizationRunner

function = IllConditioned(a=1, b=100)
trajs = TrajectoryRunner.run_comparison(
    [GradientDescent(0.05), MomentumGD(0.05, 0.9)],
    function,
    x0=[4.0, -3.0],
    steps=250,
)

fig = VisualizationRunner.create_comparison_figure(
    function.f,
    trajs,
    x_range=(-5, 5),
    y_range=(-5, 5),
    title="Optimizer Comparison",
)
```

## Constrained optimization examples

### Lagrange multiplier method

```python
import numpy as np
from dynamics.constrained import LagrangeMultiplierMethod, create_circle_constraint_problem

problem = create_circle_constraint_problem()
solver = LagrangeMultiplierMethod(step_size=0.01)

x_opt, lambda_opt, history = solver.optimize(
    problem.f,
    problem.g,
    x0=np.array([1.5, 1.5]),
    lambda0=1.0,
    steps=300,
)

print("x*:", x_opt)
print("lambda*:", lambda_opt)
print("|g(x*)|:", abs(problem.g(x_opt)))
```

### Penalty method

```python
import numpy as np
from dynamics.constrained import PenaltyMethod, create_ellipse_constraint_problem

problem = create_ellipse_constraint_problem()
solver = PenaltyMethod(step_size=0.01)

x_opt, rho_final, history = solver.optimize(
    problem.f,
    problem.g,
    x0=np.array([1.5, 0.5]),
    rho_init=1.0,
    steps=200,
    rho_increase_rate=5.0,
)

print("x*:", x_opt)
print("rho_final:", rho_final)
print("|g(x*)|:", abs(problem.g(x_opt)))
```

## Reproduce experiment scripts

```bash
python experiments/conditioning_effects.py
python experiments/initialization_sensitivity.py
python -c "from experiments.failure_modes import FailureModes; FailureModes.create_all_failure_demos()"
```
