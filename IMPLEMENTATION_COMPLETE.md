# Implementation Status

This file describes the current implemented state of the repository based on the present codebase.

## Completed Components

### Core numerical derivatives
- `core/gradients.py`: central-difference gradient via `Gradient.get_grad`
- `core/hessian.py`: finite-difference Hessian via `Hessian.get_hessian`

### Objective function suite
- `functions/quadratic.py`: convex isotropic quadratic
- `functions/ill_conditioned.py`: anisotropic quadratic with configurable coefficients
- `functions/saddle.py`: indefinite saddle function
- `functions/nonconvex.py`: smooth non-convex objective
- `functions/plateau.py`: saturating plateau objective

All function classes implement `Function2D` from `functions/base.py` and support vectorized mesh evaluation.

### Unconstrained optimizers
- `GradientDescent`
- `MomentumGD`
- `SGD` (mini-batch stochastic gradient descent over component objectives)
- `Newton`
- `GradientDescentWithLineSearch` (backtracking / golden-section)
- `MomentumWithLineSearch` (backtracking / golden-section)
- Line search primitives in `optimizers/line_search.py`

### Trajectory and diagnostics
- `Trajectory` container with summary and key properties (`n_steps`, `x_final`, `f_final`, `grad_norm_final`)
- `TrajectoryRunner` with single run, comparison run, and multistart run helpers
- `Diagnostics` with gradient norm, Hessian eigenvalues, condition numbers, and trajectory-wide diagnostic extraction

### Constrained optimization
Implemented under `dynamics/constrained/`:
- `ConstrainedOptimizer` base class
- `LagrangeMultiplierMethod`
- `PenaltyMethod`
- `ConstrainedOptimizationProblem`
- `create_circle_constraint_problem`
- `create_ellipse_constraint_problem`

### Visualization stack
- `visualization/contours.py`: 2D contours + vectorized trajectory overlays
- `visualization/vector_fields.py`: gradient vector fields with overlays
- `visualization/loss.py`: loss and gradient norm curves with robust axis formatting
- `visualization/surfaces.py`: robust Plotly 3D surface visualization with vectorized evaluation and divergent trajectory handling
- `visualization/trajectories.py`: orchestration layer for combined figures

### Experiment scripts
- `experiments/compare_optimizers.py`: multistart analysis, clustering, basin plotting
- `experiments/conditioning_effects.py`: optimizer performance across conditioning levels
- `experiments/failure_modes.py`: failure mode generation
- `experiments/initialization_sensitivity.py`: dependence on starting points

### User interfaces
- `app.py`: Streamlit UI for function/optimizer selection, diagnostics, 2D and optional 3D plotting, CSV export
- `demo.py`: command-line walkthrough of all major capabilities

For analytical functions routed through the shared trajectory helpers, SGD is
fed a repeated list of the selected objective so it remains compatible with the
generic visualization and comparison flows.

### Test suite
The repository includes a `pytest` suite in `tests/`:
- `test_gradient.py`
- `test_hessian.py`
- `test_gradient_descent.py`
- `test_momentum.py`
- `test_sgd.py`
- `test_newton.py`
- `test_line_search.py`
- `test_trajectory.py`
- `test_constrained.py`

Run with `pytest` or target specific modules with `pytest tests/<file>.py`.

## Current Notes
- Newton can fail or terminate early when Hessians are singular/indefinite in some regions.
- Several plotting and experiment workflows are intentionally compute-heavy for clarity of analysis.
