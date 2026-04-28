# Optimization Dynamics Laboratory

Optimization Dynamics Laboratory is a from-scratch Python toolkit for studying optimization algorithms.

It includes:

- Numerical gradient and Hessian estimation
- Unconstrained optimizers (Gradient Descent, Momentum, Newton, line-search variants)
- Equality-constrained solvers (Lagrange multiplier and quadratic penalty methods)
- Trajectory diagnostics (loss, gradient norm, Hessian eigenvalues, condition number)
- Visualization utilities (contours, vector fields, loss curves, 3D surfaces)
- Reproducible experiment scripts and a Streamlit UI

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

### Streamlit app

```bash
streamlit run app.py
```

### Demo script

```bash
python demo.py
```

### Failure mode figure generation

```bash
python -c "from experiments.failure_modes import FailureModes; FailureModes.create_all_failure_demos()"
```

## Experiments

- `experiments/compare_optimizers.py`: multi-start runs, clustering of converged points, basin visualizations
- `experiments/conditioning_effects.py`: behavior across condition numbers
- `experiments/failure_modes.py`: generated failure mode figures
- `experiments/initialization_sensitivity.py`: sensitivity to initial states on non-convex landscapes

## Testing

Tests are script-style files in `tests/` (they execute assertions at module level).

Run one test script:

```bash
PYTHONPATH=. python tests/test_line_search.py
```

Run all scripts:

```bash
PYTHONPATH=. python tests/test_gradient.py
PYTHONPATH=. python tests/test_hessian.py
PYTHONPATH=. python tests/test_gradient_descent.py
PYTHONPATH=. python tests/test_momentum.py
PYTHONPATH=. python tests/test_newton.py
PYTHONPATH=. python tests/test_trajectory.py
PYTHONPATH=. python tests/test_line_search.py
PYTHONPATH=. python tests/test_constrained.py
```
