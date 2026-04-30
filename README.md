# Optimization Dynamics Laboratory

Optimization Dynamics Laboratory is a from-scratch Python toolkit for studying optimization algorithms.

It includes:

- Numerical gradient and Hessian estimation
- Unconstrained optimizers (Gradient Descent, Momentum, Newton, mini-batch SGD, line-search variants)
- Equality-constrained solvers (Lagrange multiplier and quadratic penalty methods)
- Trajectory diagnostics (loss, gradient norm, Hessian eigenvalues, condition number)
- Visualization utilities (contours, vector fields, loss curves, 3D surfaces)
- Reproducible experiment scripts and a Streamlit UI

## Demo
[![Optimization Dynamics Lab Demo](https://img.youtube.com/vi/slsHDTdmXA4/0.jpg)](https://www.youtube.com/watch?v=slsHDTdmXA4)

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

The repository uses `pytest` for tests covering:

- numerical gradient and Hessian approximations
- optimizer convergence and documented instability cases
- line-search behavior
- trajectory diagnostics helpers
- constrained optimization solvers

`SGD` now operates on a list of component objectives. In the visualization and
trajectory helpers, single analytical objectives are adapted into repeated
component lists so SGD can still be compared against the other optimizers.

Run the full suite:

```bash
pytest
```

Run a focused file or test:

```bash
pytest tests/test_line_search.py
pytest tests/test_newton.py -k singular
```
