# Quick Start

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Launch the interactive app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## 3) Run the full demo

```bash
python demo.py
```

### (Optional) Execute validation scripts

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

## 4) Generate failure mode plots

```bash
python -c "from experiments.failure_modes import FailureModes; FailureModes.create_all_failure_demos()"
```

Generated figures are written to `failure_modes/`.

## Minimal API example

```python
from functions import IllConditioned
from optimizers import GradientDescent, MomentumGD, Newton
from dynamics import TrajectoryRunner

function = IllConditioned(a=1, b=100)
optimizers = [
    GradientDescent(step_size=0.05),
    MomentumGD(step_size=0.05, beta=0.9),
    Newton(),
]

trajectories = TrajectoryRunner.run_comparison(
    optimizers,
    function,
    x0=[4.0, -3.0],
    steps=300,
    tol=1e-6,
)

for traj in trajectories:
    print(traj.optimizer_name, traj.n_steps, traj.f_final)
```
