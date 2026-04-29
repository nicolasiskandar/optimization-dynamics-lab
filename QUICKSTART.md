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

### (Optional) Run the test suite

```bash
pytest
```

To run a subset:

```bash
pytest tests/test_sgd.py
pytest tests/test_constrained.py -k penalty
```

## 4) Generate failure mode plots

```bash
python -c "from experiments.failure_modes import FailureModes; FailureModes.create_all_failure_demos()"
```

Generated figures are written to `failure_modes/`.

## Minimal API example

```python
from functions import IllConditioned
from optimizers import GradientDescent, MomentumGD, Newton, SGD
from dynamics import TrajectoryRunner

function = IllConditioned(a=1, b=100)
optimizers = [
    GradientDescent(step_size=0.05),
    MomentumGD(step_size=0.05, beta=0.9),
    Newton(),
    SGD(step_size=0.01, batch_size=10),
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
