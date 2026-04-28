"""Integration checks for trajectory execution and comparison helpers."""

from dynamics.trajectory import TrajectoryRunner
from functions.quadratic import Quadratic
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import MomentumGD
from optimizers.newton import Newton


function = Quadratic()
x0 = [3, -4]

trajectory = TrajectoryRunner.run(
    optimizer=GradientDescent(step_size=0.1),
    function=function,
    x0=x0,
    steps=100,
)

print(trajectory.summary())
assert trajectory.n_steps >= 1
assert trajectory.f_final <= trajectory.f_initial

comparison = TrajectoryRunner.run_comparison(
    optimizers=[
        GradientDescent(step_size=0.1),
        MomentumGD(step_size=0.1),
        Newton(),
    ],
    function=function,
    x0=x0,
    steps=100,
)

print(f"comparison trajectories: {len(comparison)}")
assert len(comparison) == 3
assert all(traj.f_final <= traj.f_initial for traj in comparison)

multistart = TrajectoryRunner.run_multistart(
    optimizer=GradientDescent(step_size=0.1),
    function=function,
    x0_list=[[3, -4], [1, 1], [-2, 5]],
    steps=100,
)

print(f"multistart trajectories: {len(multistart)}")
assert len(multistart) == 3
assert all(traj.n_steps >= 1 for traj in multistart)
