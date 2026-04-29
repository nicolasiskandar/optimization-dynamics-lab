"""Tests for trajectory execution helpers."""

from functions.quadratic import Quadratic
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import MomentumGD
from optimizers.newton import Newton
from dynamics.trajectory import TrajectoryRunner


def test_run_returns_trajectory_with_diagnostics():
    """A single run should expose the summary properties and diagnostics arrays."""
    function = Quadratic()
    trajectory = TrajectoryRunner.run(
        optimizer=GradientDescent(step_size=0.1),
        function=function,
        x0=[3.0, -4.0],
        steps=100,
    )

    assert trajectory.optimizer_name == "GradientDescent"
    assert trajectory.n_steps == len(trajectory.trajectory)
    assert trajectory.f_final <= trajectory.f_initial
    assert len(trajectory.diagnostics["function_values"]) == trajectory.n_steps
    assert "GradientDescent" in trajectory.summary()


def test_run_comparison_preserves_requested_optimizer_order():
    """Comparison runs should return one trajectory per optimizer in order."""
    function = Quadratic()
    trajectories = TrajectoryRunner.run_comparison(
        optimizers=[
            GradientDescent(step_size=0.1),
            MomentumGD(step_size=0.1),
            Newton(),
        ],
        function=function,
        x0=[3.0, -4.0],
        steps=100,
    )

    assert [traj.optimizer_name for traj in trajectories] == [
        "GradientDescent",
        "MomentumGD",
        "Newton",
    ]
    assert all(traj.f_final <= traj.f_initial for traj in trajectories)


def test_run_multistart_returns_one_trajectory_per_initialization():
    """Multistart should execute the same optimizer for every provided start."""
    function = Quadratic()
    starts = [[3.0, -4.0], [1.0, 1.0], [-2.0, 5.0]]

    trajectories = TrajectoryRunner.run_multistart(
        optimizer=GradientDescent(step_size=0.1),
        function=function,
        x0_list=starts,
        steps=100,
    )

    assert len(trajectories) == len(starts)
    assert [traj.x0.tolist() for traj in trajectories] == starts
    assert all(traj.n_steps >= 1 for traj in trajectories)
