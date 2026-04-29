"""Tests for stochastic gradient descent."""

import numpy as np

from functions.quadratic import Quadratic
from dynamics.trajectory import TrajectoryRunner
from optimizers.sgd import SGD


def make_component_quadratics():
    """Build a small finite-sum quadratic objective for SGD tests."""
    coeffs = [(1.0, 1.0), (1.5, 0.5), (0.5, 1.5), (2.0, 1.0)]
    return [
        lambda x, a=a, b=b: a * x[0] ** 2 + b * x[1] ** 2
        for a, b in coeffs
    ]


def test_sgd_returns_expected_trajectory_shape():
    """SGD should return a 2D trajectory including the initial point."""
    optimizer = SGD(step_size=0.01, batch_size=2, seed=42)

    path = optimizer.optimize(make_component_quadratics(), x0=[
                              3.0, -4.0], steps=100)

    assert path.shape == (101, 2)


def test_sgd_is_deterministic_for_same_seed():
    """Using the same seed should reproduce the same sampled batches."""
    functions = make_component_quadratics()
    optimizer = SGD(step_size=0.01, batch_size=2, seed=42)

    path_a = optimizer.optimize(functions, x0=[3.0, -4.0], steps=50)
    path_b = optimizer.optimize(functions, x0=[3.0, -4.0], steps=50)

    assert np.array_equal(path_a, path_b)


def test_sgd_differs_for_different_seeds():
    """Different seeds should sample different mini-batch trajectories."""
    functions = make_component_quadratics()

    path_a = SGD(step_size=0.01, batch_size=2, seed=1).optimize(
        functions, x0=[3.0, -4.0], steps=50
    )
    path_b = SGD(step_size=0.01, batch_size=2, seed=2).optimize(
        functions, x0=[3.0, -4.0], steps=50
    )

    assert not np.array_equal(path_a, path_b)


def test_full_batch_sampling_is_seed_independent():
    """Full-batch SGD should be deterministic because every function is used."""
    functions = make_component_quadratics()

    path_a = SGD(step_size=0.01, batch_size=len(functions), seed=1).optimize(
        functions, x0=[3.0, -4.0], steps=30
    )
    path_b = SGD(step_size=0.01, batch_size=len(functions), seed=99).optimize(
        functions, x0=[3.0, -4.0], steps=30
    )

    assert np.array_equal(path_a, path_b)


def test_sgd_reduces_average_component_objective():
    """Mini-batch SGD should make progress on the finite-sum quadratic."""
    functions = make_component_quadratics()
    optimizer = SGD(step_size=0.05, batch_size=2, seed=42)
    x0 = np.array([3.0, -4.0])

    path = optimizer.optimize(functions, x0=x0, steps=20)
    def average_value(x): return sum(f(x) for f in functions) / len(functions)

    assert average_value(path[-1]) < average_value(x0)


def test_batch_size_cannot_exceed_number_of_functions():
    """The implementation should reject invalid mini-batch sizes."""
    functions = make_component_quadratics()

    try:
        SGD(step_size=0.01, batch_size=len(functions) + 1).optimize(
            functions, x0=[3.0, -4.0], steps=10
        )
    except ValueError as exc:
        assert "batch_size cannot be larger" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when batch_size exceeds data size")


def test_trajectory_runner_adapts_single_objective_for_sgd():
    """Shared trajectory/visualization helpers should still work with SGD."""
    function = Quadratic()
    optimizer = SGD(step_size=0.1, batch_size=4, seed=42)

    trajectory = TrajectoryRunner.run(
        optimizer, function, [3.0, -4.0], steps=20)

    assert trajectory.n_steps == 21
    assert function.f(trajectory.x_final) < function.f(trajectory.x0)
