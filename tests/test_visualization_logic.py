"""Tests for plotting-helper logic without exercising UI flows."""

import numpy as np

from dynamics.trajectory import TrajectoryRunner
from experiments.conditioning_effects import ConditioningEffects
from functions.quadratic import Quadratic
from optimizers.gradient_descent import GradientDescent
from visualization.contours import ContourPlotter
from visualization.loss import LossPlotter


def test_evaluate_function_supports_vectorized_callables():
    """Vectorized functions should be evaluated directly on the mesh."""
    X, Y = ContourPlotter.create_mesh(
        x_range=(-1.0, 1.0), y_range=(-2.0, 2.0), resolution=3)

    def vectorized_f(stacked):
        return stacked[0] ** 2 + stacked[1] ** 2

    Z = ContourPlotter.evaluate_function(vectorized_f, X, Y)

    assert Z.shape == X.shape
    assert Z[1, 1] == 0.0


def test_evaluate_function_falls_back_for_scalar_only_callables():
    """Scalar-only callables should be handled by the loop fallback path."""
    X, Y = ContourPlotter.create_mesh(
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), resolution=2)

    def scalar_f(x):
        if np.ndim(x[0]) > 0:
            raise TypeError("scalar only")
        return x[0] ** 2 + x[1] ** 2

    Z = ContourPlotter.evaluate_function(scalar_f, X, Y)

    expected = np.array([[2.0, 2.0], [2.0, 2.0]])
    assert np.allclose(Z, expected)


def test_to_positive_for_log_shifts_non_positive_values():
    """Log-scaling helper should shift non-positive finite values upward."""
    values = np.array([-3.0, 0.0, 2.0, np.inf])

    shifted, did_shift = LossPlotter._to_positive_for_log(values)

    assert did_shift
    assert np.all(shifted[:3] > 0.0)
    assert np.isinf(shifted[3])


def test_to_positive_for_log_leaves_positive_values_unshifted():
    """Purely positive inputs should be preserved aside from type coercion."""
    values = np.array([1.0, 2.0, 3.0])

    shifted, did_shift = LossPlotter._to_positive_for_log(values)

    assert not did_shift
    assert np.array_equal(shifted, values)


def test_plot_loss_curves_marks_auto_shifted_ylabel_for_negative_series():
    """Negative losses should trigger the auto-shifted y-axis label."""
    function = Quadratic()
    optimizer = GradientDescent(step_size=0.1)
    trajectory = TrajectoryRunner.run(
        optimizer, function, [3.0, -4.0], steps=5)
    trajectory.diagnostics["function_values"] = np.array(
        [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05])

    fig, ax = LossPlotter.plot_loss_curves(trajectory, title="loss")

    assert ax.get_ylabel() == "f(x) [log scale, auto-shifted]"
    fig.clf()


def test_condition_number_metric_replaces_infinities_for_plotting():
    """Infinite condition numbers should be mapped to a finite plotting sentinel."""
    function = Quadratic()
    optimizer = GradientDescent(step_size=0.1)
    trajectory = TrajectoryRunner.run(
        optimizer, function, [3.0, -4.0], steps=5)
    trajectory.diagnostics["condition_numbers"] = np.array([2.0, np.inf, 5.0])

    fig, ax = ConditioningEffects.condition_number_metric(trajectory)
    plotted = ax.lines[0].get_ydata()

    assert np.array_equal(plotted, np.array([2.0, 1e10, 5.0]))
    fig.clf()
