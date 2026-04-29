"""Tests for non-UI experiment helpers."""

import numpy as np

from experiments.compare_optimizers import MultiStartAnalyzer
from experiments.conditioning_effects import ConditioningEffects
from experiments.initialization_sensitivity import InitializationSensitivity
from functions.nonconvex import NonConvex
from functions.quadratic import Quadratic
from optimizers.gradient_descent import GradientDescent


def test_generate_grid_starts_covers_cartesian_product():
    """Grid starts should contain all x/y combinations in row-major order."""
    starts = MultiStartAnalyzer.generate_grid_starts(
        x_range=(-1.0, 1.0),
        y_range=(0.0, 2.0),
        n_per_axis=3,
    )

    assert starts.shape == (9, 2)
    assert np.allclose(starts[0], np.array([-1.0, 0.0]))
    assert np.allclose(starts[-1], np.array([1.0, 2.0]))


def test_generate_random_starts_is_reproducible_and_bounded():
    """Seeded random starts should be deterministic and stay in range."""
    starts_a = MultiStartAnalyzer.generate_random_starts(
        x_range=(-2.0, 2.0),
        y_range=(-1.0, 3.0),
        n_points=5,
        seed=42,
    )
    starts_b = MultiStartAnalyzer.generate_random_starts(
        x_range=(-2.0, 2.0),
        y_range=(-1.0, 3.0),
        n_points=5,
        seed=42,
    )

    assert np.array_equal(starts_a, starts_b)
    assert np.all(starts_a[:, 0] >= -2.0)
    assert np.all(starts_a[:, 0] <= 2.0)
    assert np.all(starts_a[:, 1] >= -1.0)
    assert np.all(starts_a[:, 1] <= 3.0)


def test_cluster_minima_groups_close_trajectories():
    """Final points within the threshold should share a cluster."""
    function = Quadratic()
    optimizer = GradientDescent(step_size=0.1)
    trajectories = [
        MultiStartAnalyzer.run_multistart(
            optimizer, function, [[3.0, -4.0]], steps=5)[0],
        MultiStartAnalyzer.run_multistart(
            optimizer, function, [[3.1, -4.1]], steps=5)[0],
        MultiStartAnalyzer.run_multistart(
            optimizer, function, [[-3.0, 4.0]], steps=5)[0],
    ]

    cluster_map = MultiStartAnalyzer.cluster_minima(
        trajectories,
        distance_threshold=0.3,
    )

    flattened = sorted(sorted(indices) for indices in cluster_map.values())
    assert flattened == [[0, 1], [2]]


def test_distance_from_optimum_is_computed_per_iterate():
    """Distance series should match the trajectory length and Euclidean norm."""
    function = Quadratic()
    optimizer = GradientDescent(step_size=0.1)
    trajectory = MultiStartAnalyzer.run_multistart(
        optimizer, function, [[3.0, -4.0]], steps=3
    )[0]

    distances = InitializationSensitivity.distance_from_optimum(
        trajectory,
        x_opt=np.array([0.0, 0.0]),
    )

    assert len(distances) == trajectory.n_steps
    assert distances[0] == 5.0
    assert distances[-1] < distances[0]


def test_convergence_rate_analysis_returns_aligned_results():
    """The analysis result should preserve starts, trajectories, and distances."""
    function = NonConvex()
    optimizer = GradientDescent(step_size=0.1)
    starts = [np.array([1.0, 2.0]), np.array([2.0, -1.0])]

    results = InitializationSensitivity.convergence_rate_analysis(
        optimizer,
        function,
        starting_points=starts,
        x_opt=np.array([-np.pi, 0.0]),
        n_steps=20,
    )

    assert len(results["trajectories"]) == len(starts)
    assert len(results["distances"]) == len(starts)
    assert results["starting_points"] == starts
    assert all(len(dist) == traj.n_steps for dist, traj in zip(
        results["distances"], results["trajectories"]))


def test_compare_conditioning_levels_reports_expected_condition_numbers():
    """Conditioning experiments should expose the configured spectrum ratios."""
    results = ConditioningEffects.compare_conditioning_levels(
        step_size=0.01, n_steps=5)

    assert set(results) == {
        "Well-conditioned (κ=2)",
        "Moderate (κ=10)",
        "Ill-conditioned (κ=100)",
        "Very ill-conditioned (κ=1000)",
    }
    assert results["Well-conditioned (κ=2)"]["condition_number"] == 2
    assert results["Moderate (κ=10)"]["condition_number"] == 10
    assert results["Ill-conditioned (κ=100)"]["condition_number"] == 100
    assert results["Very ill-conditioned (κ=1000)"]["condition_number"] == 1000
    assert all(len(data["trajectories"]) == 2 for data in results.values())
