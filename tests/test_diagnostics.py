"""Unit tests for trajectory diagnostics logic."""

import numpy as np
import pytest

from dynamics.diagnostics import Diagnostics
from functions.ill_conditioned import IllConditioned
from functions.quadratic import Quadratic
from functions.saddle import Saddle


def test_gradient_norm_matches_quadratic_analytic_value():
    """Gradient norm should agree with the analytic quadratic gradient."""
    function = Quadratic()
    x = np.array([3.0, -4.0])

    grad_norm = Diagnostics.gradient_norm(function.f, x)

    assert grad_norm == pytest.approx(np.sqrt(100.0), abs=1e-9)


def test_hessian_eigenvalues_match_ill_conditioned_quadratic():
    """Eigenvalues should recover the diagonal Hessian spectrum."""
    function = IllConditioned(a=1, b=100)

    eigenvalues = Diagnostics.hessian_eigenvalues(
        function.f, np.array([1.0, 2.0]))

    assert np.allclose(eigenvalues, np.array(
        [2.0, 200.0]), atol=5e-2, rtol=0.0)


def test_condition_number_uses_absolute_eigenvalues():
    """Indefinite Hessians should still yield a finite absolute-spectrum ratio."""
    function = Saddle()

    cond_num = Diagnostics.condition_number(function.f, np.array([1.0, -1.0]))

    assert cond_num == 1.0


def test_compute_trajectory_diagnostics_returns_aligned_series():
    """All diagnostic arrays should align with trajectory length."""
    function = Quadratic()
    trajectory = np.array([[3.0, -4.0], [1.0, -1.0], [0.0, 0.0]])

    diagnostics = Diagnostics.compute_trajectory_diagnostics(
        function.f, trajectory)

    assert set(diagnostics) == {
        "gradient_norms",
        "eigenvalues",
        "condition_numbers",
        "function_values",
    }
    assert diagnostics["gradient_norms"].shape == (3,)
    assert diagnostics["condition_numbers"].shape == (3,)
    assert diagnostics["function_values"].shape == (3,)
    assert len(diagnostics["eigenvalues"]) == 3
    assert np.allclose(
        diagnostics["function_values"], np.array([25.0, 2.0, 0.0]))
    assert diagnostics["gradient_norms"][-1] < 1e-9


def test_final_diagnostics_prints_expected_summary(capsys):
    """The human-readable summary should include the key metrics."""
    function = Quadratic()

    Diagnostics.final_diagnostics(function.f, np.array([1.0, -2.0]))
    captured = capsys.readouterr()

    assert "Final diagnostics:" in captured.out
    assert "f(x) =" in captured.out
    assert "Eigenvalues:" in captured.out
    assert "Condition number:" in captured.out
