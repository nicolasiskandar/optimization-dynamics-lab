"""Experiment utilities for reproducible optimization analyses.

This package groups higher-level analysis modules that compare optimizer
behavior under different initializations, objective conditioning levels,
and known failure-mode scenarios.
"""

from .compare_optimizers import MultiStartAnalyzer
from .conditioning_effects import ConditioningEffects
from .initialization_sensitivity import InitializationSensitivity
from .failure_modes import FailureModes

__all__ = [
    "MultiStartAnalyzer",
    "ConditioningEffects",
    "InitializationSensitivity",
    "FailureModes",
]
