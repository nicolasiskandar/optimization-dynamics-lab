"""Optimization test functions."""

from functions.base import Function2D
from functions.quadratic import Quadratic
from functions.ill_conditioned import IllConditioned
from functions.saddle import Saddle
from functions.nonconvex import NonConvex
from functions.plateau import Plateau

__all__ = [
    'Function2D',
    'Quadratic',
    'IllConditioned',
    'Saddle',
    'NonConvex',
    'Plateau'
]
