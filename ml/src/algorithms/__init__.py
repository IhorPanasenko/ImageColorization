"""Classical (training-free) colorization algorithms.

Welsh 2002: texture-based color transfer from a reference image.
Levin 2004: scribble-based colorization via sparse linear optimization.
"""
from .welsh import colorize_welsh
from .levin import colorize_levin

__all__ = ['colorize_welsh', 'colorize_levin']
