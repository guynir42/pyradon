"""
pyradon: A Python package for streak detection using Fast Radon Transform.

This package provides tools for detecting streaks in astronomical images
using the Fast Radon Transform (FRT) method.

@author: guyn
"""

__version__ = "0.1.0"

from .finder import Finder
from .simulator import Simulator

__all__ = ['Finder', 'Simulator']
