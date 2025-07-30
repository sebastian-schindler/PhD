"""
Package bastiastro for astronomical data analysis and visualization.

This package provides utilities for:
- Data cleaning and manipulation (core)
- Plotting and visualization (plotting) 
- High-dimensional analysis (highdim)
- Astronomy-specific tools (astronomy)

Usage:
    from bastiastro import *  # Import everything (backwards compatible)
    from bastiastro.plotting import plt_skyplot  # Import specific modules
"""

# Import everything from submodules for backwards compatibility
from .core import *
from .plotting import *
from .highdim import *
from .astronomy import *

# Optional: Define version
__version__ = "1.0.0"
