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

from types import ModuleType
from typing import Any

def _lazy_import(module_name: str) -> ModuleType:
	"""
	Import a module lazily with helpful error message if not available.
	
	This function provides a simple way to import optional dependencies with
	clear error messages when packages are missing.
	
	Parameters
	----------
	module_name
		Name of the module to import.
		
	Returns
	-------
	module
		The imported module object.
		
	Raises
	------
	ImportError
		If the module is not installed, with helpful installation message.
		
	Example
	-------
	# Import optional dependencies lazily
	from bastiastro import _lazy_import
	
	# For modules used with their original name
	_lazy_import('hdbscan')
	clusterer = hdbscan.HDBSCAN()
	
	# For modules needing an alias
	ak = _lazy_import('awkward')
	array = ak.Array([1, 2, 3])
	"""
	try:
		return __import__(module_name)
	except ImportError:
		raise ImportError(
			f"Optional dependency '{module_name}' is not installed. "
			f"Install it with: pip install {module_name}"
		) from None


from warnings import warn

def _warn(message: str, deprecation: bool = False, **kwargs: Any) -> None:
	"""Wrapper for warnings.warn with stacklevel=2 default and deprecation flag."""
	# Set default stacklevel if not provided
	if 'stacklevel' not in kwargs:
		kwargs['stacklevel'] = 2
	
	# Set category based on deprecation flag
	if 'category' not in kwargs:
		kwargs['category'] = DeprecationWarning if deprecation else UserWarning
	
	return warn(message, **kwargs)

# Import everything from submodules for backwards compatibility
from .core import *
from .plotting import *
from .highdim import *
from .astronomy import *

# Optional: Define version
__version__ = "1.0.0"
