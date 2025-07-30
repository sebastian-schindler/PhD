import numpy as np
import pandas as pd
import os.path as pth
import warnings
import pickle as pkl

from typing import Any, Union
from numpy.typing import ArrayLike


def no_nan(*arrays: Union[ArrayLike, pd.DataFrame]) -> Union[ArrayLike, pd.DataFrame, list[ArrayLike]]:
	"""
	Remove NaNs from an array or pandas DataFrame.
	
	For several arrays of same shape: All positions that have a NaN in any of the arrays 
	(i.e. the union of NaN positions) are removed from all arrays.
	
	Parameters
	----------
	*arrays
		One or more arrays or a single DataFrame to process. All arrays must have the same shape.
	
	Returns
	-------
	For single input, returns the cleaned array or DataFrame. For multiple inputs, 
	returns list of cleaned arrays.
	"""
	if len(arrays) == 0:
		raise ValueError("At least one array must be provided")
	
	first_array = arrays[0]
	
	# Handle DataFrame case - only supports single DataFrame
	if isinstance(first_array, pd.DataFrame):
		if len(arrays) == 1:
			return first_array.dropna()
		else:
			raise ValueError("Multiple DataFrames are not supported. Convert to numpy arrays first.")

	if first_array.ndim == 2 and len(arrays) == 1:
		print("Interpreting 2-dimensional array as multiple inputs instead of flattening array.")
		return no_nan(*first_array.T).T

	mask = ~np.isnan(first_array)
	
	if len(arrays) == 1:
		return first_array[mask]
	
	for i, array in enumerate(arrays[1:], 1):
		if array.shape != mask.shape:
			raise ValueError("All arrays must have the same shape")
		try:
			mask &= ~np.isnan(array)
		except TypeError:
			warnings.warn(f"Ignoring array no. {i+1} in mask creation due to incompatible types")
	
	return [array[mask] for array in arrays]


def cache_file(url: str) -> str:
	"""
	Cache a remote file locally and return local path to the cache copy.
	
	Parameters
	----------
	url
		URL of the remote file to cache.
	
	Returns
	-------
	Local absolute path to the cached file.
	"""

	from numpy.lib._datasource import DataSource

	cache_dir = pth.join(pth.curdir, ".temp")

	ds = DataSource(cache_dir)
	ds.open(url).close()
	
	return ds.abspath(url)


def pickle(filename: str, *obj) -> tuple:
	"""
	Save objects to a pickle file. Simple convenience wrapper.
	
	Parameters
	----------
	filename
		Path where to save the pickle file. Recommended file extension is *.pkl.
	*obj
		Python objects to save. If several are provided, will save a tuple of the individual objects.
	
	Returns
	-------
	The same object(s) as passed in (so that this wrapper is fully transparent).
	"""

	if len(obj) == 0:
		raise ValueError("No object to pickle provided. Please provide at least one object to pickle.")
	elif len(obj) == 1:
		topickle = obj[0]
	else:
		topickle = obj

	with open(filename, 'wb') as f:
		pkl.dump(topickle, f)

	return obj


def unpickle(filename: str) -> Any:
	"""
	Load a pickled object from a pickle file. Dumb wrapper for dumb people (or those that can never remember the one-liner).
	
	Parameters
	----------
	filename
		Path to a pickle file.
	
	Returns
	-------
	The object in the pickle file, duh.
	"""
	with open(filename, 'rb') as f:
		return pkl.load(f)