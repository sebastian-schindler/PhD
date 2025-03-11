from tools_matplotlib import *
# from tools_highdim import *

import numpy as np
import pandas as pd

import os.path as pth
import warnings

def no_nan(array, *args):
	"""
	Remove NaNs from an array, or pandas DataFrame.
	For several arrays (of same shape): All positions that have a NaN in any of the arrays (i.e. the union of Nan positions) are removed from all arrays. A list of the arrays without NaNs is returned.
	"""

	if type(array) is pd.DataFrame and len(args) == 0:  # if DataFrame, only one DataFrame makes sense
		return array.dropna()

	if array.ndim == 2:
		print("Interpreting 2-dimensional array as multiple inputs instead of flattening array.")
		return no_nan(*array.T).T

	mask = ~ np.isnan(array)
	
	if len(args) == 0:
		return array[mask]
	
	for i, arg in enumerate(args):
		if arg.shape != mask.shape:
			raise Exception("All arrays must have the same shape")
		try:
			mask &= ~ np.isnan(arg)
		except TypeError:
			warnings.warn("Ignoring array no. %d in mask creation due to incompatible types" % (i+2))
	
	return np.array( [array[mask]] + [arg[mask] for arg in args] )


def cache_file(url):
	"""Cache a remote file locally and return local path to the cache copy."""

	from numpy.lib._datasource import DataSource

	cache_dir = pth.join(pth.curdir, ".temp")

	ds = DataSource(cache_dir)
	ds.open(url).close()
	
	return ds.abspath(url)


import astropy.coordinates as coord
import astropy.units as u
def plt_skyplot(ra, dec, galactic=False, galaxy=False, figsize=[16,8], **kwargs):
	"""Create skyplot in equatorial or galactic coordinates from arrays of RA and Dec values in degrees."""
	
	def trafo_equ(ra):
		ra = ra + 180*u.deg # type: ignore
		ra = ra.wrap_at(180.*u.deg)
		ra = -ra
		return ra
	def trafo_gal(l):
		l = l.wrap_at(180*u.deg)
		l = -l
		return l
	
	plt.figure(figsize=figsize)
	plt.subplot(projection='mollweide')
	plt.grid()
	
	def set_xticks(left, right):
		
		xticks = np.linspace(-180, 180, 13) # 30 deg steps incl. 0 and bounds
		xticks = np.deg2rad(xticks)
		xticks = xticks[1:-1] # exclude bounds
		
		xticklabels = []
		for number in np.linspace(left, right, 13):
			xticklabels.append(u"%d \N{DEGREE SIGN}" % number)
		xticklabels = xticklabels[1:-1]
		
		plt.xticks(xticks, xticklabels)
	
	coords = coord.SkyCoord(ra*u.deg, dec*u.deg)
	if galactic:
		coords = coords.transform_to('galactic')
		coord_x = trafo_gal(coords.l)
		coord_y = coords.b
		set_xticks(180, -180)
		plt.text(0.9, 0.1, "Galactic", transform=plt.figure(1).axes[0].transAxes)
 
	else:
		coord_x = trafo_equ(coords.ra)
		coord_y = coords.dec
		set_xticks(360, 0)
		plt.text(0.9, 0.1, "Equatorial", transform=plt.figure(1).axes[0].transAxes)

	plt.scatter(coord_x.rad, coord_y.rad, **kwargs)
	
	if galaxy:
		
		plane = coord.SkyCoord(frame='galactic', l=np.linspace(0, 360, 100)*u.deg, b=0*u.deg)
		plane = plane.transform_to('icrs')
		plt.plot(trafo_equ(plane.ra).rad, plane.dec.rad, 'black', linewidth=.5)

		center = coord.SkyCoord(frame='galactic', l=0*u.deg, b=0*u.deg)
		center = center.transform_to('icrs')
		plt.plot(trafo_equ(center.ra).rad, center.dec.rad, '*k', markersize=5)


import pickle as pkl

def pickle(filename, *obj):
	"""
	Save an object to a pickle file. Dumb wrapper for dumb people (or those that can never remember the one-liner).
	
	Parameters
	----------
	filename
		A path where to save the pickle file. Recommended file extension is *.pkl.
	*obj
		Any python objects to save. If several are provided, will save a tuple of the individual objects.
	
	Returns
	-------
		The same object as passed in (so that this wrapper is fully transparent).
	"""
	# PICKLED_IDENTIFIER = "#PICKLETUPLE"

	# if not overwrite and pth.exists(filename):
	# 	pickled = unpickle(filename)
	# 	if type(pickled) == tuple and pickled[0] == PICKLED_IDENTIFIER:
	# 		topickle = pickled + obj
	# 	else:
	# 		topickle = (PICKLED_IDENTIFIER, pickled, obj)
	# else:
	# 	topickle = obj

	# pkl.dump(topickle, open(filename, 'wb'))

	topickle = obj
	if len(obj) == 1:
		topickle = obj[0]

	pkl.dump(topickle, open(filename, 'wb'))

	return obj


def unpickle(filename):
	"""
	Load a pickled object from a pickle file. Dumb wrapper for dumb people (or those that can never remember the one-liner).
	
	Parameters
	----------
	filename
		A path to a pickle file.
	
	Returns
	-------
		The object in the pickle file, duh.
	"""
	return pkl.load(open(filename, 'rb'))


from astroquery.simbad import Simbad
def get_catalog_ID(name, catalog):
	"""
	Get the ID of an object by its common name as it appears in a certain catalog. If an error is raised by the Simbad query, make sure to delay subsequent calls to this function by some time.
	
	Parameters
	----------
	name
		Common name of the object that Simbad understands.
	catalog
		Identifier of the catalog, e.g. 'WISE' for the AllWISE catalog, or '2RXS' for the second ROSAT all-sky survey.
	
	Returns
	-------
		The ID in the requested catalog of the requested object. If the object could not be found, or no (or no unique) catalog with the supplied catalog name could be found, returns None.
	"""

	IDs = Simbad.query_objectids(name)
	if IDs is None:
		return None
	IDs = [x.decode() for x in IDs['ID'].data]

	ID_candidates = [x[len(catalog)+1:] for x in IDs if x.startswith(catalog + " ")]

	if len(ID_candidates) == 1:
		return ID_candidates[0]
	return None


import re
def normalize_object_names(names):
	"""
	Normalize astronomical object names using the Simbad database.

	Parameters:
	object_names (iterable): An iterable of object names to be normalized.

	Returns:
	pd.Series: A pandas Series containing the normalized object names (with same index as the input pandas Series).
	"""
	names = pd.Series(names, copy=True)
	
	result = Simbad.query_objects(names)
	names_normalized = pd.Series(result['main_id'], index=names.index)

	# Retry with some heuristics that SIMBAD doesn't handle itself: add a blank before...
	names_modified = [re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name) for name in names[names_normalized == '']]  # ... uppercase letters following lowercase letters (e.g. CygnusA -> Cygnus A)
	names_modified = [re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', name) for name in names_modified]  # ... digits following letters (e.g. 3C273 -> 3C 273)
	result_retry = Simbad.query_objects(names_modified)
	names_normalized[names_normalized == ''] = result_retry['main_id']

	names_failed = names[names_normalized == '']
	if len(names_failed) > 0:
		print(f"Warning: SIMBAD query failed for {len(names_failed)} objects; their original names will be used instead:")
		[print(f"    {x}") for x in names_failed]
		names_normalized[names_normalized == ''] = names_failed

	# replace N whitespace characters with 1 whitespace character
	names_normalized = names_normalized.apply(lambda x: re.sub(r'\s+', ' ', x))

	# remove the 'NAME' catalog identifier
	names_normalized = names_normalized.apply(lambda x: x.replace('NAME ', '', 1))

	return names_normalized