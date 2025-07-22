import re
from typing import Iterable
import pandas as pd
from astroquery.simbad import Simbad


def get_catalog_ID(name: str, catalog: str):
	"""
	Get the ID of an object by its common name as it appears in a certain catalog. 
	
	If an error is raised by the Simbad query, make sure to delay subsequent calls to this function by some time.
	
	Parameters
	----------
	name
		Common name of the object that Simbad understands.
	catalog
		Identifier of the catalog, e.g. 'WISE' for the AllWISE catalog, or '2RXS' for the second ROSAT all-sky survey.
	
	Returns
	-------
	The ID in the requested catalog of the requested object. If the object could not be found, 
	or no (or no unique) catalog with the supplied catalog name could be found, returns None.
	"""

	IDs = Simbad.query_objectids(name)
	if IDs is None:
		return None
	IDs = [x.decode() for x in IDs['ID'].data]

	ID_candidates = [x[len(catalog)+1:] for x in IDs if x.startswith(catalog + " ")]

	if len(ID_candidates) == 1:
		return ID_candidates[0]
	return None


def normalize_object_names(names: Iterable[str]) -> pd.Series:
	"""
	Normalize astronomical object names using the Simbad database.

	Parameters
	----------
	names : iterable
		Object names to be normalized.

	Returns
	-------
	pd.Series containing the normalized object names (with same index as the input pandas Series).
	"""
	names = pd.Series(names, copy=True)
	
	result = Simbad.query_objects(names)
	names_normalized = pd.Series(result['main_id'], index=names.index)

	# Retry with some heuristics that SIMBAD doesn't handle itself: add a blank before...
	# ... uppercase letters following lowercase letters (e.g. CygnusA -> Cygnus A)
	names_modified = [re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name) for name in names[names_normalized == '']]
	# ... digits following letters (e.g. 3C273 -> 3C 273)
	names_modified = [re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', name) for name in names_modified]

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
