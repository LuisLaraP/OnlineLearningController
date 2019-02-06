"""
Utilities for managing settings objects.

A settings object is a collection of key-value pairs that represent parameters
or settings to various algorithms and objects in this project. The main way
these are stored is in JSON files. Default values are also provided for some
values, but others have to be given explicitly.
"""

import json
import pkg_resources


def getDefaults(id):
	"""
	Get the default settings for the given object.

	Parameters
	----------
	id : str
		Identifier of the object for which to get defaults. It must be of the form
		`package:file`, where `package` is the full name of the package where the
		defaults were defined and `file` is the filename of the specification.

	Returns
	-------
	settings : dict
		Settings object with the default values.
	"""
	sections = [x.strip() for x in id.split(':')]
	with pkg_resources.resource_stream(sections[0], 'data/'+sections[1]) as file:
		settings = json.load(file)
	return settings


def merge(a, b):
	"""
	Merge two settings files.

	The returned object will have all keys present in object `a` and object `b`.
	However, if both objects have the same key, the value in object `b` takes
	precedence. If there are settings objects inside any of the arguments, they
	will be merged as well.

	Parameters
	----------
	a, b : dict
		Objects to merge.

	Returns
	-------
	merged : dict
		Merged object.
	"""
	if a is None:
		return b
	if b is None:
		return a
	merged = {**a, **b}
	for key, value in a.items():
		if isinstance(value, dict) and key in b and isinstance(b[key], dict):
			merged[key] = merge(value, b[key])
	return merged
