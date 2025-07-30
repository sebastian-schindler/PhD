"""
Astronomy plotting utilities and visualizations.

This module provides specialized plotting functions for astronomical data,
including sky plots, interactive legends, and enhanced histograms.
"""

# Third-party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import astropy.coordinates as coord
import astropy.units as u

# Type checking imports
from typing import Any
from numpy.typing import ArrayLike
from astropy.units.quantity import Quantity


# plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.grid'] = True


def _transform_equatorial_coords(ra: Quantity) -> Quantity:
	"""Transform equatorial coordinates for Mollweide projection."""
	ra = ra + 180*u.deg # type: ignore
	ra = ra.wrap_at(180.*u.deg)
	ra = -ra
	return ra


def _transform_galactic_coords(longitude: Quantity) -> Quantity:
	"""Transform galactic coordinates for Mollweide projection."""
	longitude = longitude.wrap_at(180*u.deg)
	longitude = -longitude
	return longitude


def _compute_skyplot_ticks(left: float, right: float) -> tuple[np.ndarray, list[str]]:
	"""Compute x-axis tick marks and labels for sky plots.
	
	Returns
	-------
	tuple
		(tick_positions, tick_labels) where tick_positions are in radians
		and tick_labels are formatted degree strings.
	"""
	# Create 30-degree step ticks, excluding bounds
	xticks = np.linspace(-180, 180, 13)
	xticks = np.deg2rad(xticks)
	xticks = xticks[1:-1]
	
	xticklabels = []
	for number in np.linspace(left, right, 13):
		xticklabels.append(f"{number:d}\N{DEGREE SIGN}")
	xticklabels = xticklabels[1:-1]
	
	return xticks, xticklabels


def hist(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
	"""
	Plot a histogram with sensible default settings.
	
	This function is a wrapper around `matplotlib.pyplot.hist` that provides
	default values for commonly used parameters.
	
	Parameters
	----------
	*args
		Positional arguments passed to `plt.hist`.
	**kwargs
		Keyword arguments passed to `plt.hist`. Default values are provided
		for `bins` (100) and `histtype` ('step').
	
	Returns
	-------
	histogram_data
		Return value from `plt.hist` containing (n, bins, patches).
	"""
	kwargs_hist = dict(
		bins = 100, 
		histtype = 'step'
	)
	kwargs_hist.update(kwargs)

	return plt.hist(*args, **kwargs_hist)


def plt_legend_toggleable(*args: Any, pickradius: float = 7.) -> None:
	"""
	Add a legend that allows clicking on entries to toggle their visibility.
	
	This function creates an interactive legend where clicking on legend entries
	toggles the visibility of the corresponding plot elements. Invisible elements
	appear with reduced alpha in the legend.
	
	Parameters
	----------
	*args
		Arguments passed to `plt.legend()`.
	pickradius
		Area in points around the legend artist that triggers a click event.
		Values above 10 are discouraged as click areas may overlap.
	"""
	leg = plt.legend(*args)

	ax_objs, labels = plt.gca().get_legend_handles_labels()  # plot object and corresponding labels that are represented in the legend
	leg_objs = leg.legend_handles  # representation of plot object in the legend

	map_leg_to_ax = {}  # map legend objects to axis objects

	for leg_obj, ax_obj, label in zip(leg_objs, ax_objs, labels):
		leg_obj.set_picker(pickradius)  # enable mouse interaction with legend object
		map_leg_to_ax[leg_obj] = ax_obj

	def on_pick(event):  # on pick event, find the original line corresponding to the legend proxy line, and toggle its visibility

		leg_obj = event.artist
		if leg_obj not in map_leg_to_ax:
			return

		ax_obj = map_leg_to_ax[leg_obj]
		visible = not ax_obj.get_visible()
		ax_obj.set_visible(visible)

		# change the alpha on the line in the legend, so we can see what lines have been toggled
		leg_obj.set_alpha(1.0 if visible else 0.2)

		plt.gcf().canvas.draw()

	plt.gcf().canvas.mpl_connect('pick_event', on_pick)

	plt.show()


def plt_skyplot(
	ra: ArrayLike, 
	dec: ArrayLike, 
	galactic: bool = False, 
	galaxy: bool = False, 
	figsize: tuple[float, float] = (16, 8), 
	**kwargs
) -> tuple[plt.Figure, plt.Axes]:
	"""
	Create a sky plot in equatorial or galactic coordinates.

	This function creates a Mollweide projection sky plot from arrays of Right Ascension (RA) and Declination (Dec) values. The plot can be displayed in equatorial or galactic coordinates, with optional galactic plane and center overlays.

	Parameters
	----------
	ra
		Right Ascension values in degrees.
	dec
		Declination values in degrees.
	galactic
		If True, plot in galactic coordinates. If False, plot in equatorial coordinates.
	galaxy
		If True, overlay the galactic plane and galactic center on the plot.
	figsize
		Figure size as (width, height).
	**kwargs
		Additional keyword arguments passed to `plt.scatter`.
		
	Returns
	-------
	tuple
		(figure, axes) objects for further customization.
	"""
	fig = plt.figure(figsize=figsize)
	ax = plt.subplot(projection='mollweide')
	plt.grid()
	coords = coord.SkyCoord(ra*u.deg, dec*u.deg)
	if galactic:
		coords = coords.transform_to('galactic')
		coord_x = _transform_galactic_coords(coords.l)
		coord_y = coords.b
		xticks, xticklabels = _compute_skyplot_ticks(180, -180)
		plt.xticks(xticks, xticklabels)
		plt.text(0.9, 0.1, "Galactic", transform=plt.figure(1).axes[0].transAxes)
 
	else:
		coord_x = _transform_equatorial_coords(coords.ra)
		coord_y = coords.dec
		xticks, xticklabels = _compute_skyplot_ticks(360, 0)
		plt.xticks(xticks, xticklabels)
		plt.text(0.9, 0.1, "Equatorial", transform=plt.figure(1).axes[0].transAxes)

	plt.scatter(coord_x.rad, coord_y.rad, **kwargs)
	
	if galaxy:
		plane = coord.SkyCoord(frame='galactic', l=np.linspace(0, 360, 100)*u.deg, b=0*u.deg)
		plane = plane.transform_to('icrs')
		plt.plot(_transform_equatorial_coords(plane.ra).rad, plane.dec.rad, 'black', linewidth=.5)

		center = coord.SkyCoord(frame='galactic', l=0*u.deg, b=0*u.deg)
		center = center.transform_to('icrs')
		plt.plot(_transform_equatorial_coords(center.ra).rad, center.dec.rad, '*k', markersize=5)

	return fig, ax
