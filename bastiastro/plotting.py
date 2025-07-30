import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import astropy.coordinates as coord
import astropy.units as u


# plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.grid'] = True

def hist(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
	"""Plot a 2D histogram with sensible default settings and conveniences."""

	kwargs_hist = dict(
		bins = 100, 
		histtype = 'step'
	)
	kwargs_hist.update(kwargs)

	return plt.hist(*args, **kwargs_hist)


def plt_legend_toggleable(*args: Any, pickradius: float = 7.) -> None:
	"""
	Add a legend that allows clicking on entries to toggle their visibility.
	
	Parameters
	----------
	args
		Passed to `plt.legend()`.
	pickradius
		Area in points around the legend artist that trigger a click event. Values above 10 are discouraged, because then typically the click areas start to overlap.
	
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
	Create a sky plot in equatorial or galactic coordinates from arrays of RA and Dec values in degrees.

	Parameters
	----------
	ra : array-like
		Right Ascension values in degrees.
	dec : array-like
		Declination values in degrees.
	galactic
		If True, plot in galactic coordinates. If False, plot in equatorial coordinates.
	galaxy
		If True, overlay the galactic plane and galactic center on the plot.
	figsize
		Figure size as [width, height].
	**kwargs
		Additional keyword arguments passed to `plt.scatter`.
	"""

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

