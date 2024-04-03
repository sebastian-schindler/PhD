import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.grid'] = True

def hist(*arg, **kwargs):
	"""Plot a 2D histogram with sensible default settings and conveniences."""

	kwargs_hist = dict(
		bins = 100, 
		histtype = 'step'
	)
	kwargs_hist.update(kwargs)

	return plt.hist(*arg, **kwargs_hist)


def plt_legend_toggleable(*args, pickradius=7):
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

