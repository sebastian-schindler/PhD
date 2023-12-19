# tools for high-dimensional explorative data analysis

import numpy as np
import matplotlib.pyplot as plt
import corner

from tools import *


def plot_with_marginals(x, y, figsize=(10, 10), hist=False, log=False, SOI=None, names_1RXS=None):
	"""2D plot (scatter plot or color histogram) with 1D histograms of the marginals at the sides. Adapted from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html"""
	 
	fig = plt.figure(figsize=figsize)
	gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
	ax = fig.add_subplot(gs[1, 0])
	ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
	ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

	# no labels
	ax_histx.tick_params(axis="x", labelbottom=False)
	ax_histy.tick_params(axis="y", labelleft=False)

	# remove NaNs for histogramming
	if names_1RXS is None:
		x, y = no_nan(x, y)
	else:
		x, y, names_1RXS = no_nan(x, y, names_1RXS)
	
	# main plot
	if hist:
		def plot_ax(**kwargs):
			ax.clear()
			ax.hist2d(x, y, cmap=plt.matplotlib.colormaps['Blues'], **kwargs);
	else:
		log = False
		def plot_ax(**kwargs):
			ax.clear()
			ax.plot(x, y, ',', **kwargs);
	
	# x marginalization (above)
	def plot_histx(**kwargs):
		ax_histx.clear()
		ax_histx.hist(x, **kwargs);

	# y marginalisation (right)
	def plot_histy(**kwargs):
		ax_histy.clear()
		ax_histy.hist(y, orientation='horizontal', **kwargs);
	
	if log:
		plot_ax(bins=100, norm=LogNorm())
		plot_histx(bins=100, log=True)
		plot_histy(bins=100, log=True)
	else:
		if hist:
			plot_ax(bins=100)
		else:
			plot_ax()
		plot_histx(bins=100)
		plot_histy(bins=100)
	
	# add sources of interest
	if SOI is not None:
		for name, name_1RXS in SOI.items():
			mask = names_1RXS == name_1RXS
			ax.plot(x[mask], y[mask], '.r')
			ax.text(x[mask], y[mask], name)
	
	return ax, plot_ax, plot_histx, plot_histy


import hdbscan
import seaborn as sns

hdbscan_results = None
def do_cluster(data, plot_kwargs={}, plot_dims=(0, 1), **kwargs):
	
	try:
		data.shape
	except AttributeError:
		data = np.array(data)

	if "approx_min_span_tree" not in kwargs:
		kwargs["approx_min_span_tree"] = False
	
	plot_kwargs_ = {
		"linewidth": 0,
		"s": 1,
		"alpha": 0.5
	}
	plot_kwargs_.update(plot_kwargs)
	
	clusterer = hdbscan.HDBSCAN(**kwargs).fit(data.T)
	print("Found %d clusters" % (clusterer.labels_.max() + 1))

	color_palette = sns.color_palette('dark', clusterer.labels_.max()+1)
	cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
	cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]

	global hdbscan_results
	hdbscan_results = (clusterer, color_palette)

	fig = plt.figure(figsize=(18, 10))
	gs = fig.add_gridspec(2, 3, height_ratios=(3, 2))
	ax_main = fig.add_subplot(gs[0, 0])
	ax_comp = fig.add_subplot(gs[0, 1])
	ax_comp2 = fig.add_subplot(gs[0, 2])
	ax_tree = fig.add_subplot(gs[1, :])
	ax_main.sharex(ax_comp)
	ax_main.sharey(ax_comp)
	ax_comp.sharex(ax_comp2)
	ax_comp.sharey(ax_comp2)
	
	clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=color_palette, axis=ax_tree)
	
	# dimensions for plotting
	data = data[plot_dims, :]
	if data.shape[0] > 2:
		print("Note: using only two dimensions for plotting")
	
	ax_main.scatter(*data, c=cluster_member_colors, **plot_kwargs_)
	ax_comp.scatter(*data, color=(0.5, 0.5, 0.5), **plot_kwargs_)
	ax_comp2.scatter(*data.T[clusterer.labels_ >= 0].T, c=np.array(cluster_member_colors)[clusterer.labels_ >= 0], **plot_kwargs_)
	
	for i in range(clusterer.labels_.max()+1):
		
		data_cluster = data.T[clusterer.labels_ == i].T
		xy = no_nan(data_cluster[0]).min(), no_nan(data_cluster[1]).min()
		width = no_nan(data_cluster[0]).max() - xy[0]
		height = no_nan(data_cluster[1]).max() - xy[1]
		
		patch = plt.Rectangle(xy, width, height, fill=False, color=color_palette[i])
		ax_comp.add_patch(patch)
	
	plt.sca(ax_main)
	plt.tight_layout()
	
	return clusterer

def plot_tree():
	global hdbscan_results
	clusterer, color_palette = hdbscan_results
	
	plt.figure(figsize=(10, 6))
	clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=color_palette)


def cluster_corner(data, labels=None, fig=None, plot_kwargs={}, corner_kwargs={}, **kwargs):

	corner_kwargs_ = dict(bins=100, range=np.array([np.nanmin(data, axis=0), np.nanmax(data, axis=0)]).T, labels=labels, plot_contours=False, plot_density=False)
	plot_kwargs_ = dict(marker=',', alpha=0.1)
	
	plot_kwargs_.update(plot_kwargs)
	corner_kwargs_.update(corner_kwargs)

	if "approx_min_span_tree" not in kwargs:
		kwargs["approx_min_span_tree"] = False


	clusterer = hdbscan.HDBSCAN(**kwargs).fit(data)
	n_cluster = clusterer.labels_.max() + 1
	print("Found %d clusters" % n_cluster)

	color_palette = sns.color_palette('dark', n_cluster)
	color_palette.insert(0, (0.5, 0.5, 0.5))

	if not fig:
		fig = plt.figure(figsize=(20, 20))

	kwargs = dict(corner_kwargs_)
	kwargs.update(plot_datapoints=False)
	fig = corner.corner(
		no_nan(data), 
		fig=fig, 
		color=(0., 0., 0.), 
		data_kwargs=dict(plot_kwargs_), **kwargs);

	for label in range(-1, n_cluster):
		mask = clusterer.labels_ == label
		fig = corner.corner(
			no_nan(data[mask]), 
			fig=fig, 
			color=color_palette[label], 
			data_kwargs=dict(plot_kwargs_), **dict(corner_kwargs_));
	
	ndim = data.shape[1]
	hists1d = np.array(fig.axes).reshape((ndim, ndim)).diagonal()
	for ax in hists1d:
		ax.autoscale(axis='y')

	return fig
