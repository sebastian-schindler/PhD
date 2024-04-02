# tools for high-dimensional explorative data analysis

import numpy as np
import matplotlib.pyplot as plt
import corner

from tools import *

import warnings
warnings.filterwarnings("ignore", module="corner")


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


import plotly.graph_objects as go
def cluster_corner(data, labels=None, fig=None, plot_kwargs={}, corner_kwargs={}, plot_3d=False, **kwargs):

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

	if plot_3d:
		markers = []

	else:
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
		data_cluster = no_nan(data[mask])

		if data_cluster.shape[0] == 0:
			continue

		# circumvent assertion in corner package by adding first data as many times as needed
		while data_cluster.shape[0] <= data_cluster.shape[1]:
			data_cluster = np.concatenate((data_cluster, data_cluster[:1]))
			print("padded data in cluster %d to circumvent assertion error" % label)

		if plot_3d:
			marker = go.Scatter3d(
				x=data_cluster.T[0], 
				y=data_cluster.T[1], 
				z=data_cluster.T[2], 
				marker=go.scatter3d.Marker(size=1, color=color_palette[label]), 
				opacity=1, 
				mode='markers'
			)
			markers.append(marker)

		else:
			fig = corner.corner(
				data_cluster, 
				fig=fig, 
				color=color_palette[label], 
				data_kwargs=dict(plot_kwargs_), **dict(corner_kwargs_));

	if plot_3d:
		layout = go.Layout(height=1000)
		fig = go.Figure(data=markers, layout=layout)
		fig.update_layout(scene_aspectmode='cube')
		fig.show()

	else:
		ndim = data.shape[1]
		hists1d = np.array(fig.axes).reshape((ndim, ndim)).diagonal()
		for ax in hists1d:
			ax.autoscale(axis='y')

	return fig


def do_clustering(data, **kwargs):
	"""
	Perform unsupervised clustering of data with HDBSCAN algorithm.
	
	Parameters
	----------
	data
		Data points in arbitrary dimensions to be clustered, array of shape (n_samples, n_dims).
	
	Returns
	-------
	data
		The same data object that was passed in for convenience.
	cluster_labels
		Label of the associated cluster (integer) for all data points, array of length n_samples. Values of -1 denote unclustered data, 0 the first cluster, 1 the second etc.
	cluster_probabilities
		Probability of cluster association for all data points, array of length n_samples.
	"""

	kwargs_hdbscan = dict(approx_min_span_tree=False)
	kwargs_hdbscan.update(kwargs)
	
	clusterer = hdbscan.HDBSCAN(**kwargs_hdbscan).fit(data)
	n_cluster = clusterer.labels_.max() + 1

	print("Found %d clusters:" % n_cluster)
	for label in range(-1, n_cluster):
		n_entries = np.sum(clusterer.labels_ == label)
		print(" cluster %d: %d entries (%.2f %%)" % (
			label, 
			n_entries, 
			n_entries / len(clusterer.labels_) * 100
		))

	return data, clusterer.labels_, clusterer.probabilities_


import tempfile
import awkward as ak
def do_clustering_scan(data, scan_cluster_size, scan_samples=None):
	"""
	Perform unsupervised clustering of data with HDBSCAN algorithm, scanning through hyperparameters min_cluster_size and min_samples of HDBSCAN.

	Parameters
	----------
	data
		Data points in arbitrary dimensions to be clustered, array of shape (n_samples, n_dims).
	scan_cluster_size
		Tuple (min, max, step) for scanning the min_cluster_size parameter of HDBSCAN. If step size is omitted, will be set to 1.
	scan_samples
		Tuple (min, max, step) for scanning the min_samples parameter of HDBSCAN. If step size is omitted, will be set to 1. If omitted entirely, will be set to cover the entire parameter space, i.e. (1, max(scan_cluster_size)).
	Alternatively, for both scan_* parameters, a (non-tuple) iterable can be provided that already contains the scan values.

	Returns
	-------
	Awkward array with min_samples/min_cluster_size scan points along first/second axis. Third axis contains the numbers of points of each cluster, starting with unclustered points. Because the number of clusters is variable, this third axis has variable length.
	"""

	def create_iterable(scan_parameter):
		if type(scan_parameter) != tuple: # list to iterate
			return scan_parameter
		else: # (min, max, step)
			if len(scan_parameter) < 3: # (min, max) or (max, min) --> (min, max, 1)
				return range(min(scan_parameter), max(scan_parameter) + 1, 1)
			else:
				return range(scan_parameter[0], scan_parameter[1] + scan_parameter[2], scan_parameter[2])
	
	iter_cluster_size = create_iterable(scan_cluster_size)

	if scan_samples is None:
		scan_samples = (1, max(iter_cluster_size))
	iter_samples = create_iterable(scan_samples)

	if min(iter_samples) > min(iter_cluster_size) or max(iter_samples) > max(iter_cluster_size):
		raise ValueError("Only values of min_cluster_size equal to or larger than min_samples make sense! Therefore minimum/maximum of scan_samples must be smaller than minimum/maximum of scan_cluster_size.")

	print("scan range: %d <= min_samples <= %d | %d <= min_cluster_size <= %d" % (min(iter_samples), max(iter_samples), min(iter_cluster_size), max(iter_cluster_size)))

	builder = ak.ArrayBuilder()

	for min_samples in iter_samples:

		builder.begin_list()

		with tempfile.TemporaryDirectory() as cachedir:
			for min_cluster_size in iter_cluster_size:

				print("\nmin_samples = %d | min_cluster_size = %d" % (min_samples, min_cluster_size))

				_, labels, _ = do_clustering(data, min_cluster_size=min_cluster_size, min_samples=min_samples, memory=cachedir)

				_, counts = np.unique(labels, return_counts=True)
				if -1 not in labels:
					counts = np.insert(counts, 0, 0)

				builder.append(counts)

		builder.end_list()

	return builder.snapshot()


def plot_highdim(data, cluster_labels=None, cluster_probs=None, plot_type=None, fig=None, **kwargs):
	"""
	Plot clustered data in various forms.

	Parameters
	----------
	data
		Data points in arbitrary dimensions, array of shape (n_samples, n_dims).
	plot_type
		Type of plot to produce. '2d': detailled 2D plot with dendrogram; '3d': plotly 3D plot (3 dimensions only); 'corner': corner (triangle) plot; 'umap': UMAP embedding into two dimensions.
	fig
		matplotlib.figure object to re-use. If omitted, create new object with suitable figure size based on the number of dimensions.

	If the data has been clustered previously, provide the following to color-code clusters:
	cluster_labels
		Label of the associated cluster (integer) for all data points, array of length n_samples. Values of -1 denote unclustered data, 0 the first cluster, 1 the second etc.
	cluster_probs
		Probability of cluster association for all data points, array of length n_samples.
	
	kwargs will be passed to the final plot function.
	"""

	n_dim = data.shape[1]

	if n_dim < 2:
		raise ValueError("Cannot plot clusters of 1D data.")

	if plot_type == '3d' and n_dim != 3:
		plot_type = None
		print("Cannot plot 3D plot for dimensions other than 3. Reverting plot type to default.")

	if plot_type is None:
		if n_dim == 2:
			plot_type = '2d'
			print("Using detailed 2D plot type.")
		else:
			plot_type = 'corner'
			print("Using corner plot type.")

	if plot_type == '2d' and n_dim > 2:
		print("Using the first two dimensions for producing detailed 2D plot of %d dimensions." % n_dim)

	if n_dim > 30:
		warnings.warn("Do you really want to produce a corner plot of %d dimensions? Consider using a UMAP embedding instead." % n_dim)

	if cluster_labels is None:
		cluster_labels = np.full(len(data), 0)
	if cluster_probs is None:
		cluster_probs = np.full(len(data), 1)

	n_cluster = cluster_labels.max() + 1
	color_palette = sns.color_palette('dark', n_cluster)

	# grey for unclustered data
	color_palette.append((0.5, 0.5, 0.5))

	cluster_colors = [color_palette[x] for x in cluster_labels]
	cluster_colors_prob = [sns.desaturate(x, p) for x, p in zip(cluster_colors, cluster_probs)]


	# dummies
	def _loop(data_cluster, label):
		return
	def _finish():
		return


	if plot_type == 'corner':
		import corner
		import copy

		kwargs_corner = dict(
			bins = 100, 
			range = np.array([np.nanmin(data, axis=0), np.nanmax(data, axis=0)]).T, 
			plot_contours = False, 
			plot_density = False, 
			data_kwargs=dict(marker=',', alpha=.1)
		)
		kwargs_corner.update(kwargs)

		if fig is None:
			figsize = min(max(8, n_dim * 2), 24)
			fig = plt.figure(figsize=(figsize, figsize))

		# 1D histograms of entire distribution
		kwargs_ = dict(kwargs_corner)
		kwargs_.update(
			plot_datapoints=False, 
			color='black')
		corner.corner(
			no_nan(data), 
			fig=fig, 
			# fig=plt.figure(figsize=(figsize, figsize)), 
			**kwargs_);

		def _loop(data_cluster, label):

			# circumvent assertion in corner package by adding first data as many times as needed
			while data_cluster.shape[0] <= data_cluster.shape[1]:
				data_cluster = np.concatenate((data_cluster, data_cluster[:1]))
				print("Padded data in cluster %d to circumvent assertion error." % label)

			kwargs_ = dict(color=color_palette[label])
			kwargs_.update(copy.deepcopy(kwargs_corner)) # deepcopy: https://github.com/dfm/corner.py/issues/251
			# kwargs_ = copy.deepcopy(kwargs_corner)
			# kwargs_.update(
			# 	color=color_palette[label]
			# )
			corner.corner(
				data_cluster, 
				fig=plt.gcf(), 
				**kwargs_);

		def _finish():
			hists1d = np.array(plt.gcf().axes)
			hists1d = hists1d.reshape((n_dim, n_dim))
			hists1d = hists1d.diagonal()
			for ax in hists1d:
				ax.autoscale(axis='y')


	if plot_type == '3d':
		import plotly.graph_objects as go
		import matplotlib.colors as mpl_colors

		kwargs_plotly = dict(
			size = 1, 
			opacity = 1
		)
		kwargs_plotly.update(kwargs)

		markers = []
		def _loop(data_cluster, label):
			marker = go.Scatter3d(
				x=data_cluster.T[0], 
				y=data_cluster.T[1], 
				z=data_cluster.T[2], 
				marker=go.scatter3d.Marker(color=mpl_colors.to_hex(color_palette[label]), **kwargs_plotly), 
				mode='markers'
			)
			markers.append(marker)

		def _finish():
			layout = go.Layout(height=1000)
			fig = go.Figure(data=markers, layout=layout)
			fig.update_layout(scene_aspectmode='cube')
			fig.show()


	if plot_type == 'umap':
		import umap
		from matplotlib.patches import Patch

		kwargs_umap = dict(
			s = .1
		)
		kwargs_umap.update(kwargs)

		print("Running UMAP...")
		reducer = umap.UMAP()
		embedding = reducer.fit_transform(data)

		plt.scatter(embedding.T[0], embedding.T[1], c=cluster_colors_prob, **kwargs_umap)

		handles = []
		def _loop(data_cluster, label):
			handle = Patch(color=color_palette[label], label="cluster %d" % label)
			handles.append(handle)

		def _finish():
			plt.legend(handles=handles)


	if plot_type == '2d':
		raise NotImplementedError()


	# loop over clusters, independent of plot type
	for label in range(-1, n_cluster):
		mask = cluster_labels == label
		data_cluster = no_nan(data[mask])

		if len(data_cluster) == 0:
			continue

		_loop(data_cluster, label)

	_finish()
