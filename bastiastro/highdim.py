"""
High-dimensional data analysis and clustering tools.

This module provides tools for exploratory data analysis in high-dimensional
spaces, including HDBSCAN clustering, visualization functions, and utilities
for astronomical data mining and parameter space exploration.
"""

# Python built-in imports
import warnings
from warnings import warn
import os
import tempfile
from concurrent import futures

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import sklearn as sk

# Package imports
from bastiastro.core import no_nan

# Type checking imports
from typing import TYPE_CHECKING, Any, Union, Optional, Iterable, Iterator
from numpy.typing import ArrayLike
from bastiastro import _lazy_import

# Type checking imports for all optional dependencies
if TYPE_CHECKING:
	import corner
	import hdbscan
	import plotly.graph_objects
	import awkward
	import h5py

warnings.filterwarnings("ignore", module="corner")

hdbscan_results = None


def plot_with_marginals(
	x: ArrayLike, 
	y: ArrayLike, 
	figsize: tuple[float, float] = (10, 10), 
	hist: bool = False, 
	log: bool = False, 
	SOI: Optional[dict] = None, 
	names_1RXS: Optional[ArrayLike] = None
) -> tuple[mpl.axes.Axes, callable, callable, callable]:
	"""
	Create a 2D plot with 1D histograms of marginals on the sides.
	
	This function creates a scatter plot or 2D histogram with marginal 
	histograms along the x and y axes. Adapted from matplotlib gallery.
	
	Parameters
	----------
	x
		X-axis data values.
	y
		Y-axis data values.
	figsize
		Figure size as (width, height).
	hist
		If True, create 2D histogram. If False, create scatter plot.
	log
		If True, use logarithmic scaling for histograms and color normalization.
	SOI
		Sources of interest dictionary mapping names to identifiers.
	names_1RXS
		Array of source names corresponding to x, y data points.
	
	Returns
	-------
	main_axis
		The main plotting axis for the 2D scatter plot or histogram.
	plot_function
		Function to update the main plot with new keyword arguments.
	hist_x_function
		Function to update the x-axis marginal histogram.
	hist_y_function
		Function to update the y-axis marginal histogram.
	"""
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
			ax.hist2d(x, y, cmap=mpl.colormaps['Blues'], **kwargs);
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


def do_cluster(
	data: ArrayLike, 
	plot_kwargs: dict = {}, 
	plot_dims: tuple[int, int] = (0, 1), 
	**kwargs
) -> 'hdbscan.HDBSCAN':
	"""
	Perform HDBSCAN clustering and create comprehensive visualization.
	
	This function performs HDBSCAN clustering on the provided data and creates
	a multi-panel visualization showing the clustering results, including the
	condensed tree and cluster projections.
	
	Parameters
	----------
	data
		Input data array of shape (n_features, n_samples) or (n_samples, n_features).
	plot_kwargs
		Additional keyword arguments for scatter plots.
	plot_dims
		Tuple specifying which dimensions to use for 2D plotting.
	**kwargs
		Additional keyword arguments passed to HDBSCAN constructor.
	
	Returns
	-------
	clusterer
		Fitted HDBSCAN clusterer object.
		
	Raises
	------
	ImportError
		If hdbscan package is not installed.
	"""
	_lazy_import('hdbscan')
	
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

def plot_tree() -> None:
	"""
	Plot the condensed tree from the most recent HDBSCAN clustering.
	
	This function displays the condensed tree visualization from the last
	HDBSCAN clustering performed with `do_cluster`. Must be called after
	`do_cluster` has been executed.
	
	Raises
	------
	RuntimeError
		If no HDBSCAN results are available (need to run `do_cluster` first).
	"""
	global hdbscan_results
	if hdbscan_results is None:
		raise RuntimeError("No HDBSCAN results available. Run do_cluster() first.")
	
	clusterer, color_palette = hdbscan_results
	
	plt.figure(figsize=(10, 6))
	clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=color_palette)


def cluster_corner(
	data: ArrayLike, 
	labels: Optional[list[str]] = None, 
	fig: Optional[mpl.figure.Figure] = None, 
	plot_kwargs: dict = {}, 
	corner_kwargs: dict = {}, 
	plot_3d: bool = False, 
	**kwargs
) -> Union[mpl.figure.Figure, 'plotly.graph_objects.Figure']:
	"""
	Create corner plot with HDBSCAN clustering visualization.
	
	This function performs HDBSCAN clustering on the input data and visualizes
	the results as either a corner plot (2D projections) or a 3D scatter plot.
	
	Parameters
	----------
	data
		Input data array of shape (n_samples, n_features).
	labels
		Labels for the data dimensions/features.
	fig
		Existing figure to plot on. If None, creates a new figure.
	plot_kwargs
		Additional keyword arguments for plotting functions.
	corner_kwargs
		Additional keyword arguments for the corner plot.
	plot_3d
		If True, create 3D scatter plot instead of corner plot.
	**kwargs
		Additional keyword arguments passed to HDBSCAN constructor.
	
	Returns
	-------
	figure
		Figure object (matplotlib Figure or plotly Figure depending on plot_3d).
		
	Raises
	------
	ImportError
		If required packages (hdbscan, corner, or plotly) are not installed.
	"""
	_lazy_import('hdbscan')

	if plot_3d:
		_lazy_import('plotly')
		import plotly.graph_objects as go
	else:
		_lazy_import('corner')

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
			warn(f"Padded data in cluster {label} to circumvent assertion error", stacklevel=2)

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


class HDBScanClustering:
	"""
	Perform unsupervised clustering with HDBSCAN algorithm.
	
	This class provides a comprehensive interface for HDBSCAN clustering with
	automatic data standardization, hyperparameter scanning capabilities, and
	result visualization. Can run single clustering executions or parameter scans.
	
	Parameters
	----------
	data
		Input data as Pandas DataFrame, NumPy array, or any array-like structure 
		of shape (n_samples, n_features).
	standardize
		Method for data standardization. Use 'standard' or 'robust' for the
		respective sklearn preprocessing scalers, or provide a custom scaler 
		object with fit/transform methods. Set to None or empty string to 
		disable scaling.
	"""

	def __init__(
		self, 
		data: Union[ArrayLike, pd.DataFrame], 
		standardize: Optional[Union[str, object]] = 'robust'
	) -> None:
		self._scan_mode = None

		self.hdbscan_args = dict(
			approx_min_span_tree=False
		)

		if standardize is None or standardize == '':
			self.data = data
		else:
			if standardize == 'robust':
				standardize = sk.preprocessing.RobustScaler()
			elif standardize == 'standard':
				standardize = sk.preprocessing.StandardScaler()
			
			if len(data.shape) == 1:  # one-dimensional array
				data = np.array(data).reshape(-1, 1)
			self.data = standardize.fit(data).transform(data)
			
			if isinstance(data, pd.DataFrame):
				self.data = pd.DataFrame(self.data, columns=data.columns)


	@classmethod
	def _get_cluster_counts(cls, cluster_labels: ArrayLike) -> np.ndarray:
		_, counts = np.unique(cluster_labels, return_counts=True)
		if -1 not in cluster_labels:  # unclustered points not guaranteed to exist
			counts = np.insert(counts, 0, 0)
		return counts

	@classmethod
	def _reorder_clusters(cls, cluster_labels: ArrayLike) -> np.ndarray:
		"""Re-order cluster labels from largest to smallest cluster."""

		uniq_labels, uniq_counts = np.unique(cluster_labels, return_counts=True)

		# ignore unclustered (label -1) if it exists
		if uniq_labels[0] == -1:
			uniq_labels = uniq_labels[1:]
			uniq_counts = uniq_counts[1:]

		uniq_indices = np.argsort(uniq_counts)
		uniq_indices = uniq_indices[::-1] # reverse

		cluster_labels_reordered = np.array(cluster_labels)
		cluster_labels_reordered[cluster_labels_reordered > -1] = -2 # dummy value
		for label_new, label_old in enumerate(uniq_labels[uniq_indices]):
			cluster_labels_reordered[cluster_labels == label_old] = label_new

		return cluster_labels_reordered


	def _scan(self, min_samples: int) -> tuple[int, Union[list, np.ndarray]]:

		if self._scan_mode == "summary":
			result = []
		elif self._scan_mode == "full":
			result = np.empty((len(self.iter_cluster_size), len(self.data)))  # for each value of min_cluster_size: cluster label for each data point

		print("scanning: min_samples = %d ..." % min_samples, flush=True)

		with tempfile.TemporaryDirectory() as cachedir:
			for i, min_cluster_size in enumerate(self.iter_cluster_size):

				_, cluster_labels, _ = self.cluster(verbosity=0, min_cluster_size=min_cluster_size, min_samples=min_samples, memory=cachedir)

				if self._scan_mode == "summary":
					result.append(self._get_cluster_counts(cluster_labels))

				elif self._scan_mode == "full":
					result[i] = cluster_labels

		return min_samples, result  # return min_samples again to ensure correct assignment of result in multiprocess environment


	def HyperparameterScan(
		self, 
		scan_cluster_size: Union[tuple[int, ...], Iterable[int]], 
		scan_samples: Optional[Union[tuple[int, ...], Iterable[int]]] = None, 
		n_processes: Optional[int] = None, 
		**kwargs
	) -> 'HDBScanClustering':
		"""
		Prepare a multiprocess-enabled scan through the HDBSCAN hyperparameters 
		min_cluster_size and min_samples. Run the scan with `.scan_full()` or `.scan_summary()`.

		Parameters
		----------
		scan_cluster_size
			Tuple (min, max, step) for scanning the min_cluster_size parameter of HDBSCAN. 
			If step size is omitted, will be set to 1. Alternatively, any iterable 
			containing the scan values.
		scan_samples
			Tuple (min, max, step) for scanning the min_samples parameter of HDBSCAN. 
			If step size is omitted, will be set to 1. If omitted entirely, will be 
			set to cover the entire parameter space, i.e. (1, max(scan_cluster_size)).
			Alternatively, any iterable containing the scan values.
		n_processes
			Number of parallel processes to use for the hyperparameter scan. 
			If None, defaults to max(1, cpu_count - 2).
		**kwargs
			Additional keyword arguments passed to `hdbscan.HDBSCAN`.

		Returns
		-------
		self
			Returns self for method chaining.
		"""

		def create_iterable(scan_parameter):
			if type(scan_parameter) != tuple: # list to iterate
				return scan_parameter
			else: # (min, max, step)
				if len(scan_parameter) < 3: # (min, max) or (max, min) --> (min, max, 1)
					return range(min(scan_parameter), max(scan_parameter) + 1, 1)
				else:
					return range(scan_parameter[0], scan_parameter[1] + scan_parameter[2], scan_parameter[2])
		
		self.iter_cluster_size = create_iterable(scan_cluster_size)

		if scan_samples is None:
			scan_samples = (1, max(self.iter_cluster_size))
		self.iter_samples = create_iterable(scan_samples)

		if n_processes is None:
			n_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 1

		if min(self.iter_samples) > min(self.iter_cluster_size) or max(self.iter_samples) > max(self.iter_cluster_size):
			raise ValueError("Only values of min_cluster_size equal to or larger than min_samples make sense! Therefore minimum/maximum of scan_samples must be smaller than minimum/maximum of scan_cluster_size.")

		print("Hyperparameter scan prepared with %d processes in parallel." % n_processes)
		print("scan range: %d <= min_samples <= %d | %d <= min_cluster_size <= %d" % (min(self.iter_samples), max(self.iter_samples), min(self.iter_cluster_size), max(self.iter_cluster_size)))

		self.hdbscan_args.update(kwargs)
		self.n_processes = n_processes
		self._scan_mode = ""
		return self


	def cluster(
		self, 
		verbosity: int = 2, 
		**kwargs
	) -> tuple[Union[ArrayLike, pd.DataFrame], np.ndarray, np.ndarray]:
		"""
		Run single execution of HDBSCAN clustering.
		
		Parameters
		----------
		verbosity
			Verbosity level of the logging: 0 = no logging, 2 = max logging.
		**kwargs
			Additional keyword arguments passed to `hdbscan.HDBSCAN`.
		
		Returns
		-------
		data
			The data object used for clustering. If scaling is applied (default), 
			it differs from the input; if scaling is disabled, it's identical.
		cluster_labels
			Cluster labels for all data points (length n_samples). Values of -1 
			denote unclustered data, 0 the first cluster, 1 the second, etc.
		cluster_probabilities
			Probability of cluster association for all data points (length n_samples).
		"""
		_lazy_import('hdbscan')
		kwargs_hdbscan = dict(self.hdbscan_args)
		kwargs_hdbscan.update(kwargs)
		
		clusterer = hdbscan.HDBSCAN(**kwargs_hdbscan).fit(self.data)
		n_cluster = clusterer.labels_.max() + 1
		cluster_labels = self._reorder_clusters(clusterer.labels_)

		def print_stats(label, cluster_name=None):
			if cluster_name is None:
				cluster_name = f"cluster {label + 1}"
			n_entries = np.sum(cluster_labels == label)
			fraction = n_entries / len(cluster_labels)
			print(f" {cluster_name}: {n_entries} entries ({fraction * 100 :.2f} %)")

		print("Found %d clusters" % n_cluster)
		if verbosity >= 2:
			print_stats(-1, "unclustered")
			for label in range(0, n_cluster):
				print_stats(label)

		return self.data, cluster_labels, clusterer.probabilities_


	def scan_summary(
		self, 
		return_range: bool = True
	) -> Union['awkward.Array', tuple['awkward.Array', tuple[Iterable, Iterable]]]:
		"""
		After preparation with `.HyperparameterScan()`, run a summary-only scan.
		
		This retains only the number of points per cluster for each hyperparameter scan point, 
		which is sufficient for later displaying the clustering as summary statistics.

		Parameters
		----------
		return_range
			Whether to additionally return the finally used hyperparameter scan range.

		Returns
		-------
		result
			Awkward array with min_samples/min_cluster_size scan points along first/second axis. 
			Third axis contains the numbers of points of each cluster, starting with unclustered points. 
			Because the number of clusters is variable, this third axis has variable length.
		range_info
			If return_range is True, tuple containing the scanned values of min_samples and min_cluster_size.
		"""

		if self._scan_mode is None:
			raise RuntimeError("Hyperparameter scan was not prepared yet! Run `HyperparameterScan()` with appropriate arguments first to instantiate an `HDBScanClustering` object, and then call this method of the object.")
		self._scan_mode = "summary"
		ak = _lazy_import('awkward')

		with futures.ProcessPoolExecutor(max_workers=self.n_processes) as pool:
			worker_results = pool.map(self._scan, self.iter_samples)

		# sort according to hyperparameter, and keep only counts
		counts_lists = [x[1] for x in sorted(worker_results, key=lambda x: x[0])]

		builder = ak.ArrayBuilder()
		[builder.append(l) for l in counts_lists]
		result = builder.snapshot()

		if return_range:
			result = result, (self.iter_samples, self.iter_cluster_size)
		return result


	@classmethod
	def iterate_scan(
		cls, 
		hdf5_file: str, 
		dataset_name: str = "HDBSCAN_scan", 
		indices_enumerated: bool = False
	) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
		"""
		Iterate over a hyperparameter scan result stored in an HDF5 file.

		This method yields cluster labels along with the corresponding cluster parameters
		(`min_cluster_size` and `min_samples`) and their grid indices (`x` and `y`).

		Parameters
		----------
		hdf5_file
			Path to the HDF5 file containing the HDBSCAN scan results.
		dataset_name
			Name of the dataset within the HDF5 file to read.
		indices_enumerated
			Whether to yield enumerated indices (0-based counting indices), 
			or the real parameter values (default).

		Yields
		------
		cluster_labels
			Array of cluster labels at the given grid position (hyperparameter combination).
		hyperparameters
			Either the hyperparameter values or the grid indices corresponding to the 
			current hyperparameters as determined by indices_enumerated.
		"""
		_lazy_import('h5py')

		with h5py.File(hdf5_file, 'r') as infile:

			dataset = infile[dataset_name]
			iter_samples = dataset.attrs["min_samples"]
			iter_cluster_size = dataset.attrs["min_cluster_size"]

			for y, min_samples in enumerate(iter_samples):
				for x, min_cluster_size in enumerate(iter_cluster_size):
					cluster_labels = dataset[y,x]
					if indices_enumerated:
						indices = (x, y)
					else:
						indices = (min_cluster_size, min_samples)
					yield cluster_labels, indices


	@classmethod
	def summarize_scan(
		cls, 
		input_file: str, 
		dataset_name: str = "HDBSCAN_scan", 
		return_range: bool = True
	) -> Union['awkward.Array', tuple['awkward.Array', tuple[Iterable, Iterable]]]:
		"""
		Summarize a full scan as if it were run with `.scan_summary` without running it again.

		Parameters
		----------
		input_file
			Path to an HDF5 file that contains a dataset with cluster label lists 
			for each hyperparameter scan point.
		dataset_name
			Name of the dataset in the HDF5 file.
		return_range
			Whether to additionally return the finally used hyperparameter scan range.

		Returns
		-------
		result
			Awkward array with min_samples/min_cluster_size scan points along first/second axis. 
			Third axis contains the numbers of points of each cluster, starting with unclustered points. 
			Because the number of clusters is variable, this third axis has variable length.
		range_info
			If return_range is True, tuple containing the scanned values of min_samples and min_cluster_size.
		"""
		_lazy_import('h5py')
		ak = _lazy_import('awkward')

		with h5py.File(input_file, 'r') as infile:
			dataset = infile[dataset_name]
			iter_samples = dataset.attrs["min_samples"]
			iter_cluster_size = dataset.attrs["min_cluster_size"]
		x_dim = len(iter_cluster_size)
		y_dim = len(iter_samples)

		counts_grid = [[None for _ in range(y_dim)] for _ in range(x_dim)]  # nested list

		for cluster_labels, (x, y) in cls.iterate_scan(input_file, dataset_name=dataset_name, indices_enumerated=True):
			counts_grid[x][y] = cls._get_cluster_counts(cluster_labels)

		result = ak.Array(counts_grid)

		if return_range:
			result = result, (iter_samples, iter_cluster_size)
		return result


	def scan_full(
		self, 
		output_file: str, 
		dataset_name: str = "HDBSCAN_scan"
	) -> None:
		"""
		After preparation with `.HyperparameterScan()`, run a full scan. 
		
		This saves the entire clustering information for each hyperparameter scan point 
		in an HDF5 file. The memory consumption is significantly larger than for `.scan_summary()`.

		Parameters
		----------
		output_file
			Path to an HDF5 file (existing or not) where the resulting cluster label lists 
			are saved for each hyperparameter scan point.
		dataset_name
			Name of the dataset to add to the HDF5 file.
		"""

		if self._scan_mode is None:
			raise RuntimeError("Hyperparameter scan was not prepared yet! Run `HyperparameterScan()` with appropriate arguments first to instantiate an `HDBScanClustering` object, and then call this method of the object.")
		self._scan_mode = "full"
		_lazy_import('h5py')

		with h5py.File(output_file, 'a') as outfile:

			dataset = outfile.create_dataset(dataset_name, shape=(len(self.iter_samples), len(self.iter_cluster_size), len(self.data)), dtype='int')
			dataset.attrs["min_samples"] = self.iter_samples
			dataset.attrs["min_cluster_size"] = self.iter_cluster_size

			with futures.ProcessPoolExecutor(max_workers=self.n_processes) as pool:
				workers = [pool.submit(self._scan, min_samples) for min_samples in self.iter_samples]
				for worker in futures.as_completed(workers):  # iterates as soon as a worker is finished
					min_samples, cluster_labels = worker.result()
					index = self.iter_samples.index(min_samples)
					dataset[index,:] = cluster_labels

	def plot_scan(
		self, 
		cluster_scan: ArrayLike, 
		ranges: Union[tuple[int, int, int, int], tuple[ArrayLike, ArrayLike]], 
		trunc_clusters: int = 0, 
		**kwargs
	) -> mpl.figure.Figure:
		"""
		Plot the results of a HDBSCAN hyperparameter scan as a number of summary statistics.

		Parameters
		----------
		cluster_scan
			The result of a summarized hyperparameter scan for clustering 
			(i.e. only cluster counts, not cluster labels for all points).
		ranges
			Min and max values of min_cluster_size and of min_samples as 
			(min_cluster_size_min, min_cluster_size_max, min_samples_min, min_samples_max).
			Alternatively, the hyperparameter scan values as (iter_samples, iter_cluster_size) 
			from which the min and max values are taken.
		trunc_clusters
			Maximum number of clusters to display. Truncates numbers larger than this value. 
			0 disables truncation.
		**kwargs
			Additional keyword arguments passed to `matplotlib.pyplot.imshow`.

		Returns
		-------
		figure
			The generated matplotlib Figure object containing the plots.
		"""
		ak = _lazy_import('awkward')

		kwargs_imshow = dict(
			aspect = 'auto',
			interpolation = 'none',
			origin='lower',
			cmap='Blues'
		)

		if len(ranges) == 2:  # scan values provided of both hyperparameters
			iter_samples, iter_cluster_size = ranges
			extent = min(iter_cluster_size), max(iter_cluster_size), min(iter_samples), max(iter_samples)
		else:  # already min and max values provided
			extent = ranges

		extent = np.array(extent)

		# fence post problem
		extent[1] += 1
		extent[3] += 1

		# center bins at the respective values
		extent = np.array(extent) - 0.5

		kwargs_imshow["extent"] = extent
		kwargs_imshow.update(kwargs)

		# I trust awkward functions this far...
		number_clusters = np.array(ak.count(cluster_scan, axis=2)) - 1  # do not count unclustered
		cluster_scan_sorted = ak.sort(cluster_scan[:,:,1:], axis=2, ascending=False)

		shape = number_clusters.shape
		norm = len(self.data)

		size_max_cluster = np.full(shape, 0, dtype=float)
		size_secmax_cluster = np.full(shape, 0, dtype=float)

		# ... but no further: long live expressive for loops
		for i in range(shape[0]):
			for j in range(shape[1]):
				try:
					size_max_cluster[i,j] = cluster_scan_sorted[i,j][0]
					size_secmax_cluster[i,j] = cluster_scan_sorted[i,j][1]
				except IndexError:
					pass

		size_max_cluster /= norm
		size_secmax_cluster /= norm

		size_unclustered = cluster_scan[:,:,0] / norm

		fig, axes = plt.subplots(1, 4, sharey=True, figsize=(20, 5))
		axes[0].set_ylabel("HDBSCAN min_samples")
		[ax.set_xlabel("HDBSCAN min_cluster_size") for ax in axes]

		def plot_panel(ax, data, title):
			im = ax.imshow(data, **kwargs_imshow)
			plt.colorbar(im, ax=ax)

			# diagonal line
			xvals = np.linspace(extent[0], extent[1])
			ax.plot(xvals, xvals, color='black', linewidth=0.5)

			ax.set_title(title)

		# plots 2 - 4
		plot_panel(axes[1], size_max_cluster, "fraction of largest cluster")
		plot_panel(axes[2], size_secmax_cluster, "fraction of second-largest cluster")
		plot_panel(axes[3], size_unclustered, "fraction of unclustered points")

		# plot 1
		if trunc_clusters == 0:
			trunc_clusters = np.max(number_clusters)

		cmap = kwargs_imshow['cmap']
		if not isinstance(cmap, mpl.colors.Colormap):
			cmap = mpl.colormaps[cmap]
		cmap.set_over('black')

		kwargs_imshow.update(
			cmap=cmap,
			vmin=np.min(number_clusters),
			vmax=trunc_clusters
		)
		plot_panel(axes[0], number_clusters, "number of clusters")

		# write cluster numbers as text if plot not too large
		if ((extent[1] - extent[0]) < 20) and ((extent[3] - extent[2]) < 20):

			# revert bin centering for text plotting
			extent = (extent + 0.5).astype(int)

			for y, y_val in enumerate(range(extent[2], extent[3])):
				for x, x_val in enumerate(range(extent[0], extent[1])):
					z_val = number_clusters[y, x]
					if z_val > trunc_clusters:
						color = "white"
					else:
						color = "black"
					fig.axes[0].text(x_val, y_val, str(z_val),
							color=color, ha="center", va="center", fontsize=14)

		fig.set_tight_layout(True)

		return fig


def do_clustering(
	data: Union[ArrayLike, pd.DataFrame], 
	verbosity: int = 2, 
	**kwargs
) -> tuple[Union[ArrayLike, pd.DataFrame], np.ndarray, np.ndarray]:
	"""Deprecated: Use `HDBScanClustering.cluster` instead."""
	return HDBScanClustering(data).cluster(verbosity, **kwargs)


def do_clustering_scan(
	data: Union[ArrayLike, pd.DataFrame], 
	scan_cluster_size: tuple[int, ...], 
	scan_samples: Optional[tuple[int, ...]] = None, 
	n_processes: Optional[int] = None, 
	return_range: bool = False
) -> Union['awkward.Array', tuple['awkward.Array', tuple[int, int, int, int]]]:
	"""Deprecated: Use `HDBScanClustering.HyperparameterScan` instead."""

	if n_processes is None:
		n_processes = max(1, os.cpu_count() - 2) if os.cpu_count() else 1
		
	scanner = HDBScanClustering(data).HyperparameterScan(scan_cluster_size, scan_samples, n_processes)
	result = scanner.scan_summary(return_range)
	
	# old meaning of return_range
	if return_range:
		iter_samples, iter_cluster_size = result[1]
		result = result[0], (
			min(iter_cluster_size), max(iter_cluster_size),
			min(iter_samples), max(iter_samples),
		)

	return result


def plot_highdim(
	data: ArrayLike, 
	cluster_labels: Optional[ArrayLike] = None, 
	cluster_probs: Optional[ArrayLike] = None, 
	plot_type: Optional[str] = None, 
	fig: Optional[mpl.figure.Figure] = None, 
	ranges: Optional[Union[tuple[float, float], float]] = None, 
	**kwargs
) -> None:
	"""
	Plot clustered data in various visualization forms.

	This function creates visualizations of high-dimensional clustered data using different 
	plot types including 2D projections, 3D scatter plots, corner plots, and UMAP embeddings.

	Parameters
	----------
	data
		Data points array of shape (n_samples, n_dims).
	cluster_labels
		Cluster labels for data points. Integer array of length n_samples where 
		-1 denotes unclustered data, 0 the first cluster, 1 the second, etc.
	cluster_probs
		Probability of cluster association for all data points.
	plot_type
		Type of plot: '2d' (detailed 2D with dendrogram), '3d' (plotly 3D plot),
		'corner' (corner/triangle plot), 'umap' (UMAP embedding to 2D).
	fig
		Existing matplotlib Figure object to reuse. If None, creates new figure 
		with suitable size based on dimensions.
	ranges
		Data range limits for axes. Either (min, max) tuple or single value 
		for symmetric range [-ranges, ranges].
	**kwargs
		Additional keyword arguments passed to the plotting function.
	
	Raises
	------
	ValueError
		If data has less than 2 dimensions or incompatible plot_type.
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
		warn(f"Do you really want to produce a corner plot of {n_dim} dimensions? Consider using a UMAP embedding instead.", stacklevel=2)

	if cluster_labels is None:
		cluster_labels = np.full(len(data), 0)
	if cluster_probs is None:
		cluster_probs = np.full(len(data), 1)

	# re-order cluster labels from largest to smallest cluster

	uniq_labels, uniq_counts = np.unique(cluster_labels, return_counts=True)
	# ignore unclustered (label -1) if it exists
	if uniq_labels[0] == -1:
		uniq_labels = uniq_labels[1:]
		uniq_counts = uniq_counts[1:]
	uniq_indices = np.argsort(uniq_counts)
	uniq_indices = uniq_indices[::-1] # reverse

	cluster_labels_reordered = np.array(cluster_labels)
	cluster_labels_reordered[cluster_labels_reordered > -1] = -2 # dummy value
	for label_new, label_old in enumerate(uniq_labels[uniq_indices]):
		cluster_labels_reordered[cluster_labels == label_old] = label_new
	cluster_labels = cluster_labels_reordered

	n_cluster = cluster_labels.max() + 1
	color_palette = sns.color_palette('bright', n_cluster)

	# grey for unclustered data (corresponds to label -1 --> append at end of palette)
	color_palette.append((0.5, 0.5, 0.5))

	cluster_colors = [color_palette[x] for x in cluster_labels]
	cluster_colors_prob = [sns.desaturate(x, p) for x, p in zip(cluster_colors, cluster_probs)]


	# dummies
	def _loop(data_cluster, label):
		return
	def _finish(ranges):
		return


	if plot_type == 'corner':
		_lazy_import('corner')
		import copy

		kwargs_corner = dict(
			bins = 100, 
			range = np.array([np.nanmin(data, axis=0), np.nanmax(data, axis=0)]).T, 
			plot_contours = False, 
			plot_density = False, 
			data_kwargs=dict(marker=',', alpha=.1)
		)
		if type(data) is pd.DataFrame:
			kwargs_corner['labels'] = data.columns
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
				warn(f"Padded data in cluster {label} to circumvent assertion error", stacklevel=3)

			kwargs_ = dict(color=color_palette[label])
			kwargs_.update(copy.deepcopy(kwargs_corner)) # deepcopy: https://github.com/dfm/corner.py/issues/251
			# kwargs_ = copy.deepcopy(kwargs_corner)
			# kwargs_.update(
			# 	color=color_palette[label]
			# )
			corner.corner(
				data_cluster, 
				fig=fig, 
				# fig=plt.gcf(), 
				**kwargs_);

		def _finish(ranges):
			[ax.autoscale(axis='y') for ax in get_corner_axes('diag', fig=fig)]
			if ranges is not None:
				try:
					iter(ranges)
				except TypeError:  # not iterable
					ranges = (-ranges, ranges)
				for ax in get_corner_axes('corner', fig=fig):
					ax.set_xlim(ranges[0], ranges[1])
					ax.set_ylim(ranges[0], ranges[1])
				for ax in get_corner_axes('diag', fig=fig):
					ax.set_xlim(ranges[0], ranges[1])


	if plot_type == '3d':
		_lazy_import('plotly')
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

		def _finish(ranges):
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

		def _finish(ranges):
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

	_finish(ranges)


def get_corner_axes(
	which: str = 'all', 
	fig: Optional[mpl.figure.Figure] = None
) -> list[mpl.axes.Axes]:
	"""
	Return specific axes of a corner plot as a list (e.g. to perform some operation on them).
	
	Axes with the label "NOCORNER" are ignored.

	Parameters
	----------
	which
		Which axes to return: 'all' (all fixed axes, i.e. all 2D and 1D distributions),
		'corner' (lower corner, i.e. all 2D distributions), 'diag' (diagonal, i.e. all 1D distributions).
	fig
		Figure to use. If omitted, will use the current figure.

	Returns
	-------
	axes_list
		A list of the requested axes objects.
	"""
	if fig is None:
		fig = plt.gcf()

	# remove axes with magic label
	axes = [ax for ax in fig.axes if ax.get_label() != "NOCORNER"]

	n_ax = int(np.floor(np.sqrt(len(axes))))
	axes_grid = np.array(axes).reshape((n_ax, n_ax))

	if which == 'all':
		toreturn = np.tril(axes_grid, k=0)
		toreturn = toreturn[toreturn != 0]
	elif which == 'corner':
		toreturn = np.tril(axes_grid, k=-1)
		toreturn = toreturn[toreturn != 0]
	elif which == 'diag':
		toreturn = axes_grid.diagonal()
	else:
		raise ValueError(f"Invalid value for 'which': {which}. Must be 'all', 'corner', or 'diag'.")

	return list(toreturn.flatten())


def plot_hyperparameter_scan(
	data: ArrayLike, 
	cluster_scan: ArrayLike, 
	ranges: Union[tuple[int, int, int, int], tuple[Iterable, Iterable]], 
	n_cluster_trunc: int = 10, 
	**kwargs
) -> mpl.figure.Figure:
	"""
	Plot the results of a HDBSCAN hyperparameter scan as a number of summary statistics.

	Parameters
	----------
	data
		The data that has been clustered.
	cluster_scan
		The result of a summarized hyperparameter scan for clustering 
		(i.e. only cluster counts, not cluster labels for all points).
	ranges
		Min and max values of min_cluster_size and of min_samples as 
		(min_cluster_size_min, min_cluster_size_max, min_samples_min, min_samples_max).
		Alternatively, the hyperparameter scan values as (iter_samples, iter_cluster_size) 
		from which the min and max values are taken.
	n_cluster_trunc
		Maximum number of clusters to display.
	**kwargs
		Additional keyword arguments passed to `matplotlib.pyplot.imshow`.

	Returns
	-------
	figure
		The generated matplotlib Figure object containing the plots.
	"""
	ak = _lazy_import('awkward')
	kwargs_imshow = dict(
		aspect = 'auto',
		interpolation = 'none',
		origin='lower',
		cmap='Blues'
	)

	if len(ranges) == 2:  # scan values provided of both hyperparameters
		iter_samples, iter_cluster_size = ranges
		kwargs_imshow["extent"] = min(iter_cluster_size), max(iter_cluster_size), min(iter_samples), max(iter_samples)
	else:  # already min and max values provided
		kwargs_imshow["extent"] = ranges

	kwargs_imshow["extent"] = np.array(kwargs_imshow["extent"])

	# fence post problem
	kwargs_imshow["extent"][1] += 1
	kwargs_imshow["extent"][3] += 1

	# center bins at the respective values
	kwargs_imshow["extent"] = np.array(kwargs_imshow["extent"]) - 0.5

	kwargs_imshow.update(kwargs)

	# I trust awkward functions this far...
	number_clusters = np.array(ak.count(cluster_scan, axis=2)) - 1  # do not count unclustered
	number_clusters[number_clusters > n_cluster_trunc] = n_cluster_trunc + 1
	cluster_scan_sorted = ak.sort(cluster_scan[:,:,1:], axis=2, ascending=False)

	shape = number_clusters.shape
	norm = len(data)

	size_max_cluster = np.full(shape, 0, dtype=float)
	size_secmax_cluster = np.full(shape, 0, dtype=float)

	# ... but no further: long live expressive for loops
	for i in range(shape[0]):
		for j in range(shape[1]):
			try:
				size_max_cluster[i,j] = cluster_scan_sorted[i,j][0]
				size_secmax_cluster[i,j] = cluster_scan_sorted[i,j][1]
			except IndexError:
				pass

	size_max_cluster /= norm
	size_secmax_cluster /= norm

	size_unclustered = cluster_scan[:,:,0] / norm


	fig, axes = plt.subplots(1, 4, sharey=True, figsize=(20, 5))
	axes[0].set_ylabel("HDBSCAN min_samples")
	[ax.set_xlabel("HDBSCAN min_cluster_size") for ax in axes]

	im = axes[0].imshow(number_clusters, **kwargs_imshow)
	plt.colorbar(im, ax=axes[0])
	axes[0].set_title("number of clusters")

	im = axes[1].imshow(size_max_cluster, **kwargs_imshow)
	plt.colorbar(im, ax=axes[1])
	axes[1].set_title("rel. size of largest cluster")
	
	im = axes[2].imshow(size_secmax_cluster, **kwargs_imshow)
	plt.colorbar(im, ax=axes[2])
	axes[2].set_title("rel. size of second-largest cluster")
	
	im = axes[3].imshow(size_unclustered, **kwargs_imshow)
	plt.colorbar(im, ax=axes[3])
	axes[3].set_title("rel. size of unclustered points")

	fig.set_tight_layout(True)

	return fig


def plot_pairgrid(
	df_data: pd.DataFrame, 
	df_mask: Optional[Union[pd.DataFrame, list[pd.DataFrame]]] = None, 
	label: Optional[Union[str, list[str]]] = None, 
	color: Optional[Union[str, list[str]]] = None, 
	marker: Optional[Union[str, list[str]]] = None, 
	scatter_kws: dict = {}, 
	**pairplot_kws
) -> 'sns.axisgrid.PairGrid':
	"""
	Make a seaborn pairplot with two simultaneous plotting styles, optionally 
	with highlighting points according to one or more masks.
	
	Parameters
	----------
	df_data
		DataFrame with the data to plot.
	df_mask
		DataFrame or list of DataFrames where True values indicate points to highlight.
	label
		Label(s) of the highlighted points to be put into a legend.
	color
		Color(s) for the highlighted points.
	marker
		Marker style(s) for the highlighted points.
	scatter_kws
		Keyword arguments for `seaborn.scatterplot`.
	**pairplot_kws
		Keyword arguments for `seaborn.pairplot`.

	Returns
	-------
	pairgrid
		The seaborn PairGrid instance.
	"""

	# default colors can be changed in case of several masks
	color_base = sns.color_palette()[0]
	color_small_scatter = color_base

	if df_mask is not None:

		# Handle single mask input
		if isinstance(df_mask, (list, tuple)):
			n_masks = len(df_mask)
			if label is None:
				label = (None,) * n_masks
			highlight_color = sns.husl_palette(n_masks, s=1, l=0.6)
			highlight_color = sns.color_palette("tab10", n_masks)
			highlight_marker = [mpl.lines.Line2D.filled_markers[i // 10 + 1] for i in range(n_masks)]
			color_base = "lightgray"
			color_small_scatter = "black"
		else:
			n_masks = 1
			df_mask = (df_mask,)
			label = (label,)
			highlight_color = ("red",)
			highlight_marker = ('o',)

		if color is not None:
			highlight_color = color
		if marker is not None:
			highlight_marker = marker

	# Set default keyword arguments
	_pairplot_kws = dict(
		markers='.', 
		plot_kws=dict(s=10, color=color_small_scatter), 
		diag_kws=dict(color=color_base)
	)
	_pairplot_kws.update(pairplot_kws)

	# Define base scatter plot settings with explicit color and size
	scatter_base_kws = dict(color=color_base, s=20, zorder=1)
	scatter_base_kws.update(scatter_kws)

	# Generate the base pairplot
	pg = sns.pairplot(df_data, **_pairplot_kws)
	pg.map_upper(sns.scatterplot, **scatter_base_kws)
	pg.map_upper(sns.kdeplot, levels=4, color="black", linewidths=0.5, zorder=100)
	pg.map_lower(sns.kdeplot, levels=4, color="black", linewidths=0.5)

	if df_mask is not None:

		# Highlighted scatter settings with higher z-order
		scatter_hl_kws = dict(scatter_base_kws)
		scatter_hl_kws.update(zorder=10)

		# Loop through masks for custom highlighting ...
		for k, (_df_mask, _label, _highlight_color, _highlight_marker) in enumerate(zip(df_mask, label, highlight_color, highlight_marker)):

			data = df_data[_df_mask]
			scatter_hl_kws['label'] = _label
			scatter_hl_kws['color'] = _highlight_color
			scatter_hl_kws['marker'] = _highlight_marker

			# ... in each subplot
			for i, j in zip(*np.triu_indices_from(pg.axes, k=1)):
				ax = pg.axes[i, j]

				# Check if there are points to plot
				if data.iloc[:, [j, i]].notna().all(axis=1).any():

					sns.scatterplot(
						x=data.iloc[:, j],
						y=data.iloc[:, i],
						ax=ax,
						**scatter_hl_kws
					)

					# Only add the label the first time for each mask
					scatter_hl_kws['label'] = None

				# Make legend more compact
				if k + 1 == n_masks:  # last iteration over masks
					ax.set_zorder(1)  # ensures that overflowing legend is not covered
					try:
						sns.move_legend(ax, 'best', fontsize='small', 
							frameon=False, handletextpad=0, labelspacing=0, borderpad=0, borderaxespad=0, handlelength=1) # mpl.rcParams['legend.handleheight']
						ax.get_legend().zorder = 100
					except ValueError as e:
						if "no legend attached" not in str(e):
							raise e

	return pg
