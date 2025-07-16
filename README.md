# bastiastro

**Research environment bootstrap for astronomy and neutrino physics**

A Python package providing tools and utilities for astronomical data analysis, developed as part of my PhD research into neutrino emission from Active Galactic Nuclei (AGN) with the IceCube detector.

## Research Context

This package supports research investigating neutrino emission from Active Galactic Nuclei (AGN) using data-driven source selection methods. The work addresses a key challenge in neutrino astronomy: traditional AGN classifications based on historical spectral properties may not be optimal for identifying neutrino sources.

The research employs machine learning approaches to define source selections that emphasize intrinsic physical properties rather than conventional AGN classifications. Using multi-wavelength data (X-ray, infrared, optical), high-dimensional clustering algorithms like HDBSCAN identify potential neutrino sources with similar physical properties to confirmed detections like NGC 1068 and TXS 0506+056.

## Features

**🚀 Research Environment Bootstrap**
```python
from tools import *  # Complete workspace setup: numpy, pandas, matplotlib + custom tools
```

**📊 Astronomical Data Analysis**
- Sky plotting and coordinate transformations
- Multi-wavelength catalog analysis and cross-matching
- High-dimensional clustering for source classification

**🔬 Statistical Tools**
- HDBSCAN clustering with hyperparameter optimization
- Corner plots and multi-dimensional visualization
- Data cleaning and preprocessing utilities

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bastiastro.git
cd bastiastro

# Install in development mode
pip install -e .
```

## Usage

```python
# Complete workspace setup
from tools import *

# Multi-wavelength AGN analysis
catalog = unpickle("catalog_2RXS_AllWISE.pkl")
observables = ['W1', 'W2', 'W3', 'log(x-ray)', 'W1-W2', 'W2-W3']

# High-dimensional clustering for source classification
from tools.highdim import HDBScanClustering
clusterer = HDBScanClustering(catalog[observables])
data, labels, probs = clusterer.cluster(min_cluster_size=50)

# Visualize results with corner plots
plot_highdim(data, labels, probs, ranges=2.5)

# Sky distribution analysis
plt_skyplot(catalog['RA'], catalog['DEC'], galactic=True, galaxy=True)
plt.title('AGN Sample Distribution')
```

## Development

Professional Python package with modular design and useful documentation. See [CODING_STANDARDS.md](CODING_STANDARDS.md) for guidelines.

## Academic Context

**PhD Research**: Search for neutrino emission from Active Galactic Nuclei with the [IceCube Neutrino Observatory](https://github.com/icecube) using a machine-learning-based source selection
**Institution**: [Erlangen Centre for Astroparticle Physics](https://ecap.nat.fau.de/) at the [Friedrich-Alexander-Universität Erlangen-Nürnberg](https://www.fau.eu/)
**Author**: Sebastian Schindler (sebastian.schindler@fau.de)
