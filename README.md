# PhD Research Repository

**Sebastian Schindler's PhD research on neutrino emission from Active Galactic Nuclei**

This repository contains the complete research environment for my PhD project, including the `bastiastro` Python package, analysis notebooks, and supporting materials.

## Research Context

This package supports research investigating neutrino emission from Active Galactic Nuclei (AGN) using data-driven source selection methods. The work addresses a key challenge in neutrino astronomy: traditional AGN classifications based on historical choices of spectral properties may not be optimal for identifying neutrino sources.

The research employs machine learning approaches to define source selections that emphasize intrinsic physical properties rather than conventional AGN classifications. Using multi-wavelength data (X-ray, infrared, optical), high-dimensional clustering algorithms like HDBSCAN identify potential neutrino sources with similar physical properties to confirmed detections like NGC 1068 / M 77 and TXS 0506+056.

## Repository Structure

```
PhD/
├── bastiastro/         # Main Python package
├── ipynb/              # Jupyter notebooks and analysis
│   └── tools.py        # Backwards compatibility for notebooks
├── README.md           # This file
└── pyproject.toml      # Package configuration
```

## 📦 bastiastro Package

**Research environment bootstrap for astronomy and neutrino physics**

A Python package providing tools and utilities for astronomical data analysis, developed as part of my PhD research into neutrino emission from Active Galactic Nuclei (AGN) with the IceCube detector.

### Features

**🚀 Research Environment Bootstrap**
```python
from bastiastro import *
```

This simple import provides a complete workspace setup: numpy, pandas, matplotlib, as well as a number custom tools.

**📊 Astronomical Data Analysis**
- Sky plotting and coordinate transformations (`plt_skyplot`)
- Multi-wavelength catalog analysis and cross-matching
- High-dimensional clustering for source classification (`HDBScanClustering`)

**🔬 Statistical Tools**
- HDBSCAN clustering with hyperparameter optimization
- Corner plots and multi-dimensional visualization (`plot_highdim`)
- Data cleaning and preprocessing utilities (`no_nan`, `pickle`/`unpickle`)

### Installation

This is a PhD research repository containing the `bastiastro` package. To install:

```bash
# Clone the PhD research repository
git clone https://github.com/sebastian-schindler/PhD.git
cd PhD

# Install the bastiastro package in development mode
pip install -e .

# Optional: Install with high-dimensional analysis tools
pip install -e ".[highdim]"
```

**Note**: Existing notebooks in `ipynb/` can continue using `from tools import *` for backwards compatibility.

### Usage

```python
# Complete workspace setup
from bastiastro import *

# Multi-wavelength AGN analysis
catalog = unpickle("catalog_2RXS_AllWISE.pkl")
observables = ['W1', 'W2', 'W3', 'log(x-ray)', 'W1-W2', 'W2-W3']

# High-dimensional clustering for source classification
from bastiastro.highdim import HDBScanClustering
clusterer = HDBScanClustering(catalog[observables])
data, labels, probs = clusterer.cluster(min_cluster_size=50)

# Visualize results with corner plots
plot_highdim(data, labels, probs, ranges=2.5)

# Sky distribution analysis
plt_skyplot(catalog['RA'], catalog['DEC'], galactic=True, galaxy=True)
plt.title('AGN Sample Distribution')
```

### Development

Professional Python package with modular design and comprehensive documentation. See [CODING_STANDARDS.md](CODING_STANDARDS.md) for development guidelines.

**Package Structure:**
- `bastiastro.core` - Data utilities and preprocessing
- `bastiastro.plotting` - Visualization and sky plotting
- `bastiastro.astronomy` - Astronomical catalogs and object resolution
- `bastiastro.highdim` - High-dimensional analysis and clustering

## Academic Context

**PhD Research**: Search for neutrino emission from Active Galactic Nuclei with the [IceCube Neutrino Observatory](https://github.com/icecube) using a machine-learning-based source selection

**Institution**: [Erlangen Centre for Astroparticle Physics](https://ecap.nat.fau.de/) at the [Friedrich-Alexander-Universität Erlangen-Nürnberg](https://www.fau.eu/)

**Author**: Sebastian Schindler (sebastian.schindler@fau.de)
