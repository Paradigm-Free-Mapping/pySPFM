# pySPFM

A Python package for sparse hemodynamic deconvolution of fMRI data with **scikit-learn compatible estimators**.

pySPFM provides estimators for Paradigm Free Mapping (PFM) and related sparse deconvolution methods for fMRI analysis. The package follows scikit-learn conventions, making it easy to integrate with existing machine learning pipelines.

[![Latest Version](https://img.shields.io/pypi/v/pySPFM.svg)](https://pypi.python.org/pypi/pySPFM/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pySPFM.svg)](https://pypi.python.org/pypi/pySPFM/)
[![DOI](https://zenodo.org/badge/492450151.svg)](https://zenodo.org/badge/latestdoi/492450151)
[![License](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main.svg?style=shield)](https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main)
[![Documentation Status](https://readthedocs.org/projects/pyspfm/badge/?version=stable)](http://pyspfm.readthedocs.io/en/stable/?badge=stable)
[![codecov](https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM/branch/main/graph/badge.svg)](https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Features

- **scikit-learn compatible API**: Use familiar `fit()`, `transform()`, `fit_transform()` methods
- **Multiple deconvolution methods**:
  - `SparseDeconvolution`: Sparse Paradigm Free Mapping (SPFM) using LARS or FISTA
  - `LowRankPlusSparse`: SPLORA algorithm separating global and neuronal signals
  - `StabilitySelection`: Robust feature selection via bootstrap resampling
- **Multi-echo fMRI support**: Native support for multi-echo acquisitions
- **Flexible regularization**: Multiple lambda selection criteria (BIC, AIC, MAD, etc.)
- **Command-line interface**: Easy-to-use CLI for batch processing

## Installation

```bash
pip install pySPFM
```

For development:
```bash
git clone https://github.com/Paradigm-Free-Mapping/pySPFM.git
cd pySPFM
pip install -e ".[dev,tests]"
```

## Quick Start

### Python API

```python
from pySPFM import SparseDeconvolution
import numpy as np

# Load your fMRI data (n_timepoints, n_voxels)
X = np.random.randn(200, 1000)  # Example data

# Create and fit the model
model = SparseDeconvolution(tr=2.0, criterion='bic')
model.fit(X)

# Get the deconvolved activity-inducing signals
activity = model.coef_

# Get the fitted (reconstructed) signal
fitted = model.get_fitted_signal()

# Compute explained variance
score = model.score(X)
print(f"Explained variance: {score:.2%}")
```

### Low-Rank + Sparse Decomposition (SPLORA)

```python
from pySPFM import LowRankPlusSparse

model = LowRankPlusSparse(tr=2.0, eigval_threshold=0.1)
model.fit(X)

# Separate components
sparse_activity = model.coef_      # Neuronal activity
global_signal = model.low_rank_    # Global/structured component
```

### Command-Line Interface

```bash
# Sparse deconvolution
pySPFM sparse -i data.nii.gz -m mask.nii.gz -o output --tr 2.0

# Low-rank + sparse decomposition
pySPFM lowrank -i data.nii.gz -m mask.nii.gz -o output --tr 2.0

# Stability selection
pySPFM stability -i data.nii.gz -m mask.nii.gz -o output --tr 2.0 --n-surrogates 50
```

## Estimators

| Estimator | Description | Use Case |
|-----------|-------------|----------|
| `SparseDeconvolution` | Sparse PFM using LARS/FISTA | General deconvolution |
| `LowRankPlusSparse` | SPLORA algorithm | Separating global signals |
| `StabilitySelection` | Bootstrap-based selection | Robust feature detection |

## API Reference

All estimators follow scikit-learn conventions:

- `fit(X)`: Fit the model to data
- `transform(X)`: Return deconvolved signals
- `fit_transform(X)`: Fit and transform in one step
- `score(X)`: Return explained variance ratio
- `get_params()` / `set_params()`: Parameter access
- `clone()`: Create unfitted copy

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [tox](https://tox.wiki/) for testing.

### Running tests

```bash
# Current Python version
pytest pySPFM/tests/

# All supported versions (3.10, 3.11, 3.12)
tox

# Specific version
tox -e py310
```

## References

- Caballero-Gaudes, C., Moia, S., Panwar, P., Bandettini, P. A., & Gonzalez-Castillo, J. (2019). A deconvolution algorithm for multi-echo functional MRI: Multi-echo Sparse Paradigm Free Mapping. NeuroImage, 202, 116081–116081. https://doi.org/10.1016/j.neuroimage.2019.116081
- Caballero Gaudes, C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013). Paradigm free mapping with sparse regression automatically detects single-trial functional magnetic resonance imaging blood oxygenation level dependent responses. Human Brain Mapping. https://doi.org/10.1002/hbm.21452
- Karahanoǧlu, F. I., Caballero-Gaudes, C., Lazeyras, F., & Van De Ville, D. (2013). Total activation: FMRI deconvolution through spatio-temporal regularization. NeuroImage. https://doi.org/10.1016/j.neuroimage.2013.01.067
- Uruñuela, E., Bolton, T. A., Van De Ville, D., & Caballero-Gaudes, C. (2023). Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work. Aperture Neuro, 3, 1-25. https://doi.org/10.52294/001c.87574
