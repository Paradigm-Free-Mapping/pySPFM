# pySPFM

The Python version of AFNI's [3dPFM](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dPFM.html) and [3dMEPFM](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dMEPFM.html) with some extra features like the addition of a spatial regularization similar to the one used by [Total Activation](https://miplab.epfl.ch/index.php/software/total-activation).

[![Latest Version](https://img.shields.io/pypi/v/pySPFM.svg)](https://pypi.python.org/pypi/pySPFM/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pySPFM.svg)](https://pypi.python.org/pypi/pySPFM/)
[![DOI](https://zenodo.org/badge/492450151.svg)](https://zenodo.org/badge/latestdoi/492450151)
[![License](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main.svg?style=shield)](https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main)
[![Documentation Status](https://readthedocs.org/projects/pyspfm/badge/?version=stable)](http://pyspfm.readthedocs.io/en/stable/?badge=stable)
[![codecov](https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM/branch/main/graph/badge.svg)](https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Development

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management and [tox](https://tox.wiki/) for testing across multiple Python versions.

### Setting up a development environment

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Install the package in editable mode with test dependencies:
   ```bash
   uv pip install -e .[tests]
   ```

### Running tests

Run tests for your current Python version:
```bash
pytest pySPFM/tests/
```

Run tests across all supported Python versions (3.10, 3.11, 3.12) using tox:
```bash
tox
```

Run tests for a specific Python version:
```bash
tox -e py310  # For Python 3.10
```

Run linting checks:
```bash
tox -e lint
```

### Benefits of uv

- **Fast**: 10-100x faster than pip for dependency resolution and installation
- **Deterministic**: `uv.lock` ensures reproducible installations across all environments
- **Reliable**: Resolves dependencies consistently

## References

- Caballero-Gaudes, C., Moia, S., Panwar, P., Bandettini, P. A., & Gonzalez-Castillo, J. (2019). A deconvolution algorithm for multi-echo functional MRI: Multi-echo Sparse Paradigm Free Mapping. NeuroImage, 202, 116081–116081. https://doi.org/10.1016/j.neuroimage.2019.116081
- Caballero Gaudes, C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013). Paradigm free mapping with sparse regression automatically detects single-trial functional magnetic resonance imaging blood oxygenation level dependent responses. Human Brain Mapping. https://doi.org/10.1002/hbm.21452
- Karahanoǧlu, F. I., Caballero-Gaudes, C., Lazeyras, F., & Van De Ville, D. (2013). Total activation: FMRI deconvolution through spatio-temporal regularization. NeuroImage. https://doi.org/10.1016/j.neuroimage.2013.01.067
- Uruñuela, E., Bolton, T. A., Van De Ville, D., & Caballero-Gaudes, C. (2023). Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work. Aperture Neuro, 3, 1-25. https://doi.org/10.52294/001c.87574
