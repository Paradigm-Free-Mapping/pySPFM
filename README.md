# pySPFM

The Python version of AFNI's [3dPFM](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dPFM.html) and [3dMEPFM](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dMEPFM.html) with some extra features like the addition of a spatial regularization similar to the one used by [Total Activation](https://miplab.epfl.ch/index.php/software/total-activation).

[![Latest Version](https://img.shields.io/pypi/v/pySPFM.svg)](https://pypi.python.org/pypi/pySPFM/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pySPFM.svg)](https://pypi.python.org/pypi/pySPFM/)
[![DOI](https://zenodo.org/badge/492450151.svg)](https://zenodo.org/badge/latestdoi/492450151)
[![License](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/eurunuela/pySPFM/tree/main.svg?style=shield)](https://circleci.com/gh/eurunuela/pySPFM/tree/main)
[![Documentation Status](https://readthedocs.org/projects/pyspfm/badge/?version=latest)](http://pyspfm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/eurunuela/pySPFM/branch/main/graph/badge.svg)](https://codecov.io/gh/eurunuela/pySPFM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## References

- Caballero-Gaudes, C., Moia, S., Panwar, P., Bandettini, P. A., & Gonzalez-Castillo, J. (2019). A deconvolution algorithm for multi-echo functional MRI: Multi-echo Sparse Paradigm Free Mapping. NeuroImage, 202, 116081–116081. https://doi.org/10.1016/j.neuroimage.2019.116081
- Caballero Gaudes, C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013). Paradigm free mapping with sparse regression automatically detects single-trial functional magnetic resonance imaging blood oxygenation level dependent responses. Human Brain Mapping. https://doi.org/10.1002/hbm.21452
- Gaudes, C. C., Ville, D. V. D., Petridou, N., Lazeyras, F., & Gowland, P. (2011). Paradigm-free mapping with morphological component analysis: Getting most out of fMRI data. Wavelets and Sparsity XIV, 8138, 81381K. https://doi.org/10.1117/12.893920
- Karahanoǧlu, F. I., Caballero-Gaudes, C., Lazeyras, F., & Van De Ville, D. (2013). Total activation: FMRI deconvolution through spatio-temporal regularization. NeuroImage. https://doi.org/10.1016/j.neuroimage.2013.01.067
- Uruñuela, E., Bolton, T. A. W., Van De Ville, D., & Caballero-Gaudes, C. (2021). Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work. ArXiv:2107.12026 [q-Bio]. http://arxiv.org/abs/2107.12026
