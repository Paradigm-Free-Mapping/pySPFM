# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""pySPFM: Paradigm Free Mapping for hemodynamic deconvolution of fMRI data.

pySPFM is a Python package for hemodynamic deconvolution of fMRI data using
sparse regularization methods. It provides scikit-learn compatible estimators
for various deconvolution approaches.

Main Estimators
---------------
SparseDeconvolution
    Sparse Paradigm Free Mapping (SPFM) using LARS or FISTA solvers.
LowRankPlusSparse
    Low-rank plus sparse decomposition (SPLORA) for separating global
    and neuronal signals.
StabilitySelection
    Stability selection for robust sparse deconvolution.

Examples
--------
>>> from pySPFM import SparseDeconvolution
>>> import numpy as np
>>> X = np.random.randn(100, 50)  # 100 timepoints, 50 voxels
>>> model = SparseDeconvolution(tr=2.0, criterion='bic')
>>> model.fit(X)
SparseDeconvolution(criterion='bic', tr=2.0)
>>> coef = model.coef_  # Activity-inducing signals

See Also
--------
pySPFM.base : Base classes and utilities.
pySPFM.decomposition : All deconvolution estimators.
"""
import warnings

from pySPFM.__about__ import __copyright__, __credits__, __packagename__, __version__

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

# Import base classes and utilities
from pySPFM.base import (
    BaseEstimator,
    DeconvolutionMixin,
    NotFittedError,
    RegressorMixin,
    TransformerMixin,
    check_is_fitted,
    clone,
)

# Import main estimators for convenient access
from pySPFM.decomposition import (
    LowRankPlusSparse,
    SparseDeconvolution,
    StabilitySelection,
)

__all__ = [
    # Version info
    "__copyright__",
    "__credits__",
    "__packagename__",
    "__version__",
    # Main estimators
    "SparseDeconvolution",
    "LowRankPlusSparse",
    "StabilitySelection",
    # Base classes
    "BaseEstimator",
    "TransformerMixin",
    "RegressorMixin",
    "DeconvolutionMixin",
    # Utilities
    "clone",
    "check_is_fitted",
    "NotFittedError",
]
