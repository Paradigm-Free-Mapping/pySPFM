"""Internal solvers for hemodynamic deconvolution.

This module contains low-level solver implementations. These are not part of the
public API and should not be imported directly by users. Instead, use the
estimator classes in `pySPFM.decomposition`.

Modules
-------
fista : FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) solver
lars : LARS (Least Angle Regression) solver with regularization path
hrf_generator : HRF (Hemodynamic Response Function) matrix generation
debiasing : Debiasing methods for sparse estimates
select_lambda : Regularization parameter selection methods
stability_selection : Stability selection for robust feature selection
spatial_regularization : Spatial regularization methods
"""

from pySPFM._solvers.debiasing import debiasing_block, debiasing_spike
from pySPFM._solvers.fista import (
    fista,
    proximal_operator_lasso,
    proximal_operator_mixed_norm,
)
from pySPFM._solvers.hrf_generator import HRFMatrix
from pySPFM._solvers.lars import select_optimal_lambda, solve_regularization_path
from pySPFM._solvers.select_lambda import select_lambda
from pySPFM._solvers.stability_selection import stability_selection

__all__ = [
    # FISTA solver
    "fista",
    "proximal_operator_lasso",
    "proximal_operator_mixed_norm",
    # LARS solver
    "solve_regularization_path",
    "select_optimal_lambda",
    # HRF generation
    "HRFMatrix",
    # Debiasing
    "debiasing_block",
    "debiasing_spike",
    # Lambda selection
    "select_lambda",
    # Stability selection
    "stability_selection",
]
