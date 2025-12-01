"""Preprocessing utilities for pySPFM CLI.

This module contains functions for data preprocessing operations
such as regressor removal from fMRI data.
"""

import numpy as np


def remove_regressors(data, regressors, n_scans, n_echoes=1):
    """Remove confound regressors from fMRI data using linear regression.

    This function regresses out confound signals from each voxel's time series
    using ordinary least squares. The intercept is preserved, only the
    regressor effects are removed.

    Parameters
    ----------
    data : ndarray of shape (n_timepoints, n_voxels)
        The fMRI data to clean. For multi-echo data, timepoints from different
        echoes are concatenated along the first axis.
    regressors : ndarray of shape (n_scans, n_regressors)
        Confound regressors to remove. Should have the same number of rows as
        scans per echo (not total timepoints for multi-echo).
    n_scans : int
        Number of scans (timepoints) per echo.
    n_echoes : int, optional
        Number of echoes in the data. Default is 1 for single-echo data.

    Returns
    -------
    cleaned_data : ndarray of shape (n_timepoints, n_voxels)
        Data with regressor effects removed.

    Notes
    -----
    The regression model includes an intercept term, but only the regressor
    effects (not the intercept) are subtracted from the data. This preserves
    the mean signal level.

    For multi-echo data, the same regressors are applied to each echo
    independently.

    Examples
    --------
    >>> import numpy as np
    >>> from pySPFM.cli._preprocessing import remove_regressors
    >>> # Single-echo example
    >>> data = np.random.randn(100, 50)  # 100 timepoints, 50 voxels
    >>> regressors = np.random.randn(100, 3)  # 3 confound regressors
    >>> cleaned = remove_regressors(data, regressors, n_scans=100)
    >>> cleaned.shape
    (100, 50)

    >>> # Multi-echo example (3 echoes)
    >>> data_me = np.random.randn(300, 50)  # 3 echoes * 100 scans
    >>> cleaned_me = remove_regressors(
    ...     data_me, regressors, n_scans=100, n_echoes=3
    ... )
    >>> cleaned_me.shape
    (300, 50)
    """
    # Make a copy to avoid modifying the original data
    cleaned_data = data.copy()
    n_voxels = data.shape[1]

    # Ensure regressors is 2D
    if regressors.ndim == 1:
        regressors = regressors.reshape(-1, 1)

    # Build design matrix with intercept
    design_matrix = np.hstack([np.ones((n_scans, 1)), regressors])

    # Regress out confounds for each voxel and echo
    for vox_idx in range(n_voxels):
        for echo_idx in range(n_echoes):
            start_idx = echo_idx * n_scans
            end_idx = (echo_idx + 1) * n_scans
            y = cleaned_data[start_idx:end_idx, vox_idx]
            beta = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
            # Remove only regressor effects (not intercept)
            cleaned_data[start_idx:end_idx, vox_idx] = y - np.dot(design_matrix[:, 1:], beta[1:])

    return cleaned_data
