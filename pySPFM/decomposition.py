"""Sparse hemodynamic deconvolution estimators.

This module contains estimators for sparse deconvolution of fMRI data:

- `SparseDeconvolution`: Sparse Paradigm Free Mapping (SPFM) using LARS or FISTA.
- `LowRankPlusSparse`: Low-Rank plus Sparse decomposition (SPLORA).
- `StabilitySelection`: Stability selection for robust deconvolution.
"""

import logging
from abc import abstractmethod

import numpy as np
from dask import compute
from dask import delayed as delayed_dask

from pySPFM._solvers.debiasing import debiasing_block, debiasing_spike
from pySPFM._solvers.fista import fista
from pySPFM._solvers.hrf_generator import HRFMatrix
from pySPFM._solvers.lars import solve_regularization_path
from pySPFM._solvers.stability_selection import stability_selection
from pySPFM.base import (
    BaseEstimator,
    DeconvolutionMixin,
    TransformerMixin,
    check_is_fitted,
)

LGR = logging.getLogger("GENERAL")

__all__ = [
    "SparseDeconvolution",
    "LowRankPlusSparse",
    "StabilitySelection",
]


class _BaseDeconvolution(DeconvolutionMixin, TransformerMixin, BaseEstimator):
    """Base class for hemodynamic deconvolution estimators.

    This class provides common functionality for all deconvolution estimators.

    Parameters
    ----------
    tr : float
        Repetition time (TR) of the fMRI acquisition in seconds.
    te : list of float, default=None
        Echo times in seconds for multi-echo data. If None or [0], assumes
        single-echo data.
    hrf_model : str, default='spm'
        HRF model to use. Options are 'spm', 'glover', or a path to a custom
        HRF file (.1D or .txt).
    block_model : bool, default=False
        If True, estimate innovation signals (block model).
        If False, estimate activity-inducing signals (spike model).
    n_jobs : int, default=1
        Number of parallel jobs for voxel-wise computation.
    """

    _parameter_constraints = {
        "tr": [float, int],
        "hrf_model": [str],
        "block_model": [bool],
        "n_jobs": [int],
    }

    @abstractmethod
    def __init__(
        self,
        *,
        tr,
        te=None,
        hrf_model="spm",
        block_model=False,
        n_jobs=1,
    ):
        self.tr = tr
        self.te = te if te is not None else [0]
        self.hrf_model = hrf_model
        self.block_model = block_model
        self.n_jobs = n_jobs

    def _generate_hrf_matrix(self, n_scans):
        """Generate the HRF convolution matrix.

        Parameters
        ----------
        n_scans : int
            Number of timepoints.

        Returns
        -------
        hrf_matrix : ndarray of shape (n_scans * n_echoes, n_scans)
            The HRF convolution matrix.
        """
        # Convert TE from ms to s if needed
        te = self.te
        if te is not None and len(te) > 0:
            if all(t >= 1 for t in te if t > 0):
                te = [t / 1000 for t in te]

        hrf_obj = HRFMatrix(te=te, block=self.block_model, model=self.hrf_model)
        hrf_obj.generate_hrf(tr=self.tr, n_scans=n_scans)
        return hrf_obj.hrf_

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the deconvolution model.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels)
            The fMRI timeseries data.
        y : None
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        pass

    def transform(self, X):
        """Apply deconvolution to the data.

        For most deconvolution methods, transform returns the stored
        coefficients from fit, as the deconvolution is data-specific.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels)
            The fMRI timeseries data.

        Returns
        -------
        coef : ndarray of shape (n_timepoints, n_voxels)
            The deconvolved activity-inducing or innovation signals.
        """
        check_is_fitted(self, ["coef_"])
        return self.coef_


class SparseDeconvolution(_BaseDeconvolution):
    """Sparse hemodynamic deconvolution using LARS or FISTA.

    This estimator implements Sparse Paradigm Free Mapping (SPFM) for
    deconvolution of fMRI data. It estimates activity-inducing signals
    (spike model) or innovation signals (block model) from BOLD timeseries.

    Parameters
    ----------
    tr : float
        Repetition time (TR) of the fMRI acquisition in seconds.
    te : list of float, default=None
        Echo times in seconds for multi-echo data. If None or [0], assumes
        single-echo data.
    hrf_model : str, default='spm'
        HRF model to use. Options are 'spm', 'glover', or a path to a custom
        HRF file (.1D or .txt).
    block_model : bool, default=False
        If True, estimate innovation signals (block model).
        If False, estimate activity-inducing signals (spike model).
    criterion : str, default='bic'
        Criterion for lambda selection:

        - 'bic', 'aic': Information criteria (use LARS solver).
        - 'mad', 'mad_update', 'ut', 'lut', 'factor', 'pcg', 'eigval':
          Noise-based criteria (use FISTA solver).

    debias : bool, default=True
        If True, perform debiasing step to recover true amplitude.
    group : float, default=0.0
        Weight for spatial grouping (L2,1-norm). Range [0, 1].
        0 means no grouping (pure L1/LASSO).
    pcg : float, default=0.8
        Percentage of maximum lambda to use (for criterion='pcg').
    factor : float, default=1.0
        Factor to multiply noise estimate for lambda selection.
    max_iter : int, default=400
        Maximum number of iterations for the solver.
    tol : float, default=1e-6
        Convergence tolerance.
    n_jobs : int, default=1
        Number of parallel jobs for voxel-wise computation.
    positive : bool, default=False
        If True, enforce non-negative coefficients.

    Attributes
    ----------
    coef_ : ndarray of shape (n_timepoints, n_voxels)
        Estimated activity-inducing (spike) or innovation (block) signals.
    lambda_ : ndarray of shape (n_voxels,)
        Selected regularization parameter for each voxel.
    hrf_matrix_ : ndarray of shape (n_timepoints * n_echoes, n_timepoints)
        The HRF convolution matrix used for deconvolution.
    n_features_in_ : int
        Number of voxels seen during fit.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from pySPFM import SparseDeconvolution
    >>> import numpy as np
    >>> # Simulate fMRI data (100 timepoints, 50 voxels)
    >>> X = np.random.randn(100, 50)
    >>> # Fit sparse deconvolution
    >>> model = SparseDeconvolution(tr=2.0, criterion='bic')
    >>> model.fit(X)
    SparseDeconvolution(criterion='bic', tr=2.0)
    >>> # Get deconvolved signals
    >>> coef = model.transform(X)
    >>> coef.shape
    (100, 50)

    See Also
    --------
    LowRankPlusSparse : Low-rank plus sparse decomposition.
    StabilitySelection : Stability selection for robust feature selection.

    References
    ----------
    .. [1] Caballero-Gaudes, C., et al. (2013). "Paradigm Free Mapping with
       Sparse Regression Automatically Detects Single-Trial Functional
       Magnetic Resonance Imaging Blood Oxygenation Level Dependent
       Responses." Human Brain Mapping.
    """

    _lars_criteria = ["bic", "aic"]
    _fista_criteria = ["mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"]

    def __init__(
        self,
        *,
        tr,
        te=None,
        hrf_model="spm",
        block_model=False,
        criterion="bic",
        debias=True,
        group=0.0,
        pcg=0.8,
        factor=1.0,
        lambda_echo=-1,
        max_iter=400,
        min_iter=50,
        tol=1e-6,
        n_jobs=1,
        positive=False,
    ):
        super().__init__(
            tr=tr,
            te=te,
            hrf_model=hrf_model,
            block_model=block_model,
            n_jobs=n_jobs,
        )
        self.criterion = criterion
        self.debias = debias
        self.group = group
        self.pcg = pcg
        self.factor = factor
        self.lambda_echo = lambda_echo
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.positive = positive

    def fit(self, X, y=None):
        """Fit the sparse deconvolution model.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels) or \
                (n_timepoints * n_echoes, n_voxels)
            The fMRI timeseries data. For multi-echo data, timepoints from
            different echoes should be concatenated along the first axis.
        y : None
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_te = len(self.te) if self.te[0] != 0 else 1
        n_total, n_voxels = X.shape
        n_scans = n_total // n_te

        self.n_features_in_ = n_voxels
        self.n_samples_ = n_total

        # Generate HRF matrix
        self.hrf_matrix_ = self._generate_hrf_matrix(n_scans)

        # Initialize outputs
        self.coef_ = np.zeros((n_scans, n_voxels))
        self.lambda_ = np.zeros(n_voxels)

        # Solve using LARS or FISTA
        if self.criterion in self._lars_criteria:
            self._fit_lars(X, n_scans, n_voxels)
        elif self.criterion in self._fista_criteria:
            self._fit_fista(X, n_scans, n_voxels)
        else:
            raise ValueError(
                f"Invalid criterion '{self.criterion}'. Must be one of "
                f"{self._lars_criteria + self._fista_criteria}."
            )

        # Perform debiasing
        if self.debias:
            self._debias(X, n_scans)

        return self

    def _fit_lars(self, X, n_scans, n_voxels):
        """Fit using LARS algorithm."""
        n_lambdas = int(np.ceil(n_scans))

        futures = []
        for vox_idx in range(n_voxels):
            fut = delayed_dask(solve_regularization_path, pure=False)(
                self.hrf_matrix_,
                X[:, vox_idx],
                n_lambdas,
                self.criterion,
                False,  # use_fista
                None,  # regressors
            )
            futures.append(fut)

        results = compute(futures, scheduler="synchronous")[0]

        for vox_idx in range(n_voxels):
            self.coef_[:, vox_idx] = np.squeeze(results[vox_idx][0])
            self.lambda_[vox_idx] = np.squeeze(results[vox_idx][1])

    def _fit_fista(self, X, n_scans, n_voxels):
        """Fit using FISTA algorithm."""
        futures = []
        for vox_idx in range(n_voxels):
            fut = delayed_dask(fista, pure=False)(
                self.hrf_matrix_,
                X[:, vox_idx],
                criterion=self.criterion,
                max_iter=self.max_iter,
                min_iter=self.min_iter,
                tol=self.tol,
                group=self.group,
                pcg=self.pcg,
                factor=self.factor,
                lambda_echo=self.lambda_echo,
                positive_only=self.positive,
            )
            futures.append(fut)

        results = compute(futures, scheduler="synchronous")[0]

        for vox_idx in range(n_voxels):
            self.coef_[:, vox_idx] = np.squeeze(results[vox_idx][0])
            self.lambda_[vox_idx] = np.squeeze(results[vox_idx][1])

    def _debias(self, X, n_scans):
        """Perform debiasing step."""
        # For block model, need non-block HRF for debiasing
        if self.block_model:
            te = self.te
            if te is not None and len(te) > 0:
                if all(t >= 1 for t in te if t > 0):
                    te = [t / 1000 for t in te]
            hrf_obj = HRFMatrix(te=te, block=False, model=self.hrf_model)
            hrf_obj.generate_hrf(tr=self.tr, n_scans=n_scans)
            hrf_debias = hrf_obj.hrf_

            self.coef_ = debiasing_block(hrf=hrf_debias, y=X, estimates_matrix=self.coef_)
        else:
            self.coef_, _ = debiasing_spike(
                self.hrf_matrix_, X, self.coef_, non_negative=self.positive
            )


class LowRankPlusSparse(_BaseDeconvolution):
    """Low-rank plus sparse deconvolution (SPLORA).

    This estimator implements the Sparse and Low-Rank Paradigm Free Mapping
    (SPLORA) algorithm, which separates fMRI data into low-rank (global/
    structured) and sparse (transient neuronal) components.

    Parameters
    ----------
    tr : float
        Repetition time (TR) of the fMRI acquisition in seconds.
    te : list of float, default=None
        Echo times in seconds for multi-echo data. If None or [0], assumes
        single-echo data.
    hrf_model : str, default='spm'
        HRF model to use. Options are 'spm', 'glover', or a path to a custom
        HRF file (.1D or .txt).
    block_model : bool, default=False
        If True, estimate innovation signals (block model).
        If False, estimate activity-inducing signals (spike model).
    criterion : str, default='mad_update'
        Criterion for lambda selection. Options: 'mad', 'mad_update', 'ut',
        'lut', 'factor', 'pcg', 'eigval'.
    eigval_threshold : float, default=0.1
        Eigenvalue threshold for low-rank estimation. Eigenvalues below
        this fraction of the maximum are set to zero.
    debias : bool, default=True
        If True, perform debiasing step to recover true amplitude.
    group : float, default=0.0
        Weight for spatial grouping (L2,1-norm). Range [0, 1].
    max_iter : int, default=100
        Maximum number of FISTA iterations.
    n_jobs : int, default=1
        Number of parallel jobs for voxel-wise computation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_timepoints, n_voxels)
        Estimated sparse (neuronal) activity-inducing signals.
    low_rank_ : ndarray of shape (n_timepoints, n_voxels)
        Estimated low-rank (global/structured) component.
    lambda_ : ndarray of shape (n_voxels,)
        Selected regularization parameter for each voxel.
    hrf_matrix_ : ndarray of shape (n_timepoints * n_echoes, n_timepoints)
        The HRF convolution matrix used for deconvolution.
    n_features_in_ : int
        Number of voxels seen during fit.

    Examples
    --------
    >>> from pySPFM import LowRankPlusSparse
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)  # 100 timepoints, 50 voxels
    >>> model = LowRankPlusSparse(tr=2.0)
    >>> model.fit(X)
    LowRankPlusSparse(tr=2.0)
    >>> sparse_signal = model.coef_
    >>> global_signal = model.low_rank_

    See Also
    --------
    SparseDeconvolution : Standard sparse deconvolution without low-rank.

    References
    ----------
    .. [1] Uruñuela, E., et al. (2021). "A low rank and sparse paradigm free
       mapping algorithm for EEG-informed fMRI." IEEE ISBI.
    """

    def __init__(
        self,
        *,
        tr,
        te=None,
        hrf_model="spm",
        block_model=False,
        criterion="mad_update",
        eigval_threshold=0.1,
        debias=True,
        group=0.0,
        factor=1.0,
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        n_jobs=1,
    ):
        super().__init__(
            tr=tr,
            te=te,
            hrf_model=hrf_model,
            block_model=block_model,
            n_jobs=n_jobs,
        )
        self.criterion = criterion
        self.eigval_threshold = eigval_threshold
        self.debias = debias
        self.group = group
        self.factor = factor
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol

    def fit(self, X, y=None):
        """Fit the low-rank plus sparse deconvolution model.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels) or \
                (n_timepoints * n_echoes, n_voxels)
            The fMRI timeseries data.
        y : None
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        from scipy.linalg import svd

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_te = len(self.te) if self.te[0] != 0 else 1
        n_total, n_voxels = X.shape
        n_scans = n_total // n_te

        self.n_features_in_ = n_voxels
        self.n_samples_ = n_total

        # Generate HRF matrix
        self.hrf_matrix_ = self._generate_hrf_matrix(n_scans)

        # Initialize components
        self.coef_ = np.zeros((n_scans, n_voxels))
        self.low_rank_ = np.zeros((n_total, n_voxels))
        self.lambda_ = np.zeros(n_voxels)

        # Iterative low-rank + sparse decomposition
        residual = X.copy()
        coef_old = np.zeros((n_scans, n_voxels))

        for iter_idx in range(self.max_iter):
            # Step 1: Estimate low-rank component from residual
            U, s, Vt = svd(residual, full_matrices=False)

            # Threshold eigenvalues
            threshold = self.eigval_threshold * s[0]
            s_thresholded = np.where(s > threshold, s, 0)

            self.low_rank_ = U @ np.diag(s_thresholded) @ Vt

            # Step 2: Estimate sparse component from (X - low_rank)
            X_sparse = X - self.low_rank_

            futures = []
            for vox_idx in range(n_voxels):
                fut = delayed_dask(fista, pure=False)(
                    self.hrf_matrix_,
                    X_sparse[:, vox_idx],
                    criterion=self.criterion,
                    max_iter=50,
                    min_iter=self.min_iter,
                    tol=self.tol,
                    group=self.group,
                    factor=self.factor,
                )
                futures.append(fut)

            results = compute(futures, scheduler="synchronous")[0]

            for vox_idx in range(n_voxels):
                self.coef_[:, vox_idx] = np.squeeze(results[vox_idx][0])
                self.lambda_[vox_idx] = np.squeeze(results[vox_idx][1])

            # Update residual for next iteration
            fitted_sparse = np.dot(self.hrf_matrix_, self.coef_)
            residual = X - fitted_sparse

            # Check convergence (simplified)
            if iter_idx > 0:
                change = np.linalg.norm(self.coef_ - coef_old) / (np.linalg.norm(coef_old) + 1e-10)
                if change < self.tol:
                    break

            coef_old = self.coef_.copy()

        self.n_iter_ = iter_idx + 1

        # Perform debiasing
        if self.debias:
            if self.block_model:
                te = self.te
                if te is not None and len(te) > 0:
                    if all(t >= 1 for t in te if t > 0):
                        te = [t / 1000 for t in te]
                hrf_obj = HRFMatrix(te=te, block=False, model=self.hrf_model)
                hrf_obj.generate_hrf(tr=self.tr, n_scans=n_scans)
                hrf_debias = hrf_obj.hrf_

                self.coef_ = debiasing_block(
                    hrf=hrf_debias, y=X - self.low_rank_, estimates_matrix=self.coef_
                )
            else:
                self.coef_, _ = debiasing_spike(self.hrf_matrix_, X - self.low_rank_, self.coef_)

        return self

    def get_fitted_signal(self):
        """Get the full fitted signal (low-rank + HRF*sparse).

        Returns
        -------
        fitted : ndarray of shape (n_timepoints, n_voxels)
            The full reconstructed signal.
        """
        check_is_fitted(self, ["coef_", "low_rank_", "hrf_matrix_"])
        return self.low_rank_ + np.dot(self.hrf_matrix_, self.coef_)


class StabilitySelection(_BaseDeconvolution):
    """Stability selection for robust sparse deconvolution.

    This estimator uses stability selection to identify robust features
    in the sparse deconvolution problem. It runs multiple deconvolutions
    on subsampled data and returns the selection frequency (AUC) for each
    timepoint.

    Parameters
    ----------
    tr : float
        Repetition time (TR) of the fMRI acquisition in seconds.
    te : list of float, default=None
        Echo times in seconds for multi-echo data.
    hrf_model : str, default='spm'
        HRF model to use.
    block_model : bool, default=False
        If True, estimate innovation signals (block model).
    n_surrogates : int, default=50
        Number of bootstrap surrogates.
    n_lambdas : int, default=None
        Number of lambda values in the regularization path.
        If None, uses n_scans.
    threshold : float, default=0.6
        Selection threshold for considering a feature as selected.
    n_jobs : int, default=1
        Number of parallel jobs.

    Attributes
    ----------
    selection_frequency_ : ndarray of shape (n_timepoints, n_voxels)
        Selection frequency (AUC) for each timepoint and voxel.
    coef_ : ndarray of shape (n_timepoints, n_voxels)
        Binary selection indicators based on threshold.
    hrf_matrix_ : ndarray
        The HRF convolution matrix.
    n_features_in_ : int
        Number of voxels.

    Examples
    --------
    >>> from pySPFM import StabilitySelection
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> model = StabilitySelection(tr=2.0, n_surrogates=20)
    >>> model.fit(X)
    StabilitySelection(n_surrogates=20, tr=2.0)
    >>> selection_freq = model.selection_frequency_

    See Also
    --------
    SparseDeconvolution : Standard sparse deconvolution.

    References
    ----------
    .. [1] Meinshausen, N. and Bühlmann, P. (2010). "Stability selection."
       Journal of the Royal Statistical Society: Series B.
    """

    def __init__(
        self,
        *,
        tr,
        te=None,
        hrf_model="spm",
        block_model=False,
        n_surrogates=50,
        n_lambdas=None,
        threshold=0.6,
        n_jobs=1,
    ):
        super().__init__(
            tr=tr,
            te=te,
            hrf_model=hrf_model,
            block_model=block_model,
            n_jobs=n_jobs,
        )
        self.n_surrogates = n_surrogates
        self.n_lambdas = n_lambdas
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit the stability selection model.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels)
            The fMRI timeseries data.
        y : None
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_te = len(self.te) if self.te[0] != 0 else 1
        n_total, n_voxels = X.shape
        n_scans = n_total // n_te

        self.n_features_in_ = n_voxels
        self.n_samples_ = n_total

        # Generate HRF matrix
        self.hrf_matrix_ = self._generate_hrf_matrix(n_scans)

        n_lambdas = self.n_lambdas if self.n_lambdas is not None else n_scans

        # Initialize selection frequencies
        self.selection_frequency_ = np.zeros((n_scans, n_voxels))

        # Run stability selection for each voxel
        futures = []
        for vox_idx in range(n_voxels):
            fut = delayed_dask(stability_selection)(
                self.hrf_matrix_,
                X[:, vox_idx],
                n_lambdas,
                self.n_surrogates,
            )
            futures.append(fut)

        results = compute(futures, scheduler="synchronous")[0]

        for vox_idx in range(n_voxels):
            self.selection_frequency_[:, vox_idx] = np.squeeze(results[vox_idx])

        # Threshold to get binary selections
        self.coef_ = (self.selection_frequency_ >= self.threshold).astype(float)

        return self

    def transform(self, X):
        """Return selection frequencies.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels)
            The fMRI timeseries data (not used, returns stored frequencies).

        Returns
        -------
        selection_frequency : ndarray of shape (n_timepoints, n_voxels)
            Selection frequency (AUC) for each timepoint.
        """
        check_is_fitted(self, ["selection_frequency_"])
        return self.selection_frequency_
