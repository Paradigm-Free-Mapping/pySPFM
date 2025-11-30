"""Base classes for hemodynamic deconvolution estimators.

This module provides the base classes for all estimators in pySPFM,
following the scikit-learn estimator API conventions.

Similar to scikit-learn's base module, this provides:

- `BaseEstimator`: Base class with get_params/set_params.
- `TransformerMixin`: Mixin class providing fit_transform.
- `RegressorMixin`: Mixin class providing score based on R².

References
----------
.. [1] scikit-learn developers. "Developing scikit-learn estimators."
   https://scikit-learn.org/stable/developers/develop.html
"""

import logging
import warnings
from abc import ABCMeta, abstractmethod
from inspect import signature

import numpy as np

LGR = logging.getLogger("GENERAL")

__all__ = [
    "BaseEstimator",
    "TransformerMixin",
    "RegressorMixin",
    "DeconvolutionMixin",
    "clone",
    "check_is_fitted",
    "NotFittedError",
]


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and maintain compatibility with scikit-learn.

    Examples
    --------
    >>> from pySPFM.base import NotFittedError
    >>> from pySPFM import SparseDeconvolution
    >>> try:
    ...     SparseDeconvolution().transform([[1, 2], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This SparseDeconvolution instance is not fitted yet...")
    """

    pass


def clone(estimator, *, safe=True):
    """Construct a new unfitted estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It returns a new estimator
    with the same parameters that has not been fitted on any data.

    Parameters
    ----------
    estimator : estimator object
        The estimator to be cloned.
    safe : bool, default=True
        If safe is False, clone will fall back to a deep copy on objects
        that are not estimators. Currently not implemented.

    Returns
    -------
    estimator : estimator object
        The deep copy of the input estimator.

    Examples
    --------
    >>> from pySPFM import SparseDeconvolution
    >>> from pySPFM.base import clone
    >>> estimator = SparseDeconvolution(tr=2.0, criterion='bic')
    >>> cloned = clone(estimator)
    >>> estimator is cloned
    False
    >>> estimator.get_params() == cloned.get_params()
    True
    """
    klass = estimator.__class__
    params = estimator.get_params(deep=False)
    new_estimator = klass(**params)

    return new_estimator


def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.
    attributes : str or list of str, default=None
        Attribute name(s) given as string or a list of strings.
        If None, attributes is set to ``[v for v in vars(estimator)
        if v.endswith("_") and not v.startswith("__")]``.
    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
    all_or_any : callable, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is not an estimator instance.
    NotFittedError
        If the attributes are not found.

    Examples
    --------
    >>> from pySPFM.base import check_is_fitted, NotFittedError
    >>> from pySPFM import SparseDeconvolution
    >>> estimator = SparseDeconvolution(tr=2.0)
    >>> try:
    ...     check_is_fitted(estimator)
    ... except NotFittedError:
    ...     print("Not fitted!")
    Not fitted!
    """
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator} is not an estimator instance.")

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]
        fitted = len(attrs) > 0

    if not fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})


class BaseEstimator:
    """Base class for all estimators in pySPFM.

    All estimators should specify all the parameters that can be set at the
    class level in their ``__init__`` as explicit keyword arguments
    (no ``*args`` or ``**kwargs``).

    Notes
    -----
    All estimators should implement:

    - ``fit(X, y=None)``: Fit the model to the data.
    - ``get_params(deep=True)``: Get parameters for this estimator.
    - ``set_params(**params)``: Set the parameters of this estimator.

    Attributes set during fit should end with an underscore (e.g., ``coef_``).

    Examples
    --------
    >>> from pySPFM.base import BaseEstimator
    >>> class MyEstimator(BaseEstimator):
    ...     def __init__(self, *, param1=1, param2='default'):
    ...         self.param1 = param1
    ...         self.param2 = param2
    ...     def fit(self, X, y=None):
    ...         self.fitted_ = True
    ...         return self
    >>> est = MyEstimator(param1=5)
    >>> est.get_params()
    {'param1': 5, 'param2': 'default'}
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        sig = signature(init)
        parameters = [
            p for p in sig.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"pySPFM estimators should always specify their parameters "
                    f"in the signature of their __init__ (no varargs). "
                    f"{cls} with constructor {sig} doesn't follow this convention."
                )
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params = {}

        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {list(self._get_param_names())!r}."
                )

            if delim:
                nested_params.setdefault(key, {})[sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            getattr(self, key).set_params(**sub_params)

        return self

    def __repr__(self, N_CHAR_MAX=700):
        """Return a string representation of the estimator."""
        from pySPFM._utils import _estimator_repr

        return _estimator_repr(self, N_CHAR_MAX=N_CHAR_MAX)

    def _validate_params(self):
        """Validate types and values of constructor parameters.

        The expected type and values must be defined in the
        ``_parameter_constraints`` class attribute, which is a dictionary
        mapping parameter names to constraints.
        """
        if not hasattr(self, "_parameter_constraints"):
            return

        from pySPFM._utils import validate_parameter_constraints

        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )


class TransformerMixin:
    """Mixin class for all transformers in pySPFM.

    This mixin provides the ``fit_transform`` method.

    If a class inherits from both ``TransformerMixin`` and ``BaseEstimator``,
    the ``fit_transform`` method will call ``fit`` and ``transform`` methods.
    """

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits the estimator to ``X`` and ``y`` with optional parameters
        ``fit_params``, and returns a transformed version of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y, **fit_params).transform(X)


class RegressorMixin:
    """Mixin class for all regressors in pySPFM.

    This mixin provides the ``score`` method based on R² (coefficient of
    determination).
    """

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` w.r.t. ``y``.
        """
        from pySPFM._utils import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class DeconvolutionMixin:
    """Mixin class for hemodynamic deconvolution estimators.

    This mixin provides methods specific to fMRI deconvolution:

    - ``score``: Based on explained variance of the fitted signal.
    - ``get_fitted_signal``: Returns the HRF-convolved estimates.
    - ``get_residuals``: Returns the residuals after deconvolution.
    """

    def score(self, X, y=None):
        """Return the explained variance ratio of the deconvolution.

        The score is computed as :math:`R^2 = 1 - SS_{res} / SS_{tot}`,
        where :math:`SS_{res}` is the residual sum of squares and
        :math:`SS_{tot}` is the total sum of squares.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels)
            Test samples (fMRI timeseries).
        y : None
            Not used, present for API consistency.

        Returns
        -------
        score : float
            Explained variance ratio averaged across voxels.
        """
        check_is_fitted(self, ["coef_", "hrf_matrix_"])

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get the fitted signal
        fitted = self.get_fitted_signal()

        # Compute explained variance per voxel
        ss_res = np.sum((X - fitted) ** 2, axis=0)
        ss_tot = np.sum((X - X.mean(axis=0)) ** 2, axis=0)

        # Avoid division by zero
        valid = ss_tot > 0
        r2 = np.zeros(X.shape[1])
        r2[valid] = 1 - (ss_res[valid] / ss_tot[valid])

        return np.mean(r2)

    def get_fitted_signal(self):
        """Get the fitted signal (HRF convolved with estimates).

        Returns
        -------
        fitted : ndarray of shape (n_timepoints, n_voxels)
            The fitted signal (reconstruction).
        """
        check_is_fitted(self, ["coef_", "hrf_matrix_"])
        return np.dot(self.hrf_matrix_, self.coef_)

    def get_residuals(self, X):
        """Get the residuals after deconvolution.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_voxels)
            The original fMRI data.

        Returns
        -------
        residuals : ndarray of shape (n_timepoints, n_voxels)
            The residuals (X - fitted_signal).
        """
        check_is_fitted(self, ["coef_", "hrf_matrix_"])
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X - self.get_fitted_signal()


def _validate_data(estimator, X, y=None, *, reset=True, **check_params):
    """Validate input data and set n_features_in_.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to validate data for.
    X : array-like of shape (n_samples, n_features)
        The input data.
    y : array-like, default=None
        Target values.
    reset : bool, default=True
        Whether to reset the n_features_in_ attribute.
    **check_params : dict
        Parameters passed to check_array.

    Returns
    -------
    out : ndarray or tuple of ndarray
        The validated data.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    if reset:
        estimator.n_features_in_ = n_features
        estimator.n_samples_ = n_samples

    if y is None:
        return X
    else:
        y = np.asarray(y)
        return X, y
