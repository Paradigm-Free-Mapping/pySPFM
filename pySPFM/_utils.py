"""Utility functions for pySPFM estimators."""

import numpy as np


def _estimator_repr(estimator, N_CHAR_MAX=700):
    """Build a representation string for an estimator.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to represent.
    N_CHAR_MAX : int, default=700
        Maximum number of characters to display.

    Returns
    -------
    repr_str : str
        The string representation.
    """
    class_name = estimator.__class__.__name__
    params = estimator.get_params(deep=False)

    # Format parameters
    param_strs = []
    for key, value in sorted(params.items()):
        if isinstance(value, str):
            value_str = f"'{value}'"
        elif isinstance(value, float):
            value_str = f"{value:.4g}"
        elif isinstance(value, list | tuple):
            if len(value) > 3:
                value_str = f"[{value[0]}, {value[1]}, ..., {value[-1]}]"
            else:
                value_str = repr(value)
        else:
            value_str = repr(value)
        param_strs.append(f"{key}={value_str}")

    params_str = ", ".join(param_strs)
    repr_str = f"{class_name}({params_str})"

    if len(repr_str) > N_CHAR_MAX:
        repr_str = repr_str[: N_CHAR_MAX - 3] + "..."

    return repr_str


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Compute R² (coefficient of determination) regression score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
            default='uniform_average'
        Defines aggregating of multiple output scores.

    Returns
    -------
    score : float or ndarray of floats
        The R² score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        ss_res = np.sum(sample_weight[:, np.newaxis] * (y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum(
            sample_weight[:, np.newaxis]
            * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2,
            axis=0,
        )
    else:
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

    # Avoid division by zero
    valid = ss_tot != 0
    scores = np.ones(y_true.shape[1])
    scores[valid] = 1 - (ss_res[valid] / ss_tot[valid])
    scores[~valid] = 0.0

    if multioutput == "raw_values":
        return scores
    elif multioutput == "uniform_average":
        return np.mean(scores)
    elif multioutput == "variance_weighted":
        if sample_weight is not None:
            avg_weights = np.sum(
                sample_weight[:, np.newaxis] * (y_true - np.mean(y_true, axis=0)) ** 2, axis=0
            )
        else:
            avg_weights = np.var(y_true, axis=0)
        return np.average(scores, weights=avg_weights)
    else:
        raise ValueError(f"Invalid multioutput value: {multioutput}")


def check_array(array, *, ensure_2d=True, dtype=None, allow_nd=False, copy=False):
    """Input validation on an array.

    Parameters
    ----------
    array : array-like
        Input object to check / convert.
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    dtype : dtype, default=None
        Data type of result. If None, the dtype of the input is preserved.
    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.
    copy : bool, default=False
        Whether to force a copy.

    Returns
    -------
    array_converted : ndarray
        The converted and validated array.
    """
    array = np.asarray(array, dtype=dtype)

    if copy:
        array = array.copy()

    if array.ndim == 1 and ensure_2d:
        array = array.reshape(-1, 1)

    if array.ndim > 2 and not allow_nd:
        raise ValueError(f"Found array with dim {array.ndim}. Expected <= 2.")

    return array


def validate_parameter_constraints(parameter_constraints, params, caller_name):
    """Validate parameters against constraints.

    Parameters
    ----------
    parameter_constraints : dict
        Dictionary mapping parameter names to constraints.
    params : dict
        Dictionary of parameter names and values.
    caller_name : str
        Name of the calling class or function.

    Raises
    ------
    ValueError
        If a parameter doesn't satisfy its constraints.
    """
    for param_name, constraints in parameter_constraints.items():
        if param_name not in params:
            continue

        param_value = params[param_name]

        for constraint in constraints:
            if isinstance(constraint, type):
                if isinstance(param_value, constraint):
                    break
            elif isinstance(constraint, tuple):
                if constraint[0] == "interval":
                    _, dtype, left, right, closed = constraint
                    if closed == "left":
                        valid = left <= param_value < right
                    elif closed == "right":
                        valid = left < param_value <= right
                    elif closed == "both":
                        valid = left <= param_value <= right
                    elif closed == "neither":
                        valid = left < param_value < right
                    if valid:
                        break
            elif callable(constraint):
                if constraint(param_value):
                    break
        else:
            raise ValueError(
                f"The {param_name!r} parameter of {caller_name} must satisfy "
                f"the constraints {constraints}. Got {param_value!r} instead."
            )
