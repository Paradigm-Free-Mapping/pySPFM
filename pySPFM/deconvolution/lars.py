"""Least Angle Regression (LARS) method for deconvolution."""
import numpy as np
from sklearn.linear_model import lars_path


def select_optimal_lambda(residuals, non_zero_count, n_scans, criterion="bic"):
    """Select optimal lambda based on the model selection criterion (BIC and AIC)

    Parameters
    ----------
    residuals : ndarray
        Residuals of the model
    non_zero_count : ndarray
        Number of non-zero coefficients for each lambda
    nscans : int
        Number of scans
    criterion : str, optional
        Criterion to find the optimal solution, by default "bic"

    Returns
    -------
    index_optimal_lambda : int
        Index of the optimal lambda
    """
    if criterion == "bic":
        # BIC regularization curve
        optimization_curve = n_scans * np.log(residuals) + np.log(n_scans) * non_zero_count
    elif criterion == "aic":
        # AIC regularization curve
        optimization_curve = n_scans * np.log(residuals) + 2 * non_zero_count

    # Optimal lambda is given by the minimum of the optimization curve
    idx_optimal_lambda = np.argmin(optimization_curve)

    return idx_optimal_lambda


def solve_regularization_path(X, y, nlambdas, criterion="bic"):
    """Solve the regularization path with the LARS algorithm.

    Parameters
    ----------
    X : ndarray
        Design matrix
    y : ndarray
        Voxel time-series
    nlambdas : int
        Number of lambdas to be tested
    criterion : str, optional
        Criterion to find the optimal solution, by default "bic"

    Returns
    -------
    coef_path : ndarray
        Estimates of the coefficients for the optimal lambda
    lambdas : ndarray
        Lambda of the optimal solution
    """
    n_scans = y.shape[0]

    # If y is a vector, add a dimension to make it a matrix
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # LARS path
    lambdas, _, coef_path = lars_path(
        X,
        np.squeeze(y),
        method="lasso",
        Gram=np.dot(X.T, X),
        Xy=np.dot(X.T, np.squeeze(y)),
        max_iter=nlambdas - 1,
        eps=1e-9,
    )

    # Compute residuals for model selection criterion (BIC and AIC)
    residuals = np.sum((np.repeat(y, nlambdas, axis=-1) - np.dot(X, coef_path)) ** 2, axis=0)

    optimal_lambda_idx = select_optimal_lambda(
        residuals, np.count_nonzero(coef_path, axis=0), n_scans, criterion
    )

    return coef_path[:, optimal_lambda_idx], lambdas[optimal_lambda_idx]
