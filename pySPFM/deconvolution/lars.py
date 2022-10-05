"""Least Angle Regression (LARS) method for deconvolution."""
import logging

import numpy as np
from sklearn.linear_model import lars_path

LGR = logging.getLogger("GENERAL")


def select_optimal_lambda(residuals, non_zero_count, n_scans, criterion="bic"):
    """Select optimal lambda based on the model selection criterion (BIC and AIC)

    Parameters
    ----------
    residuals : ndarray
        Residuals of the model
    non_zero_count : ndarray
        Number of non-zero coefficients for each lambda
    n_scans : int
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


def solve_regularization_path(X, y, n_lambdas, criterion="bic"):
    """Solve the regularization path with the LARS algorithm.

    Parameters
    ----------
    X : ndarray
        Design matrix
    y : ndarray
        Voxel time-series
    n_lambdas : int
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
    n_scans = X.shape[1]

    # If y is a vector, add a dimension to make it a matrix
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # Initialize variables to store the results
    coef_path = np.zeros((n_scans, n_lambdas))
    lambdas = np.zeros((n_lambdas,))

    # LARS path
    lambdas_temp, _, coef_path_temp = lars_path(
        X,
        np.squeeze(y),
        method="lasso",
        Gram=np.dot(X.T, X),
        Xy=np.dot(X.T, np.squeeze(y)),
        max_iter=n_lambdas - 1,
        eps=1e-9,
    )

    # Store the results
    coef_path[:, : len(lambdas_temp)] = coef_path_temp
    lambdas[: len(lambdas_temp)] = lambdas_temp

    # Compute residuals for model selection criterion (BIC and AIC)
    residuals = np.sum((np.repeat(y, n_lambdas, axis=-1) - np.dot(X, coef_path)) ** 2, axis=0)

    if criterion == "stability":
        optimal_lambda = lambdas
        coefs = coef_path
    else:
        optimal_lambda_idx = select_optimal_lambda(
            residuals, np.count_nonzero(coef_path, axis=0), n_scans, criterion
        )
        optimal_lambda = lambdas[optimal_lambda_idx]
        coefs = coef_path[:, optimal_lambda_idx]

    return coefs, optimal_lambda
